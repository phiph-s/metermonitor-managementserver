import datetime
import base64
import json
from io import BytesIO
from PIL import Image
import sqlite3
import urllib.request
import urllib.error
import time

from lib.functions import reevaluate_latest_picture
from lib.model_singleton import get_meter_predictor
from lib.ha_auth import add_ha_auth_header

GLOBAL_CAPTURE_LOCK = []

def capture_from_ha_source(config, source_config):
    """Capture image from HA camera source. Returns (raw_bytes, format)."""
    cam_entity_id = source_config.get('camera_entity_id')
    flash_entity_id = source_config.get('flash_entity_id')
    flash_delay_ms = source_config.get('flash_delay_ms', 10000)

    if cam_entity_id in GLOBAL_CAPTURE_LOCK:
        raise RuntimeError(f"Capture from {cam_entity_id} is already in progress. Multiple simultaneous captures are not supported.")
    GLOBAL_CAPTURE_LOCK.append(cam_entity_id)
    try:
        if not cam_entity_id:
            raise ValueError("No camera_entity_id in source config")

        # HA API functions
        def _ha_request_json_with_method(url, method='GET', body=None):
            full_url = f"{config['homeassistant']['url']}{url}"
            req = urllib.request.Request(full_url, method=method)
            add_ha_auth_header(req, config)
            req.add_header('Content-Type', 'application/json')
            if body:
                req.data = json.dumps(body).encode('utf-8')
            try:
                with urllib.request.urlopen(req) as response:
                    return json.loads(response.read().decode('utf-8'))
            except urllib.error.HTTPError as e:
                raise Exception(f"HA API error {e.code}: {e.read().decode('utf-8')}")
            except Exception as e:
                raise Exception(f"HA API unexpected error: {e}")

        def _ha_get_bytes(url):
            full_url = f"{config['homeassistant']['url']}{url}"
            req = urllib.request.Request(full_url)
            add_ha_auth_header(req, config)
            try:
                with urllib.request.urlopen(req, timeout=30) as response:
                    return response.read()
            except urllib.error.HTTPError as e:
                error_msg = f"HA API error {e.code} on {url}"
                try:
                    error_body = e.read().decode('utf-8')
                    if error_body:
                        error_msg += f": {error_body}"
                except:
                    pass
                if e.code == 500:
                    error_msg += " (Camera may be offline, unreachable, or taking too long to respond)"
                raise Exception(error_msg)
            except urllib.error.URLError as e:
                raise Exception(f"HA API connection error on {url}: {e}")
            except Exception as e:
                raise Exception(f"HA API unexpected error on {url}: {e}")

        flash_enabled = False
        try:
            if flash_entity_id and flash_entity_id.strip():
                _ha_request_json_with_method(
                    "/api/services/light/turn_on",
                    method='POST',
                    body={'entity_id': flash_entity_id},
                )
                flash_enabled = True
                time.sleep(max(0.0, flash_delay_ms / 1000.0))

            raw = _ha_get_bytes(f"/api/camera_proxy/{cam_entity_id}")
            return raw, "jpeg", flash_enabled
        finally:
            if flash_enabled:
                try:
                    _ha_request_json_with_method(
                        "/api/services/light/turn_off",
                        method='POST',
                        body={'entity_id': flash_entity_id},
                    )
                except Exception:
                    pass
    finally:
        GLOBAL_CAPTURE_LOCK.remove(cam_entity_id)

def capture_from_http_source(source_config):
    """Capture image from a simple HTTP endpoint. Returns (raw_bytes, format)."""
    url = (source_config or {}).get('url')
    headers = (source_config or {}).get('headers') or {}
    body = (source_config or {}).get('body')

    if not url or not isinstance(url, str):
        raise ValueError("No url in source config")
    if not url.startswith("http://") and not url.startswith("https://"):
        raise ValueError("URL must start with http:// or https://")

    req = urllib.request.Request(url, method='GET')
    if isinstance(headers, dict):
        for key, value in headers.items():
            if key is None:
                continue
            req.add_header(str(key), "" if value is None else str(value))

    if body is not None and body != "":
        if not isinstance(body, str):
            body = json.dumps(body)
        req.data = body.encode('utf-8')
        if "Content-Type" not in req.headers:
            req.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            raw = response.read()
            content_type = response.headers.get('Content-Type', '')
    except urllib.error.HTTPError as e:
        raise Exception(f"HTTP source error {e.code}: {e.read().decode('utf-8', errors='ignore')}")
    except urllib.error.URLError as e:
        raise Exception(f"HTTP source connection error: {e}")
    except Exception as e:
        raise Exception(f"HTTP source unexpected error: {e}")

    fmt = None
    if "png" in content_type.lower():
        fmt = "png"
    elif "jpeg" in content_type.lower() or "jpg" in content_type.lower():
        fmt = "jpeg"
    else:
        try:
            fmt = Image.open(BytesIO(raw)).format.lower()
        except Exception:
            fmt = "jpeg"

    return raw, fmt, False

def process_captured_image(db_file, name, raw_image, format_, config, meter_predictor, publish=True, mqtt_client=None):
    """Process the captured image: save to DB and reevaluate."""
    b64 = base64.b64encode(raw_image).decode('utf-8')
    img = Image.open(BytesIO(raw_image))
    width, height = img.size
    timestamp = datetime.datetime.now().isoformat()

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        # Update or insert into watermeters
        cursor.execute("SELECT 1 FROM watermeters WHERE name = ?", (name,))
        exists = cursor.fetchone()
        meter_is_new = False
        if exists:
            cursor.execute('''
                UPDATE watermeters 
                SET 
                    picture_number = picture_number + 1,
                    wifi_rssi = NULL,
                    picture_format = ?,
                    picture_timestamp = ?,
                    picture_width = ?,
                    picture_height = ?,
                    picture_length = ?,
                    picture_data = ?,
                    picture_data_bbox = NULL
                WHERE name = ?
            ''', (
                format_,
                timestamp,
                width,
                height,
                len(raw_image),
                b64,
                name
            ))
        else:
            cursor.execute('''
                INSERT INTO watermeters (name, picture_number, wifi_rssi, picture_format, picture_timestamp, picture_width, picture_height, picture_length, picture_data, setup, picture_data_bbox)
                VALUES (?,?,?,?,?,?,?,?,?,?,NULL)
            ''', (
                name,
                1,
                None,
                format_,
                timestamp,
                width,
                height,
                len(raw_image),
                b64,
                0
            ))
            # Also insert default settings
            cursor.execute('''
                INSERT OR IGNORE INTO settings
                (name, threshold_low, threshold_high, threshold_last_low, threshold_last_high,
                 islanding_padding, segments, rotated_180, shrink_last_3, extended_last_digit,
                 max_flow_rate, conf_threshold, roi_extractor, template_id, segment_mode, digit_models, use_correctional_alg)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ''', (
                name,
                0,
                125,
                0,
                125,
                20,
                7,
                False,
                False,
                False,
                1.0,
                None,
                "yolo",
                None,
                "display",
                None,
                True
            ))
            meter_is_new = True

        conn.commit()
        print(f"[CAPTURE] Saved image for {name}")

        # If the meter is new and we have an MQTT client, publish HA discovery registration.
        if meter_is_new and mqtt_client is not None:
            try:
                from lib.functions import publish_registration
                publish_registration(mqtt_client, config, name, "value")
                print(f"[CAPTURE] Published MQTT registration for new meter {name}")
            except Exception as e:
                print(f"[CAPTURE] Failed to publish MQTT registration for {name}: {e}")

        # Process the image
        try:
            result = reevaluate_latest_picture(db_file, name, meter_predictor,
                                               config, publish=publish, mqtt_client=mqtt_client)
            if result and len(result) >= 3:
                boundingboxed_image = result[2]
            else:
                boundingboxed_image = None
                if result is None:
                    print(f"[CAPTURE] No bounding box generated for {name} (reevaluate returned None - meter likely not set up yet)")
                else:
                    print(f"[CAPTURE] No bounding box in result for {name}")
        except Exception as e:
            print(f"[CAPTURE] Error processing image for {name}: {e}")
            import traceback
            traceback.print_exc()
            boundingboxed_image = None

        if boundingboxed_image:
            cursor.execute('''
                UPDATE watermeters
                SET picture_data_bbox = ?
                WHERE name = ?
            ''', (
                boundingboxed_image,
                name
            ))
            conn.commit()
            print(f"[CAPTURE] Saved boundingboxed image for {name}")
        else:
            print(f"[CAPTURE] Skipping bounding box save for {name} (no bbox available)")

    return timestamp

def capture_and_process_source(config, db_file, source_row, meter_predictor, mqtt_client=None):
    config_json = source_row['config_json']
    if not config_json:
        return

    try:
        cfg = json.loads(config_json)
        try:
            source_type = source_row['source_type']
        except Exception:
            source_type = source_row.get('source_type') if isinstance(source_row, dict) else None
        if source_type == 'ha_camera':
            raw_image, format_, _ = capture_from_ha_source(config, cfg)
        elif source_type == 'http':
            raw_image, format_, _ = capture_from_http_source(cfg)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
        timestamp = process_captured_image(db_file, source_row['name'], raw_image, format_, config, meter_predictor, publish=True, mqtt_client=mqtt_client)

        # Update source last_success_ts
        with sqlite3.connect(db_file) as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE sources SET last_success_ts = ?, last_error = NULL WHERE id = ?", (timestamp, source_row['id']))
            conn.commit()
        print(f"[CAPTURE] Successfully captured and processed source {source_row['name']}")
        return timestamp
    except Exception as e:
        error_msg = str(e)
        print(f"[CAPTURE] Failed to capture source {source_row['name']}: {error_msg}")
        # Update source with error, and only set last_success_ts when it is missing
        with sqlite3.connect(db_file) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT last_success_ts FROM sources WHERE id = ?", (source_row['id'],))
            row = cursor.fetchone()
            last_success_ts = row[0] if row else None
            if last_success_ts:
                cursor.execute(
                    "UPDATE sources SET last_error = ? WHERE id = ?",
                    (error_msg, source_row['id'])
                )
            else:
                now = datetime.datetime.now().isoformat()
                cursor.execute(
                    "UPDATE sources SET last_error = ?, last_success_ts = ? WHERE id = ?",
                    (error_msg, now, source_row['id'])
                )
            conn.commit()
        raise
