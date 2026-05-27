from lib.log import log
import datetime
import threading
import time
import base64

import paho.mqtt.client as mqtt
import json
import sqlite3
from typing import Dict, Any
from io import BytesIO
from PIL import Image

from lib.functions import reevaluate_latest_picture, publish_registration
from lib.model_singleton import get_meter_predictor
from lib.capture_utils import make_thumbnail
import traceback

from lib.global_alerts import add_alert, remove_alert

class MQTTHandler:

    def __init__(self,config, db_file: str = 'watermeters.db', forever: bool = False):
        self.db_file = db_file
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.config = config
        self.forever = forever
        self.should_reconnect = True
        self.topic = None
        self._reconnect_in_progress = False
        # Used to signal _reconnect() loop when _on_connect fires
        self._connect_result = threading.Event()
        self._connect_success = False
        # Capture waiters: meter_name -> list of threading.Event
        self._capture_waiters: Dict[str, list] = {}
        self._capture_waiters_lock = threading.Lock()
        # Use singleton instance (shared with HTTP server)
        self.meter_preditor = get_meter_predictor()
        log("[MQTT] Using shared meter predictor singleton instance.")

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            log("[MQTT] Successfully connected to MQTT broker")
            self._connect_success = True
            self._connect_result.set()
            remove_alert("mqtt")
        else:
            code_str = str(reason_code)
            log(f"[MQTT] Connection rejected: {reason_code}")
            add_alert("mqtt", f"MQTT connection rejected: {code_str}")
            self._connect_success = False
            self._connect_result.set()
            return

        # Re-subscribe on every connect (handles reconnect after disconnect)
        if self.topic:
            self.client.subscribe(self.topic)
            log(f"[MQTT] Subscribed to topic '{self.topic}'")

        # send registration message for all watermeters
        with sqlite3.connect(self.db_file, timeout=30) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name, COALESCE(meter_type, 'WATER'), unit FROM watermeters"
            )
            rows = cursor.fetchall()
            for row in rows:
                publish_registration(self.client, self.config, row[0], "value",
                                     meter_type=row[1] or 'WATER', unit=row[2])

    def _on_disconnect(self, client, userdata, rc, properties=None, packet=None, reason=None):
        log(f"Disconnected with code {rc}")
        if not self.should_reconnect:
            return
        add_alert("mqtt", "MQTT connection lost")
        # Guard against double reconnect loop (on_connect failure also triggers disconnect)
        if not self._reconnect_in_progress:
            self._reconnect_in_progress = True
            t = threading.Thread(target=self._reconnect, daemon=True)
            t.start()

    def _reconnect(self):
        """Reconnect loop with exponential backoff.
        Waits for _on_connect to confirm auth success before exiting.
        Only removes the alert when the connection is truly established."""
        delay = 1
        max_delay = 60

        while self.should_reconnect:
            try:
                log("[MQTT] Reconnecting to MQTT broker...")
                self._connect_result.clear()
                self._connect_success = False
                self.client.reconnect()
                # Wait for _on_connect – TCP success alone is not enough
                got_result = self._connect_result.wait(timeout=15)
                if self._connect_success:
                    self._reconnect_in_progress = False
                    return
                # Auth rejected or no CONNACK within timeout
                reason = "connection rejected" if got_result else "broker unreachable or not responding"
                add_alert("mqtt", f"MQTT error: {reason} – retrying in {delay}s")
                log(f"[MQTT] {reason}, retrying in {delay}s...")
                time.sleep(delay)
                delay = min(delay * 2, max_delay)
            except Exception as e:
                add_alert("mqtt", f"MQTT error: {e}")
                log(f"[MQTT] Reconnect failed: {e}, retrying in {delay}s...")
                time.sleep(delay)
                delay = min(delay * 2, max_delay)

        self._reconnect_in_progress = False

    # Validate the incoming message
    def _on_message(self, client, userdata, msg):
        data = json.loads(msg.payload)
        self._process_message(data, msg.topic)

    def _validate_message(self, data: Dict[str, Any]) -> bool:
        # Required top-level fields
        required_fields = {'name', 'picture'}
        if not all(field in data for field in required_fields):
            return False

        # Required fields in picture
        required_picture_fields = {'data'}

        if not isinstance(data['picture'], dict):
            return False

        if not all(field in data['picture'] for field in required_picture_fields):
            return False

        return True

    @staticmethod
    def _parse_timestamp(raw_timestamp):
        if raw_timestamp is None or raw_timestamp == "" or raw_timestamp == "0":
            return datetime.datetime.now().isoformat()

        # Numeric unix timestamp (seconds or ms)
        if isinstance(raw_timestamp, (int, float)):
            ts = float(raw_timestamp)
            if ts > 1e12:
                ts /= 1000.0
            return datetime.datetime.fromtimestamp(ts).isoformat()

        if isinstance(raw_timestamp, str):
            value = raw_timestamp.strip()
            if not value:
                return datetime.datetime.now().isoformat()

            # Numeric string unix timestamp (seconds or ms)
            if value.isdigit():
                ts = float(value)
                if ts > 1e12:
                    ts /= 1000.0
                return datetime.datetime.fromtimestamp(ts).isoformat()

            # ISO timestamp — no timezone → treat as local time;
            # Z or offset → convert to local time
            try:
                dt = datetime.datetime.fromisoformat(value)
                if dt.tzinfo is not None:
                    dt = dt.astimezone().replace(tzinfo=None)
                return dt.isoformat()
            except Exception:
                return datetime.datetime.now().isoformat()

        return datetime.datetime.now().isoformat()

    @staticmethod
    def _decode_picture_data(raw_data):
        if not isinstance(raw_data, str):
            raise ValueError("picture.data must be a base64 string")

        data = raw_data.strip()
        fmt_from_prefix = None

        # Accept data URI format: data:image/png;base64,....
        if data.startswith("data:"):
            header, _, payload = data.partition(",")
            data = payload
            if "/" in header and ";" in header:
                try:
                    fmt_from_prefix = header.split("/")[1].split(";")[0].lower()
                except Exception:
                    fmt_from_prefix = None

        decoded = base64.b64decode(data, validate=True)
        canonical_b64 = base64.b64encode(decoded).decode("utf-8")
        return decoded, canonical_b64, fmt_from_prefix

    @staticmethod
    def _detect_image_meta(image_bytes, fmt_hint=None):
        with Image.open(BytesIO(image_bytes)) as img:
            width, height = img.size
            fmt = (fmt_hint or img.format or "jpeg").lower()
        if fmt == "jpg":
            fmt = "jpeg"
        return fmt, width, height

    # Process the incoming message
    def _process_message(self, data: Dict[str, Any], mqtt_topic: str = None):
        try:
            if not self._validate_message(data):
                log(f"[MQTT] Invalid message format received at {datetime.datetime.now().isoformat()}: {data}")
                return

            log(f"[MQTT] Received message for watermeter {data['name']}")


            timestamp = self._parse_timestamp(data['picture'].get('timestamp'))
            image_bytes, picture_data_b64, fmt_from_prefix = self._decode_picture_data(data['picture']['data'])
            picture_format, picture_width, picture_height = self._detect_image_meta(
                image_bytes,
                fmt_hint=(data['picture'].get('format') or fmt_from_prefix)
            )
            picture_length = len(image_bytes)
            picture_thumbnail = make_thumbnail(image_bytes, picture_format)
            wifi_rssi = data.get('WiFi-RSSI')

            with sqlite3.connect(self.db_file, timeout=30) as conn:

                cursor = conn.cursor()
                #check if watermeter exists
                cursor.execute("SELECT picture_number FROM watermeters WHERE name = ?", (data['name'],))
                row = cursor.fetchone()
                meter_exists = row is not None

                capabilities_raw = data.get('capabilities')
                capabilities_json = json.dumps(capabilities_raw) if capabilities_raw is not None else None

                if not meter_exists:
                    cursor.execute('''
                        INSERT INTO watermeters (name, picture_number, wifi_rssi, picture_format, picture_timestamp, picture_width, picture_height, picture_length, picture_data, picture_thumbnail, setup, picture_data_bbox, capabilities)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,NULL,?)
                    ''', (
                        data['name'],
                        1,
                        wifi_rssi,
                        picture_format,
                        timestamp,
                        picture_width,
                        picture_height,
                        picture_length,
                        picture_data_b64,
                        picture_thumbnail,
                        0,
                        capabilities_json,
                    ))
                    cursor.execute('''
                                    INSERT OR IGNORE INTO settings
                                    (name, threshold_low, threshold_high, threshold_last_low, threshold_last_high,
                                     islanding_padding, segments, rotated_180, shrink_last_3, extended_last_digit,
                                     max_flow_rate, conf_threshold, roi_extractor, template_id, segment_mode, digit_models, decimals, use_correctional_alg)
                                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                                ''', (
                        data['name'],
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
                        3,
                        True
                    ))

                    publish_registration(self.client, self.config, data['name'], "value",
                                         meter_type='WATER', unit=None)
                else:
                    next_picture_number = int(row[0] or 0) + 1
                    cursor.execute('''
                            UPDATE watermeters
                            SET
                                picture_number = ?,
                                wifi_rssi = ?,
                                picture_format = ?,
                                picture_timestamp = ?,
                                picture_width = ?,
                                picture_height = ?,
                                picture_length = ?,
                                picture_data = ?,
                                picture_thumbnail = ?,
                                picture_data_bbox = NULL
                            WHERE name = ?
                        ''', (
                        next_picture_number,
                        wifi_rssi,
                        picture_format,
                        timestamp,
                        picture_width,
                        picture_height,
                        picture_length,
                        picture_data_b64,
                        picture_thumbnail,
                        data['name']
                    ))

                # Update capabilities if the device reported them
                if capabilities_json is not None:
                    cursor.execute(
                        "UPDATE watermeters SET capabilities = ? WHERE name = ?",
                        (capabilities_json, data['name']),
                    )

                # Ensure MQTT source entry exists and keep mqtt_topic up to date
                cursor.execute(
                    "SELECT config_json FROM sources WHERE name = ? AND source_type = 'mqtt'",
                    (data['name'],),
                )
                src_row = cursor.fetchone()
                if src_row is None:
                    src_cfg = json.dumps({"mqtt_topic": mqtt_topic}) if mqtt_topic else None
                    cursor.execute(
                        "INSERT INTO sources (name, source_type, enabled, poll_interval_s, config_json, updated_ts) "
                        "VALUES (?, 'mqtt', 1, NULL, ?, datetime('now'))",
                        (data['name'], src_cfg),
                    )
                elif mqtt_topic:
                    try:
                        src_cfg = json.loads(src_row[0]) if src_row[0] else {}
                    except Exception:
                        src_cfg = {}
                    src_cfg["mqtt_topic"] = mqtt_topic
                    cursor.execute(
                        "UPDATE sources SET config_json = ? WHERE name = ? AND source_type = 'mqtt'",
                        (json.dumps(src_cfg), data['name']),
                    )
                conn.commit()
                
                # check if source for watermeter exists and is enabled
                cursor.execute("SELECT enabled FROM sources WHERE name = ? AND source_type = 'mqtt'", (data['name'],))
                result = cursor.fetchone()

                if result is None or not result[0]:
                    log(f"[MQTT] Source for watermeter {data['name']} is disabled or does not exist")
                    return
                
                log(f"[MQTT] Saved/updated metadata of {data['name']} to database.")
                _, _, boundingboxed_image = reevaluate_latest_picture(self.db_file, data['name'], self.meter_preditor,
                                                                      self.config, publish=True,
                                                                      mqtt_client=self.client,
                                                                      notify_realtime=True)
                # Insert boundingboxed image into database
                if boundingboxed_image:
                    cursor.execute('''
                        UPDATE watermeters
                        SET picture_data_bbox = ?
                        WHERE name = ?
                    ''', (
                        boundingboxed_image,
                        data['name']
                    ))
                    conn.commit()
                    log(f"[MQTT] Saved boundingboxed image of {data['name']} to database.")

                self._notify_capture_waiters(data['name'])

        except Exception as e:
            log(f"[MQTT] Error processing message: {e}")
            # print traceback
            traceback.print_exc()

    # Start the MQTT client
    def start(self,
              broker: str = 'localhost',
              port: int = 1883,
              topic: str = "MeterMonitor/#",
              username: str = None,
              password: str = None):

        self.topic = topic
        add_alert("mqtt", "Connecting to MQTT broker")

        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect

        if username and password:
            self.client.username_pw_set(username, password)

        try:
            self.client.connect(broker, port)
        except Exception as e:
            log(f"[MQTT] Error connecting to MQTT broker: {e}")
            add_alert("mqtt", f"Failed to connect to MQTT broker: {e}")
            return
        # subscribe is handled in _on_connect to also cover reconnects
        if self.forever:
            self.client.loop_forever()
        else:
            self.client.loop_start()

    # ── capture waiters ───────────────────────────────────────────────────────────

    def register_capture_waiter(self, name: str) -> threading.Event:
        evt = threading.Event()
        with self._capture_waiters_lock:
            self._capture_waiters.setdefault(name, []).append(evt)
        return evt

    def unregister_capture_waiter(self, name: str, evt: threading.Event) -> None:
        with self._capture_waiters_lock:
            waiters = self._capture_waiters.get(name, [])
            try:
                waiters.remove(evt)
            except ValueError:
                pass

    def _notify_capture_waiters(self, name: str) -> None:
        with self._capture_waiters_lock:
            for evt in self._capture_waiters.get(name, []):
                evt.set()

    def stop(self):
        self.should_reconnect = False
        self._connect_result.set()  # unblock any waiting _reconnect() thread immediately
        self.client.loop_stop()
        try:
            self.client.disconnect()
        except Exception:
            pass
