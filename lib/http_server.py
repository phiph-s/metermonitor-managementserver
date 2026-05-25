from lib.log import log
import json
import os
import asyncio
from io import BytesIO
import shutil
import tempfile


import numpy as np
from PIL import Image
import cv2
from fastapi import FastAPI, HTTPException, Body, Header, Depends, WebSocket, WebSocketDisconnect, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, conint
import base64
import sqlite3
import zlib
import re
from typing import List, Optional

import urllib.request
import urllib.error
import socket
import time
import uuid
from datetime import datetime

from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, FileResponse, StreamingResponse

from lib.functions import reevaluate_latest_picture, add_history_entry, reevaluate_digits, publish_registration, compute_daily_avg
from lib.ha_flash_suggestion import suggest_flash_entity
from lib.flash_handler import (
    clone_or_pull_firmware,
    firmware_exists,
    get_kconfig_options,
    read_current_sdkconfig,
    build_firmware_stream,
    get_flash_args,
    FIRMWARE_DIR,
)
from lib.model_singleton import get_meter_predictor
from lib.global_alerts import get_alerts, add_alert, remove_alert, set_change_callback
from lib.ha_auth import get_ha_token, add_ha_auth_header
from lib.threshold_optimizer import search_thresholds_for_meter
from lib.capture_utils import capture_and_process_source, capture_from_ha_source, capture_from_http_source
from lib.meter_processing.roi_extractors.orb_extractor import ORBExtractor
from lib.meter_processing.roi_extractors.static_rect_extractor import StaticRectExtractor
from lib.realtime_events import evaluation_event_hub


# http server class
# FastAPOI automatically creates a documentation for the API on the path /docs

def prepare_setup_app(config, lifespan, restart_mqtt_fn=None, get_mqtt_client_fn=None):
    app = FastAPI(lifespan=lifespan)
    SECRET_KEY = config['secret_key']
    db_connection = lambda: sqlite3.connect(config['dbfile'])

    # Push alert changes over WebSocket
    set_change_callback(lambda current_alerts: evaluation_event_hub.notify({"type": "alerts_updated", "alerts": current_alerts}))

    # Warn user if secret key is not changed
    if config['secret_key'] == "change_me" and config['enable_auth']:
        add_alert("authentication", "Please change the secret key in the configuration file!")

    # Get singleton instance of meter predictor (shared with MQTT handler)
    meter_preditor = get_meter_predictor()

    log("[HTTP] Using shared meter predictor singleton instance.")

    # CORS Konfiguration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def register_realtime_loop():
        evaluation_event_hub.set_loop(asyncio.get_running_loop())

    # When using home assistant ingress, restrict access to the ingress IP
    @app.middleware("http")
    async def restrict_ip_middleware(request, call_next):
        client_ip = request.client.host  # Get the requester's IP address
        if config["ingress"] and client_ip != "172.30.32.2": # Home Assistant IP for Ingress
            return JSONResponse(status_code=403, content={"message": "Forbidden"})

        return await call_next(request)

    # Authentication
    def authenticate(secret: str = Header(None)):
        if not config['enable_auth']:
            return
        if secret != SECRET_KEY:
            raise HTTPException(status_code=401, detail="Unauthorized")

    @app.websocket("/api/ws/evaluations")
    async def websocket_evaluations(websocket: WebSocket, secret: Optional[str] = Query(None)):
        evaluation_event_hub.set_loop(asyncio.get_running_loop())
        if config['enable_auth'] and secret != SECRET_KEY:
            await websocket.close(code=1008)
            return
        await evaluation_event_hub.add_client(websocket, initial_payload={"type": "alerts_updated", "alerts": get_alerts()})
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            await evaluation_event_hub.remove_client(websocket)

    # Models
    class PictureData(BaseModel):
        format: str
        timestamp: str
        width: int
        height: int
        length: int
        data: str

    class ConfigRequest(BaseModel):
        name: str
        picture_number: int
        WiFi_RSSI: int
        picture: PictureData

    class SettingsRequest(BaseModel):
        name: str
        threshold_low: int
        threshold_high: int
        threshold_last_low: int
        threshold_last_high: int
        islanding_padding: int
        segments: conint(ge=2)
        rotated_180: bool
        shrink_last_3: bool
        extended_last_digit: bool
        max_flow_rate: float
        conf_threshold: float
        roi_extractor: Optional[str] = None
        template_id: Optional[str] = None
        segment_mode: Optional[str] = None
        digit_models: Optional[List[str]] = None
        decimals: Optional[int] = 3
        use_correctional_alg: Optional[bool] = True
        meter_type: Optional[str] = 'WATER'
        unit: Optional[str] = None
        flip_horizontal: Optional[bool] = False
        brightness_adjust: Optional[int] = 0
        contrast_adjust: Optional[int] = 0
        saturation_adjust: Optional[int] = 0

    class CaptureNowRequest(BaseModel):
        cam_entity_id: Optional[str] = None
        flash_entity_id: Optional[str] = None
        flash_delay_ms: Optional[int] = None
        http_url: Optional[str] = None
        http_headers: Optional[dict] = None
        http_body: Optional[str] = None

    class SettingsUpdateRequest(BaseModel):
        threshold_low: int
        threshold_high: int
        threshold_last_low: int
        threshold_last_high: int
        islanding_padding: int
        segments: conint(ge=2)
        rotated_180: bool
        shrink_last_3: bool
        extended_last_digit: bool
        max_flow_rate: float
        conf_threshold: Optional[float] = None
        roi_extractor: Optional[str] = None
        template_id: Optional[str] = None
        segment_mode: Optional[str] = None
        digit_models: Optional[List[str]] = None
        decimals: Optional[int] = 3
        use_correctional_alg: Optional[bool] = True
        meter_type: Optional[str] = 'WATER'
        unit: Optional[str] = None
        flip_horizontal: Optional[bool] = False
        brightness_adjust: Optional[int] = 0
        contrast_adjust: Optional[int] = 0
        saturation_adjust: Optional[int] = 0

    class TemplateCreateRequest(BaseModel):
        name: str
        extractor: str
        reference_image_base64: str
        image_width: int
        image_height: int
        display_corners: Optional[List[List[float]]] = None
        segment_mode: Optional[str] = None
        digit_corners: Optional[List[List[List[float]]]] = None

    class EvalRequest(BaseModel):
        eval: str

    class SetupData(BaseModel):
        value: int
        timestamp: str

    class SetReadValueRequest(BaseModel):
        value: int

    class DatasetUpload(BaseModel):
        name: str
        labels: List[str]
        colored: List[str]
        thresholded: List[str]

    class ThresholdSearchRequest(BaseModel):
        steps: int = 10

    class GlobalSettingsUpdate(BaseModel):
        mqtt_broker: Optional[str] = None
        mqtt_port: Optional[int] = None
        mqtt_username: Optional[str] = None
        mqtt_password: Optional[str] = None
        mqtt_topic: Optional[str] = None
        max_history: Optional[int] = None
        max_evals: Optional[int] = None

    # --- Camera source models (HA entity polling) ---
    class CameraSourceBase(BaseModel):
        name: str
        camera_entity_id: str
        enabled: bool = True
        poll_interval_s: int = 10 * 60  # default 10 minutes

    class CameraSourceCreate(CameraSourceBase):
        pass

    class CameraSourceUpdate(BaseModel):
        camera_entity_id: Optional[str] = None
        enabled: Optional[bool] = None
        poll_interval_s: Optional[int] = None

    # --- Generic source models (mqtt / ha_camera / http) ---
    class SourceBase(BaseModel):
        name: str
        source_type: str
        enabled: bool = True
        poll_interval_s: Optional[int] = None
        config: Optional[dict] = None

    class SourceCreate(SourceBase):
        pass

    class SourceUpdate(BaseModel):
        enabled: Optional[bool] = None
        poll_interval_s: Optional[int] = None
        config: Optional[dict] = None

    # Helper to sanitize meter name for filenames
    def _sanitize_name(name: str) -> str:
        # allow alnum, dash and underscore; replace others with _
        return re.sub(r"[^A-Za-z0-9_-]", "_", name)

    def _ensure_meter_exists(db, meter_name: str):
        cur = db.cursor()
        cur.execute("SELECT name FROM watermeters WHERE name = ?", (meter_name,))
        if cur.fetchone() is None:
            cur.execute(
                "INSERT INTO watermeters (name, picture_number, wifi_rssi, picture_format, picture_timestamp, picture_width, picture_height, picture_length, picture_data, setup) "
                "VALUES (?, 0, 0, '', '', 0, 0, 0, '', 0)",
                (meter_name,),
            )

            cur.execute('''
                           INSERT OR IGNORE INTO settings
                           (name, threshold_low, threshold_high, threshold_last_low, threshold_last_high,
                            islanding_padding, segments, rotated_180, shrink_last_3, extended_last_digit,
                            max_flow_rate, conf_threshold, roi_extractor, template_id, segment_mode,
                            digit_models, decimals, use_correctional_alg)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                           ''', (
                               meter_name, 0, 125, 0, 125, 20, 7, False, False, False,
                               1.0, None, 'yolo', None, 'display', None, 3, True
                           ))

    def _normalize_source_type(source_type: str) -> str:
        st = (source_type or "").strip().lower()
        # allow some aliases
        if st in {"ha", "homeassistant", "ha_camera", "camera"}:
            return "ha_camera"
        if st in {"http", "http_endpoint", "webhook"}:
            return "http"
        if st in {"mqtt"}:
            return "mqtt"
        return st

    # --- Sources CRUD ---
    @app.get("/api/sources", dependencies=[Depends(authenticate)])
    def list_sources():
        db = db_connection()
        db.row_factory = sqlite3.Row
        cur = db.cursor()
        cur.execute(
            "SELECT id, name, source_type, enabled, poll_interval_s, config_json, last_success_ts, last_error, created_ts, updated_ts "
            "FROM sources ORDER BY id DESC"
        )
        out = []
        for row in cur.fetchall():
            d = dict(row)
            try:
                d["config"] = json.loads(d.pop("config_json")) if d.get("config_json") else None
            except Exception:
                d["config"] = None
                d.pop("config_json", None)
            out.append(d)
        return {"sources": out}

    @app.post("/api/sources", dependencies=[Depends(authenticate)])
    def create_source(payload: SourceCreate):
        st = _normalize_source_type(payload.source_type)
        if st not in {"mqtt", "ha_camera", "http"}:
            raise HTTPException(status_code=400, detail="Invalid source_type. Allowed: mqtt, ha_camera, http")

        if payload.poll_interval_s is not None and payload.poll_interval_s < 1:
            raise HTTPException(status_code=400, detail="poll_interval_s must be >= 1")

        if st in {"ha_camera"}:
            if not payload.config or not payload.config.get("camera_entity_id"):
                raise HTTPException(status_code=400, detail="Missing config.camera_entity_id")
            if payload.poll_interval_s is None:
                raise HTTPException(status_code=400, detail="poll_interval_s is required for ha_camera")
            # flash_entity_id is optional (used for ESPHome LED)
            if payload.config.get("flash_entity_id") is not None and not isinstance(payload.config.get("flash_entity_id"), str):
                raise HTTPException(status_code=400, detail="config.flash_entity_id must be a string")
            if payload.config.get("flash_delay_ms") is not None:
                try:
                    dms = int(payload.config.get("flash_delay_ms"))
                except Exception:
                    raise HTTPException(status_code=400, detail="config.flash_delay_ms must be an integer")
                if dms < 0 or dms > 10000:
                    raise HTTPException(status_code=400, detail="config.flash_delay_ms must be between 0 and 10000")

        if st in {"http"}:
            if not payload.config or not payload.config.get("url"):
                raise HTTPException(status_code=400, detail="Missing config.url")
            url = payload.config.get("url", "")
            if not isinstance(url, str) or not (url.startswith("http://") or url.startswith("https://")):
                raise HTTPException(status_code=400, detail="config.url must start with http:// or https://")
            headers = payload.config.get("headers")
            if headers is not None and not isinstance(headers, dict):
                raise HTTPException(status_code=400, detail="config.headers must be an object")
            body = payload.config.get("body")
            if body is not None and not isinstance(body, str):
                raise HTTPException(status_code=400, detail="config.body must be a string")
            if payload.poll_interval_s is None:
                raise HTTPException(status_code=400, detail="poll_interval_s is required for http")

        if st in {"mqtt"}:
            return HTTPException(status_code=400, detail="Source type mqtt cannot be created via HTTP API")

        db = db_connection()
        db.row_factory = sqlite3.Row
        _ensure_meter_exists(db, payload.name)
        cfg_json = json.dumps(payload.config) if payload.config is not None else None

        cur = db.cursor()
        cur.execute(
            "INSERT INTO sources (name, source_type, enabled, poll_interval_s, config_json, updated_ts) "
            "VALUES (?, ?, ?, ?, ?, datetime('now'))",
            (payload.name, st, 1 if payload.enabled else 0, payload.poll_interval_s, cfg_json),
        )
        db.commit()

        # Trigger initial capture and processing
        try:
            cur.execute("SELECT * FROM sources WHERE name = ? AND source_type = ?", (payload.name, st))
            source_row = cur.fetchone()
            if source_row and st in {'ha_camera', 'http'}:
                capture_and_process_source(config, config['dbfile'], source_row, meter_preditor)
        except Exception as e:
            log(f"[ERROR] Initial capture processing failed for source {payload.name}: {e}")

        return {"message": "Source created"}

    @app.put("/api/sources/{source_id}", dependencies=[Depends(authenticate)])
    def update_source(source_id: int, payload: SourceUpdate):
        if payload.poll_interval_s is not None and payload.poll_interval_s < 1:
            raise HTTPException(status_code=400, detail="poll_interval_s must be >= 1")

        db = db_connection()
        db.row_factory = sqlite3.Row
        cur = db.cursor()
        cur.execute("SELECT * FROM sources WHERE id = ?", (source_id,))
        row = cur.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Source not found")

        enabled = (1 if payload.enabled else 0) if payload.enabled is not None else row["enabled"]
        poll_interval_s = payload.poll_interval_s if payload.poll_interval_s is not None else row["poll_interval_s"]
        if payload.config is None:
            cfg_json = row["config_json"]
        else:
            cfg_json = json.dumps(payload.config)

        cur.execute(
            "UPDATE sources SET enabled = ?, poll_interval_s = ?, config_json = ?, updated_ts = datetime('now') WHERE id = ?",
            (enabled, poll_interval_s, cfg_json, source_id),
        )
        db.commit()
        return {"message": "Source updated"}

    @app.delete("/api/sources/{source_id}", dependencies=[Depends(authenticate)])
    def delete_source(source_id: int):
        db = db_connection()
        cur = db.cursor()
        cur.execute("DELETE FROM sources WHERE id = ?", (source_id,))
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Source not found")
        db.commit()
        return {"message": "Source deleted", "id": source_id}

    @app.post("/api/templates", dependencies=[Depends(authenticate)])
    def create_template(payload: TemplateCreateRequest):
        if not payload.reference_image_base64:
            raise HTTPException(status_code=400, detail="reference_image_base64 is required")

        segment_mode = (payload.segment_mode or "display").strip().lower()
        if segment_mode not in {"display", "each_digit"}:
            raise HTTPException(status_code=400, detail="Invalid segment_mode")

        display_corners = payload.display_corners or []
        digit_corners = payload.digit_corners or []

        if segment_mode == "display":
            if len(display_corners) != 4:
                raise HTTPException(status_code=400, detail="display_corners must contain 4 points")
        else:
            if not digit_corners:
                raise HTTPException(status_code=400, detail="digit_corners is required for each_digit mode")
            for quad in digit_corners:
                if len(quad) != 4:
                    raise HTTPException(status_code=400, detail="digit_corners must contain quads with 4 points")

        try:
            img_bytes = base64.b64decode(payload.reference_image_base64)
            img_np = np.frombuffer(img_bytes, np.uint8)
            reference_image = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid reference image: {e}")

        if reference_image is None:
            raise HTTPException(status_code=400, detail="Failed to decode reference image")

        def order_points(pts):
            pts = np.array(pts, dtype=np.float32)
            s = pts.sum(axis=1)
            diff = np.diff(pts, axis=1).ravel()
            top_left = pts[np.argmin(s)]
            bottom_right = pts[np.argmax(s)]
            top_right = pts[np.argmin(diff)]
            bottom_left = pts[np.argmax(diff)]
            return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

        points = None
        if display_corners and len(display_corners) == 4:
            points = np.array(display_corners, dtype=np.float32)
            if float(np.max(points)) <= 2.0:
                points[:, 0] *= payload.image_width
                points[:, 1] *= payload.image_height
            points = order_points(points)

        digit_points = None
        if digit_corners:
            digit_points = np.array(digit_corners, dtype=np.float32)
            if float(np.max(digit_points)) <= 2.0:
                digit_points[:, :, 0] *= payload.image_width
                digit_points[:, :, 1] *= payload.image_height
            digit_points = np.array([order_points(quad) for quad in digit_points], dtype=np.float32)

        if points is None and digit_points is not None:
            all_points = digit_points.reshape(-1, 2)
            rect = cv2.minAreaRect(all_points)
            box = cv2.boxPoints(rect)
            points = order_points(box)

        width_a = np.linalg.norm(points[0] - points[1])
        width_b = np.linalg.norm(points[2] - points[3])
        height_a = np.linalg.norm(points[0] - points[3])
        height_b = np.linalg.norm(points[1] - points[2])
        target_width = max(int(width_a), int(width_b))
        target_height = max(int(height_a), int(height_b))
        target_width_ext = int(target_width * 1.2)
        target_height_ext = int(target_height * 1.2)

        config_dict = {
            "display_corners": points.tolist(),
            "target_width": target_width,
            "target_height": target_height,
            "target_width_ext": target_width_ext,
            "target_height_ext": target_height_ext,
            "segment_mode": segment_mode
        }
        if digit_points is not None:
            config_dict["digit_corners"] = digit_points.tolist()

        extractor_type = (payload.extractor or "").lower()
        if extractor_type not in {"orb", "static_rect"}:
            raise HTTPException(status_code=400, detail="Invalid extractor type for templates")

        try:
            if extractor_type == "orb":
                extractor = ORBExtractor(reference_image, config_dict)
            elif extractor_type == "static_rect":
                extractor = StaticRectExtractor(reference_image, config_dict)
            else:
                raise ValueError(f"Unsupported extractor type: {extractor_type}")

            ref_b64, config_json, precomputed_b64 = extractor.serialize_template()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to create template: {e}")

        template_id = str(uuid.uuid4())
        created_at = datetime.utcnow().isoformat()

        db = db_connection()
        cur = db.cursor()
        cur.execute(
            """
            INSERT INTO templates
            (id, name, created_at, reference_image_base64, image_width, image_height, config_json, precomputed_data_base64)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                template_id,
                payload.name,
                created_at,
                ref_b64,
                payload.image_width,
                payload.image_height,
                config_json,
                precomputed_b64
            )
        )
        db.commit()

        return {"id": template_id, "name": payload.name}

    @app.get("/api/templates/{template_id}", dependencies=[Depends(authenticate)])
    def get_template(template_id: str):
        db = db_connection()
        cur = db.cursor()
        cur.execute(
            """
            SELECT id, name, created_at, reference_image_base64, image_width, image_height, config_json, precomputed_data_base64
            FROM templates
            WHERE id = ?
            """,
            (template_id,)
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Template not found")

        config = {}
        try:
            config = json.loads(row[6]) if row[6] else {}
        except Exception:
            config = {}

        return {
            "id": row[0],
            "name": row[1],
            "created_at": row[2],
            "reference_image_base64": row[3],
            "image_width": row[4],
            "image_height": row[5],
            "config": config
        }

    @app.post("/api/sources/{source_id}/capture", dependencies=[Depends(authenticate)])
    def trigger_source_capture(source_id: int):
        log(f"[HTTP] Manual capture trigger for source ID {source_id}")
        db = db_connection()
        db.row_factory = sqlite3.Row
        cur = db.cursor()
        cur.execute("SELECT * FROM sources WHERE id = ?", (source_id,))
        row = cur.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Source not found")

        try:
            capture_and_process_source(config, config['dbfile'], row, meter_preditor)
        except Exception as e:
            # print stack trace for debug logging
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Capture processing failed: {e}")

        return {"message": "Capture and processing triggered"}

    # --- Camera sources CRUD (compat wrapper around sources table) ---
    @app.get("/api/camera-sources", dependencies=[Depends(authenticate)])
    def list_camera_sources():
        db = db_connection()
        db.row_factory = sqlite3.Row
        cur = db.cursor()
        cur.execute(
            "SELECT id, name, enabled, poll_interval_s, config_json, last_success_ts, last_error, created_ts, updated_ts "
            "FROM sources WHERE source_type = 'ha_camera' ORDER BY id DESC"
        )
        out = []
        for row in cur.fetchall():
            d = dict(row)
            cfg = None
            try:
                cfg = json.loads(d.get("config_json")) if d.get("config_json") else None
            except Exception:
                cfg = None
            d.pop("config_json", None)
            d["camera_entity_id"] = (cfg or {}).get("camera_entity_id")
            d["flash_entity_id"] = (cfg or {}).get("flash_entity_id")
            out.append(d)
        return {"camera_sources": out}

    @app.post("/api/camera-sources", dependencies=[Depends(authenticate)])
    def create_camera_source(payload: CameraSourceCreate):
        # forward into generic sources
        s = SourceCreate(
            name=payload.name,
            source_type="ha_camera",
            enabled=payload.enabled,
            poll_interval_s=payload.poll_interval_s,
            config={"camera_entity_id": payload.camera_entity_id},
        )
        return create_source(s)

    @app.put("/api/camera-sources/{source_id}", dependencies=[Depends(authenticate)])
    def update_camera_source(source_id: int, payload: CameraSourceUpdate):
        db = db_connection()
        db.row_factory = sqlite3.Row
        cur = db.cursor()
        cur.execute("SELECT * FROM sources WHERE id = ? AND source_type = 'ha_camera'", (source_id,))
        row = cur.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Camera source not found")

        cfg = None
        try:
            cfg = json.loads(row["config_json"]) if row["config_json"] else {}
        except Exception:
            cfg = {}

        if payload.camera_entity_id is not None:
            cfg["camera_entity_id"] = payload.camera_entity_id

        su = SourceUpdate(
            enabled=payload.enabled,
            poll_interval_s=payload.poll_interval_s,
            config=cfg,
        )
        return update_source(source_id, su)

    @app.delete("/api/camera-sources/{source_id}", dependencies=[Depends(authenticate)])
    def delete_camera_source(source_id: int):
        # ensure type matches for nicer errors
        db = db_connection()
        db.row_factory = sqlite3.Row
        cur = db.cursor()
        cur.execute("SELECT 1 FROM sources WHERE id = ? AND source_type = 'ha_camera'", (source_id,))
        if cur.fetchone() is None:
            raise HTTPException(status_code=404, detail="Camera source not found")
        return delete_source(source_id)

    # --- Home Assistant REST helper (Supervisor token preferred, fallback manual token) ---
    def _get_ha_config() -> dict:
        return config.get('homeassistant') or {}

    def _get_ha_base_url() -> Optional[str]:
        ha_cfg = _get_ha_config()
        url = ha_cfg.get('url')
        return url.rstrip('/') if isinstance(url, str) and url.strip() else None

    def _get_ha_token() -> Optional[str]:
        return get_ha_token(config)

    def _ha_request_json_with_method(path: str, method: str = 'GET', body: Optional[dict] = None) -> dict:
        base_url = _get_ha_base_url()
        token = _get_ha_token()
        if not base_url:
            raise HTTPException(status_code=400, detail="Home Assistant URL not configured")
        if not token:
            raise HTTPException(status_code=400, detail="Home Assistant token not configured")

        ha_cfg = _get_ha_config()
        use_supervisor = bool(ha_cfg.get('use_supervisor_token', True))
        timeout_s = int(ha_cfg.get('request_timeout_s', 10) or 10)
        url = f"{base_url}{path}"
        data = json.dumps(body).encode('utf-8') if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header('Authorization', f'Bearer {token}')
        if body is not None:
            req.add_header('Content-Type', 'application/json')

        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                raw = resp.read().decode('utf-8')
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            detail = f"HA API error {e.code} on {path}. "
            if e.code == 401:
                if use_supervisor and 'supervisor' in base_url:
                    detail += "Supervisor endpoint auth failed. Try using http://homeassistant.local:port instead."
                else:
                    detail += "Token authentication failed. Verify your token is valid."
            raise HTTPException(status_code=502, detail=detail)
        except (urllib.error.URLError, socket.timeout) as e:
            raise HTTPException(status_code=502, detail=f"HA API unreachable: {e}")
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"HA API unexpected error: {e}")

    def _ha_request_json(path: str) -> dict:
        return _ha_request_json_with_method(path, method='GET', body=None)

    @app.get('/api/ha/status', dependencies=[Depends(authenticate)])
    def ha_status():
        base_url = _get_ha_base_url()
        ha_cfg = _get_ha_config()
        sup_token_present = bool(os.environ.get('SUPERVISOR_TOKEN'))
        use_sup = bool(ha_cfg.get('use_supervisor_token', True))
        configured_token_present = bool((ha_cfg.get('token') or '').strip())

        status = {
            'configured': bool(base_url) and (configured_token_present or (use_sup and sup_token_present)),
            'base_url': base_url,
            'use_supervisor_token': use_sup,
            'supervisor_token_present': sup_token_present,
            'manual_token_present': configured_token_present,
            'ok': False,
        }

        if not status['configured']:
            return status

        # simple call
        _ = _ha_request_json('/api/config')
        status['ok'] = True
        return status

    @app.get('/api/ha/cameras', dependencies=[Depends(authenticate)])
    def ha_cameras():
        try:
            states = _ha_request_json('/api/states')

            cams = []
            if isinstance(states, list):
                for st in states:
                    ent_id = st.get('entity_id')
                    if not isinstance(ent_id, str) or not ent_id.startswith('camera.'):
                        continue

                    attrs = st.get('attributes') or {}

                    try:
                        suggested = suggest_flash_entity(
                            ha_base_url=_get_ha_base_url(),  # z.B. http://homeassistant.local:8123
                            ha_token=_get_ha_token(),  # Long-lived access token
                            camera_entity_id=ent_id,
                            states=states
                        )
                    except Exception as e:
                        log(f"[FLASH-SUGGEST] error for {ent_id}: {e}")
                        suggested = None

                    cams.append({
                        'entity_id': ent_id,
                        'name': attrs.get('friendly_name') or ent_id,
                        'suggested_flash_entity_id': suggested
                    })

            cams.sort(key=lambda x: x['entity_id'])
            return {'cameras': cams}
        except:
            # print stack trace for debug logging
            import traceback
            traceback.print_exc()

            raise HTTPException(status_code=500, detail="Failed to fetch cameras from Home Assistant")

    class HaServiceCall(BaseModel):
        domain: str
        service: str
        entity_id: str

    @app.post('/api/ha/service', dependencies=[Depends(authenticate)])
    def ha_call_service(payload: HaServiceCall):
        # Minimal helper for flash control (light.turn_on/off)
        if not payload.domain or not payload.service or not payload.entity_id:
            raise HTTPException(status_code=400, detail='domain, service, entity_id are required')
        res = _ha_request_json_with_method(
            f"/api/services/{payload.domain}/{payload.service}",
            method='POST',
            body={'entity_id': payload.entity_id},
        )
        return {'result': res}

    def _ha_get_bytes(path: str) -> bytes:
        """Fetch raw bytes from HA (used for camera proxy images)."""
        base_url = _get_ha_base_url()
        token = _get_ha_token()
        if not base_url:
            raise HTTPException(status_code=400, detail="Home Assistant URL not configured")
        if not token:
            raise HTTPException(status_code=400, detail="Home Assistant token not configured")

        ha_cfg = _get_ha_config()
        use_supervisor = bool(ha_cfg.get('use_supervisor_token', True))
        timeout_s = int(ha_cfg.get('request_timeout_s', 10) or 10)
        url = f"{base_url}{path}"
        req = urllib.request.Request(url)
        req.add_header('Authorization', f'Bearer {token}')

        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            detail = f"HA API error {e.code} on {path}. "
            if e.code == 401:
                if use_supervisor and 'supervisor' in base_url:
                    detail += "Supervisor endpoint auth failed. Try using http://homeassistant.local:port instead."
                else:
                    detail += "Token authentication failed. Verify your token is valid."
            raise HTTPException(status_code=502, detail=detail)
        except (urllib.error.URLError, socket.timeout) as e:
            raise HTTPException(status_code=502, detail=f"HA API unreachable: {e}")
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"HA API unexpected error: {e}")

    def _extract_ha_camera_config(source_row: sqlite3.Row) -> dict:
        try:
            cfg = json.loads(source_row["config_json"]) if source_row["config_json"] else {}
            return cfg if isinstance(cfg, dict) else {}
        except Exception:
            return {}

    @app.post('/api/capture-now', dependencies=[Depends(authenticate)])
    def capture_now(payload: CaptureNowRequest):
        try:
            if payload.http_url:
                raw, itype, _ = capture_from_http_source({
                    'url': payload.http_url,
                    'headers': payload.http_headers,
                    'body': payload.http_body,
                })
                b64 = base64.b64encode(raw).decode('utf-8')
                return {
                    "result": True,
                    "data": b64,
                    "format": itype,
                    "flash_used": False,
                }

            cam_entity_id = payload.cam_entity_id
            if not cam_entity_id:
                raise HTTPException(status_code=400, detail="cam_entity_id or http_url is required")
            flash_entity_id = payload.flash_entity_id
            flash_delay_ms = payload.flash_delay_ms or 10000  # default if not provided

            raw, itype, flash_enabled = capture_from_ha_source(config, {
                'camera_entity_id': cam_entity_id,
                'flash_entity_id': flash_entity_id,
                'flash_delay_ms': flash_delay_ms,
            })
            b64 = base64.b64encode(raw).decode('utf-8')
            return {
                "result": True,
                "data": b64,
                "format": itype,
                "flash_used": bool(flash_enabled),
            }
        except HTTPException as he:
            raise he
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Capture failed: {e}")

    # --- Watermeter settings (CRUD) ---
    @app.get("/api/settings", dependencies=[Depends(authenticate)])
    def list_settings():
        db = db_connection()
        db.row_factory = sqlite3.Row
        cur = db.cursor()
        cur.execute(
            "SELECT name, threshold_low, threshold_high, threshold_last_low, threshold_last_high, islanding_padding, segments, rotated_180, shrink_last_3, extended_last_digit, max_flow_rate, conf_threshold, roi_extractor, template_id, segment_mode, digit_models, decimals, use_correctional_alg "
            "FROM settings ORDER BY name"
        )
        out = [dict(row) for row in cur.fetchall()]
        for row in out:
            if row.get("digit_models"):
                try:
                    row["digit_models"] = json.loads(row["digit_models"])
                except Exception:
                    row["digit_models"] = None
        return {"settings": out}

    @app.post("/api/settings", dependencies=[Depends(authenticate)])
    def create_settings(payload: SettingsRequest):
        db = db_connection()
        cur = db.cursor()
        digit_models = json.dumps(payload.digit_models) if payload.digit_models is not None else None
        decimals = payload.decimals if payload.decimals is not None else 3
        cur.execute(
            "INSERT INTO settings (name, threshold_low, threshold_high, threshold_last_low, threshold_last_high, islanding_padding, segments, rotated_180, shrink_last_3, extended_last_digit, max_flow_rate, conf_threshold, roi_extractor, template_id, segment_mode, digit_models, decimals, use_correctional_alg) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (payload.name, payload.threshold_low, payload.threshold_high, payload.threshold_last_low, payload.threshold_last_high, payload.islanding_padding, payload.segments, 1 if payload.rotated_180 else 0, 1 if payload.shrink_last_3 else 0, 1 if payload.extended_last_digit else 0, payload.max_flow_rate, payload.conf_threshold, payload.roi_extractor or "yolo", payload.template_id, payload.segment_mode or "display", digit_models, decimals, 1 if payload.use_correctional_alg else 0),
        )
        db.commit()
        return {"message": "Settings created"}

    @app.put("/api/settings/{name}", dependencies=[Depends(authenticate)])
    def update_settings(name: str, payload: SettingsUpdateRequest):
        db = db_connection()
        cur = db.cursor()
        cur.execute("SELECT * FROM settings WHERE name = ?", (name,))
        row = cur.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Settings not found")

        roi_extractor = payload.roi_extractor or "yolo"
        template_id = payload.template_id
        if roi_extractor not in {"orb", "static_rect"}:
            template_id = None
        segment_mode = payload.segment_mode or "display"
        digit_models = json.dumps(payload.digit_models) if payload.digit_models is not None else None
        decimals = payload.decimals if payload.decimals is not None else 3

        cur.execute(
            "UPDATE settings SET threshold_low = ?, threshold_high = ?, threshold_last_low = ?, threshold_last_high = ?, islanding_padding = ?, segments = ?, rotated_180 = ?, shrink_last_3 = ?, extended_last_digit = ?, max_flow_rate = ?, conf_threshold = ?, roi_extractor = ?, template_id = ?, segment_mode = ?, digit_models = ?, decimals = ?, use_correctional_alg = ? WHERE name = ?",
            (payload.threshold_low, payload.threshold_high, payload.threshold_last_low, payload.threshold_last_high, payload.islanding_padding, payload.segments, 1 if payload.rotated_180 else 0, 1 if payload.shrink_last_3 else 0, 1 if payload.extended_last_digit else 0, payload.max_flow_rate, payload.conf_threshold, roi_extractor, template_id, segment_mode, digit_models, decimals, 1 if payload.use_correctional_alg else 0, name),
        )
        db.commit()
        return {"message": "Settings updated"}

    @app.delete("/api/settings/{name}", dependencies=[Depends(authenticate)])
    def delete_settings(name: str):
        db = db_connection()
        cur = db.cursor()
        cur.execute("DELETE FROM settings WHERE name = ?", (name,))
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Settings not found")
        db.commit()
        return {"message": "Settings deleted", "name": name}

    # --- Watermeter history (CRUD) ---
    @app.get("/api/history", dependencies=[Depends(authenticate)])
    def list_history():
        db = db_connection()
        db.row_factory = sqlite3.Row
        cur = db.cursor()
        cur.execute(
            "SELECT name, value, timestamp FROM history ORDER BY name, timestamp"
        )
        out = [dict(row) for row in cur.fetchall()]
        return {"history": out}

    @app.post("/api/history", dependencies=[Depends(authenticate)])
    def create_history_entry(payload: SetupData):
        db = db_connection()
        cur = db.cursor()
        cur.execute(
            "INSERT INTO history (name, value, timestamp) VALUES (?, ?, ?)",
            (payload.value, payload.timestamp),
        )
        db.commit()
        return {"message": "History entry created"}

    @app.delete("/api/history/{name}", dependencies=[Depends(authenticate)])
    def delete_history(name: str):
        db = db_connection()
        cur = db.cursor()
        cur.execute("DELETE FROM history WHERE name = ?", (name,))
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="History not found")
        db.commit()
        return {"message": "History deleted", "name": name}

    @app.get("/api/alerts", dependencies=[Depends(authenticate)])
    def get_current_alerts():
        return get_alerts()

    @app.get("/api/discovery", dependencies=[Depends(authenticate)])
    def get_discovery():
        cursor = db_connection().cursor()
        cursor.execute(
            "SELECT name, picture_timestamp, wifi_rssi, "
            "(SELECT source_type FROM sources WHERE name = watermeters.name LIMIT 1), "
            "picture_thumbnail "
            "FROM watermeters WHERE setup = 0"
        )
        return {"watermeters": [tuple(row) for row in cursor.fetchall()]}

    @app.post("/api/dataset/upload", dependencies=[Depends(authenticate)])
    def upload_dataset(payload: DatasetUpload):
        # Validate equal lengths
        n = len(payload.labels)
        if not (len(payload.colored) == n == len(payload.thresholded)):
            raise HTTPException(status_code=400,
                                detail="'labels', 'colored' and 'thresholded' arrays must have equal length")

        # allowed labels are 0-9 and 'r'
        allowed = set([str(i) for i in range(10)] + ["r"])

        out_root = config.get('output_dataset', '/data/output_dataset')
        # Ensure root exists and create a per-meter root folder
        os.makedirs(out_root, exist_ok=True)

        saved = 0
        meter_name = _sanitize_name(payload.name)
        meter_root = os.path.join(out_root, meter_name)
        os.makedirs(meter_root, exist_ok=True)

        for idx in range(n):
            raw_label = payload.labels[idx]
            label = str(raw_label)
            if label not in allowed:
                raise HTTPException(status_code=400, detail=f"Invalid label at index {idx}: {label}")

            # ensure per-meter color and th label folders exist
            color_label_dir = os.path.join(meter_root, 'color', label)
            th_label_dir = os.path.join(meter_root, 'th', label)
            os.makedirs(color_label_dir, exist_ok=True)
            os.makedirs(th_label_dir, exist_ok=True)

            # decode images
            try:
                col_bytes = base64.b64decode(payload.colored[idx])
            except Exception:
                raise HTTPException(status_code=400, detail=f"Invalid base64 in 'colored' at index {idx}")
            try:
                th_bytes = base64.b64decode(payload.thresholded[idx])
            except Exception:
                raise HTTPException(status_code=400, detail=f"Invalid base64 in 'thresholded' at index {idx}")

            # compute crc32 of combined bytes for uniqueness
            crc_input = bytes(col_bytes) + bytes(th_bytes)
            crc = zlib.crc32(crc_input) & 0xFFFFFFFF
            crc_hex = f"{crc:08x}"

            filename = f"{label}_{meter_name}_{crc_hex}.png"
            filepath_col = os.path.join(color_label_dir, filename)
            filepath_th = os.path.join(th_label_dir, filename)

            # write files
            try:
                with open(filepath_col, 'wb') as f:
                    f.write(col_bytes)
                with open(filepath_th, 'wb') as f:
                    f.write(th_bytes)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to write files for index {idx}: {e}")

            saved += 1

        return {"saved": saved, "output_root": out_root}

    @app.get("/api/dataset/{name}/download", dependencies=[Depends(authenticate)])
    def download_dataset(name: str):
        out_root = config.get('output_dataset', '/data/output_dataset')
        meter_name = _sanitize_name(name)
        meter_root = os.path.join(out_root, meter_name)

        if not os.path.isdir(meter_root):
            raise HTTPException(status_code=404, detail="Dataset not found")

        # Create a temporary zip file
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, f"{meter_name}_dataset.zip")

        try:
            # Create zip archive
            shutil.make_archive(zip_path.replace('.zip', ''), 'zip', meter_root)

            # Read the zip file
            with open(zip_path, 'rb') as f:
                zip_data = f.read()

            # Clean up temp directory
            shutil.rmtree(temp_dir)

            # Return as streaming response
            return StreamingResponse(
                BytesIO(zip_data),
                media_type="application/zip",
                headers={
                    "Content-Disposition": f"attachment; filename={meter_name}_dataset.zip"
                }
            )
        except Exception as e:
            # Clean up on error
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            raise HTTPException(status_code=500, detail=f"Failed to create zip: {str(e)}")

    @app.delete("/api/dataset/{name}", dependencies=[Depends(authenticate)])
    def delete_dataset(name: str):
        out_root = config.get('output_dataset', '/data/output_dataset')
        meter_name = _sanitize_name(name)
        meter_root = os.path.join(out_root, meter_name)

        if not os.path.isdir(meter_root):
            raise HTTPException(status_code=404, detail="Dataset not found")

        try:
            shutil.rmtree(meter_root)
            return {"message": "Dataset deleted", "name": name}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to delete dataset: {str(e)}")

    @app.get("/api/config", dependencies=[Depends(authenticate)])
    def get_config():
        tconfig = config.copy()
        del tconfig['secret_key']
        return tconfig

    @app.get("/api/watermeters", dependencies=[Depends(authenticate)])
    def get_watermeters():
        cursor = db_connection().cursor()
        cursor.execute("""
                       SELECT w.name,
                              w.picture_timestamp,
                              w.wifi_rssi,
                              (SELECT value FROM history h WHERE h.name = w.name ORDER BY timestamp DESC LIMIT 1),
                              w.picture_thumbnail,
                              w.picture_data_bbox IS NOT NULL,
                              (SELECT COALESCE(s.decimals, 3) FROM settings s WHERE s.name = w.name LIMIT 1),
                              COALESCE(w.meter_type, 'WATER'),
                              w.unit
                       FROM watermeters w
                       WHERE w.setup = 1
                       """)

        result = []
        for row in cursor.fetchall():
            has_bbox = bool(row[5])
            decimals = row[6] if row[6] is not None else 3
            meter_type = row[7] if row[7] is not None else 'WATER'
            unit = row[8]
            result.append((row[0], row[1], row[2], row[3], row[4], has_bbox, decimals, meter_type, unit))

        return {"watermeters": result}

    @app.post("/api/setup/{name}/finish", dependencies=[Depends(authenticate)])
    def post_setup_finished(name: str, data: SetupData):
        db = db_connection()
        cursor = db.cursor()
        cursor.execute("UPDATE watermeters SET setup = 1 WHERE name = ?", (name,))
        db.commit()
        target_brightness, confidence, _ = reevaluate_latest_picture(config['dbfile'], name, meter_preditor, config,
                                                                     skip_setup_overwriting=False)
        add_history_entry(config['dbfile'], name, data.value, 1, target_brightness, data.timestamp, config, manual=True)

        # clear evaluations
        cursor = db.cursor()
        cursor.execute("UPDATE evaluations SET outdated = true WHERE name = ?", (name,))
        db.commit()

        return {"message": "Setup completed"}

    @app.post("/api/setup/{name}/enable", dependencies=[Depends(authenticate)])
    def post_setup_enable(name: str):
        db = db_connection()
        cursor = db.cursor()
        cursor.execute("UPDATE watermeters SET setup = 0 WHERE name = ?", (name,))
        db.commit()
        return {"message": "Setup completed"}

    @app.get("/api/watermeters/{name}/history", dependencies=[Depends(authenticate)])
    def get_watermeter_history(name: str):
        cursor = db_connection().cursor()
        cursor.execute("SELECT value, timestamp, confidence, manual FROM history WHERE name = ?", (name,))
        return {"history": [row for row in cursor.fetchall()]}

    @app.get("/api/watermeters/{name}/daily-history", dependencies=[Depends(authenticate)])
    def get_watermeter_daily_history(name: str):
        cursor = db_connection().cursor()
        cursor.execute("SELECT value, date FROM daily_history WHERE name = ? ORDER BY date ASC", (name,))
        return {"history": [row for row in cursor.fetchall()]}

    @app.get("/api/watermeters/{name}/stats", dependencies=[Depends(authenticate)])
    def get_watermeter_stats(name: str):
        """Return today/daily-avg/yearly-avg consumption stats for the card view."""
        cursor = db_connection().cursor()

        # Latest meter reading from history
        cursor.execute(
            "SELECT value FROM history WHERE name = ? ORDER BY ROWID DESC LIMIT 1", (name,)
        )
        row = cursor.fetchone()
        current_value = row[0] if row else None

        # Last daily_history entry (most recent completed day)
        cursor.execute(
            "SELECT value, date FROM daily_history WHERE name = ? ORDER BY date DESC LIMIT 1", (name,)
        )
        row = cursor.fetchone()
        last_daily_value, last_daily_date = (row[0], row[1]) if row else (None, None)

        # Today's consumption = current - last daily snapshot
        today_consumption = None
        if current_value is not None and last_daily_value is not None:
            today_consumption = max(0, current_value - last_daily_value)

        # Daily average and yearly average from daily_history deltas
        cursor.execute(
            "SELECT value, date FROM daily_history WHERE name = ? ORDER BY date ASC", (name,)
        )
        rows = cursor.fetchall()

        daily_avg, yearly_avg, days_of_data = compute_daily_avg(rows)

        # Median interval between consecutive history entries (seconds)
        cursor.execute("""
            SELECT AVG(diff) FROM (
                SELECT diff FROM (
                    SELECT (julianday(timestamp) - julianday(LAG(timestamp) OVER (ORDER BY timestamp))) * 86400.0 AS diff
                    FROM history WHERE name = ?
                ) WHERE diff IS NOT NULL
                ORDER BY diff
                LIMIT  2 - ((SELECT COUNT(*) FROM history WHERE name = ?) - 1) % 2
                OFFSET ((SELECT COUNT(*) FROM history WHERE name = ?) - 2) / 2
            )
        """, (name, name, name))
        row = cursor.fetchone()
        median_interval = row[0] if row else None

        return {
            "today_consumption": today_consumption,
            "last_daily_date": last_daily_date,
            "daily_avg": daily_avg,
            "yearly_avg": yearly_avg,
            "extrapolated": days_of_data is not None and days_of_data < 365,
            "days_of_data": days_of_data,
            "median_interval": median_interval,
        }

    @app.get("/api/watermeters/{name}", dependencies=[Depends(authenticate)])
    def get_watermeter(name: str):
        cursor = db_connection().cursor()
        cursor.execute("SELECT * FROM watermeters WHERE name = ?", (name,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Watermeter not found")
        # check for dataset presence under configured output root for this meter
        out_root = config.get('output_dataset', '/data/output_dataset')
        meter_root = os.path.join(out_root, _sanitize_name(name))
        dataset_present = False
        if os.path.isdir(meter_root):
            for _dirpath, _dirnames, files in os.walk(meter_root):
                if any(f.lower().endswith('.png') for f in files):
                    dataset_present = True
                    break

        picture_data = row[8]
        picture_bbox = row[10]
        if isinstance(picture_data, (bytes, bytearray)):
            picture_data = base64.b64encode(picture_data).decode('utf-8')
        if isinstance(picture_bbox, (bytes, bytearray)):
            picture_bbox = base64.b64encode(picture_bbox).decode('utf-8')

        return {
            "name": row[0],
            "picture_number": row[1],
            "WiFi-RSSI": row[2],
            "picture": {
                "format": row[3],
                "timestamp": row[4],
                "width": row[5],
                "height": row[6],
                "length": row[7],
                "data": picture_data,
                "data_bbox": picture_bbox
            },
            "dataset_present": dataset_present
        }

    @app.delete("/api/watermeters/{name}", dependencies=[Depends(authenticate)])
    def delete_watermeter(name: str):
        db = db_connection()
        cursor = db.cursor()
        cursor.execute("DELETE FROM watermeters WHERE name = ?", (name,))
        cursor.execute("DELETE FROM evaluations WHERE name = ?", (name,))
        cursor.execute("DELETE FROM history WHERE name = ?", (name,))
        cursor.execute("DELETE FROM settings WHERE name = ?", (name,))
        cursor.execute("DELETE FROM sources WHERE name = ?", (name,))
        db.commit()
        # Remove any alerts associated with this meter
        for key in list(get_alerts().keys()):
            if key.startswith(f'polling_{name}'):
                remove_alert(key)
        return {"message": "Watermeter deleted", "name": name}

    @app.post("/api/setup", dependencies=[Depends(authenticate)])
    def setup_watermeter(config: ConfigRequest):
        from lib.capture_utils import make_thumbnail
        raw_bytes = base64.b64decode(config.picture.data)
        thumbnail = make_thumbnail(raw_bytes, config.picture.format)
        db = db_connection()
        cursor = db.cursor()
        cursor.execute(
            """
            INSERT INTO watermeters (name, picture_number, wifi_rssi, picture_format,
                                     picture_timestamp, picture_width, picture_height, picture_length, picture_data,
                                     picture_thumbnail, setup)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (
                config.name,
                config.picture_number,
                config.WiFi_RSSI,
                config.picture.format,
                config.picture.timestamp,
                config.picture.width,
                config.picture.height,
                config.picture.length,
                config.picture.data,
                thumbnail
            )
        )
        db.commit()
        return {"message": "Watermeter configured", "name": config.name}

    @app.get("/api/settings/{name}", dependencies=[Depends(authenticate)])
    @app.get("/api/watermeters/{name}/settings", dependencies=[Depends(authenticate)])
    def get_settings(name: str):
        db = db_connection()
        db.row_factory = sqlite3.Row
        cursor = db.cursor()
        cursor.execute(
            "SELECT threshold_low, threshold_high, threshold_last_low, threshold_last_high, islanding_padding, segments, shrink_last_3, extended_last_digit, max_flow_rate, rotated_180, conf_threshold, roi_extractor, template_id, segment_mode, digit_models, decimals, use_correctional_alg, flip_horizontal, brightness_adjust, contrast_adjust, saturation_adjust FROM settings WHERE name = ?",
            (name,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Thresholds not found")
        digit_models = row[14]
        if digit_models:
            try:
                digit_models = json.loads(digit_models)
            except Exception:
                digit_models = None
        # Fetch meter_type and unit from watermeters table
        cursor.execute("SELECT COALESCE(meter_type, 'WATER'), unit FROM watermeters WHERE name = ?", (name,))
        wm_row = cursor.fetchone()
        meter_type = wm_row[0] if wm_row else 'WATER'
        unit = wm_row[1] if wm_row else None
        return {
            "threshold_low": row[0],
            "threshold_high": row[1],
            "threshold_last_low": row[2],
            "threshold_last_high": row[3],
            "islanding_padding": row[4],
            "segments": row[5],
            "shrink_last_3": row[6],
            "extended_last_digit": row[7],
            "max_flow_rate": row[8],
            "rotated_180": row[9],
            "conf_threshold": row[10],
            "roi_extractor": row[11],
            "template_id": row[12],
            "segment_mode": row[13],
            "digit_models": digit_models,
            "decimals": row[15],
            "use_correctional_alg": row[16],
            "meter_type": meter_type,
            "unit": unit,
            "flip_horizontal": bool(row[17]) if row[17] is not None else False,
            "brightness_adjust": int(row[18]) if row[18] is not None else 0,
            "contrast_adjust": int(row[19]) if row[19] is not None else 0,
            "saturation_adjust": int(row[20]) if row[20] is not None else 0,
        }

    @app.post("/api/settings", dependencies=[Depends(authenticate)])
    def set_settings(settings: SettingsRequest):
        db = db_connection()
        cursor = db.cursor()
        digit_models = json.dumps(settings.digit_models) if settings.digit_models is not None else None
        decimals = settings.decimals if settings.decimals is not None else 3
        cursor.execute(
            """
            INSERT INTO settings (name, threshold_low, threshold_high, threshold_last_low, threshold_last_high,
                                  islanding_padding, segments, shrink_last_3, extended_last_digit, max_flow_rate,
                                  rotated_180, conf_threshold, roi_extractor, template_id, segment_mode, digit_models, decimals, use_correctional_alg,
                                  flip_horizontal, brightness_adjust, contrast_adjust, saturation_adjust)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET threshold_low=excluded.threshold_low,
                                            threshold_high=excluded.threshold_high,
                                            threshold_last_low=excluded.threshold_last_low,
                                            threshold_last_high=excluded.threshold_last_high,
                                            islanding_padding=excluded.islanding_padding,
                                            segments=excluded.segments,
                                            shrink_last_3=excluded.shrink_last_3,
                                            extended_last_digit=excluded.extended_last_digit,
                                            max_flow_rate=excluded.max_flow_rate,
                                            rotated_180=excluded.rotated_180,
                                            conf_threshold=excluded.conf_threshold,
                                            roi_extractor=excluded.roi_extractor,
                                            template_id=excluded.template_id,
                                            segment_mode=excluded.segment_mode,
                                            digit_models=excluded.digit_models,
                                            decimals=excluded.decimals,
                                            use_correctional_alg=excluded.use_correctional_alg,
                                            flip_horizontal=excluded.flip_horizontal,
                                            brightness_adjust=excluded.brightness_adjust,
                                            contrast_adjust=excluded.contrast_adjust,
                                            saturation_adjust=excluded.saturation_adjust
            """,
            (settings.name, settings.threshold_low, settings.threshold_high, settings.threshold_last_low,
             settings.threshold_last_high, settings.islanding_padding,
             settings.segments, settings.shrink_last_3, settings.extended_last_digit, settings.max_flow_rate,
             settings.rotated_180, settings.conf_threshold, settings.roi_extractor or "yolo", settings.template_id, settings.segment_mode or "display", digit_models, decimals, settings.use_correctional_alg,
             1 if settings.flip_horizontal else 0, settings.brightness_adjust or 0, settings.contrast_adjust or 0, settings.saturation_adjust or 0)
        )
        cursor.execute(
            "UPDATE watermeters SET meter_type = ?, unit = ? WHERE name = ?",
            (settings.meter_type or 'WATER', settings.unit, settings.name)
        )
        db.commit()
        return {"message": "Thresholds set", "name": settings.name}

    @app.put("/api/watermeters/{name}/settings", dependencies=[Depends(authenticate)])
    def update_settings(name: str, settings: SettingsUpdateRequest):
        db = db_connection()
        cursor = db.cursor()
        digit_models = json.dumps(settings.digit_models) if settings.digit_models is not None else None
        decimals = settings.decimals if settings.decimals is not None else 3
        cursor.execute(
            """
            INSERT INTO settings (name, threshold_low, threshold_high, threshold_last_low, threshold_last_high,
                                  islanding_padding, segments, shrink_last_3, extended_last_digit, max_flow_rate,
                                  rotated_180, conf_threshold, roi_extractor, template_id, segment_mode, digit_models, decimals, use_correctional_alg,
                                  flip_horizontal, brightness_adjust, contrast_adjust, saturation_adjust)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET threshold_low=excluded.threshold_low,
                                            threshold_high=excluded.threshold_high,
                                            threshold_last_low=excluded.threshold_last_low,
                                            threshold_last_high=excluded.threshold_last_high,
                                            islanding_padding=excluded.islanding_padding,
                                            segments=excluded.segments,
                                            shrink_last_3=excluded.shrink_last_3,
                                            extended_last_digit=excluded.extended_last_digit,
                                            max_flow_rate=excluded.max_flow_rate,
                                            rotated_180=excluded.rotated_180,
                                            conf_threshold=excluded.conf_threshold,
                                            roi_extractor=excluded.roi_extractor,
                                            template_id=excluded.template_id,
                                            segment_mode=excluded.segment_mode,
                                            digit_models=excluded.digit_models,
                                            decimals=excluded.decimals,
                                            use_correctional_alg=excluded.use_correctional_alg,
                                            flip_horizontal=excluded.flip_horizontal,
                                            brightness_adjust=excluded.brightness_adjust,
                                            contrast_adjust=excluded.contrast_adjust,
                                            saturation_adjust=excluded.saturation_adjust
            """,
            (name, settings.threshold_low, settings.threshold_high, settings.threshold_last_low,
             settings.threshold_last_high, settings.islanding_padding,
             settings.segments, settings.shrink_last_3, settings.extended_last_digit, settings.max_flow_rate,
             settings.rotated_180, settings.conf_threshold, settings.roi_extractor or "yolo", settings.template_id, settings.segment_mode or "display", digit_models, decimals, settings.use_correctional_alg,
             1 if settings.flip_horizontal else 0, settings.brightness_adjust or 0, settings.contrast_adjust or 0, settings.saturation_adjust or 0)
        )
        cursor.execute(
            "UPDATE watermeters SET meter_type = ?, unit = ? WHERE name = ?",
            (settings.meter_type or 'WATER', settings.unit, name)
        )
        db.commit()
        if get_mqtt_client_fn:
            mqtt_client = get_mqtt_client_fn()
            if mqtt_client:
                publish_registration(mqtt_client, config, name, 'value',
                                     meter_type=settings.meter_type or 'WATER',
                                     unit=settings.unit)
        return {"message": "Settings updated", "name": name}

    @app.post("/api/watermeters/{name}/set-read-value", dependencies=[Depends(authenticate)])
    def post_set_read_value(name: str, data: SetReadValueRequest):
        """Manually set the current meter reading to correct the correction algorithm baseline."""
        db = db_connection()
        cursor = db.cursor()
        # Preserve target_brightness from the last history entry so the next
        # reevaluation uses the same thresholding as before (0.0 would make digits black).
        cursor.execute(
            "SELECT target_brightness FROM history WHERE name = ? ORDER BY ROWID DESC LIMIT 1",
            (name,)
        )
        row = cursor.fetchone()
        target_brightness = row[0] if row and row[0] is not None else None
        timestamp = datetime.now().isoformat()
        add_history_entry(config['dbfile'], name, data.value, 1, target_brightness, timestamp, config, manual=True)
        cursor.execute("UPDATE evaluations SET outdated = true WHERE name = ?", (name,))
        db.commit()
        return {"message": "Read value set"}

    @app.post("/api/watermeters/{name}/search_thresholds", dependencies=[Depends(authenticate)])
    def search_thresholds(name: str, request: ThresholdSearchRequest):
        """
        Search for optimal threshold values by maximizing digit recognition confidence.

        This endpoint performs a grid search over threshold values and returns the
        combination that yields the highest confidence from the digit recognition model.

        Args:
            name: Water meter name
            request: ThresholdSearchRequest with steps parameter (3-25)

        Returns:
            Optimal threshold values and confidence metrics
        """
        # Validate meter exists
        cursor = db_connection().cursor()
        cursor.execute("SELECT name FROM watermeters WHERE name = ?", (name,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Watermeter not found")

        # Clamp steps to valid range
        steps = max(3, min(request.steps, 25))

        log(f"[HTTP] Starting threshold search for {name} with steps={steps}")

        try:
            result = search_thresholds_for_meter(
                config['dbfile'],
                name,
                meter_preditor,
                steps=steps
            )

            if "error" in result:
                log(f"[HTTP] Threshold search error: {result['error']}")
            else:
                log(f"[HTTP] Threshold search completed: threshold={result['threshold']}, "
                      f"threshold_last={result['threshold_last']}, avg_conf={result.get('avg_confidence', 0):.3f}")

            return result

        except Exception as e:
            log(f"[HTTP] Threshold search failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Threshold search failed: {str(e)}")

    @app.post("/api/watermeters/{name}/evaluations/reevaluate", dependencies=[Depends(authenticate)])
    def reevaluate_latest(name: str, skip_setup_overwriting: bool = False):
        try:
            r = reevaluate_latest_picture(config['dbfile'], name, meter_preditor, config, skip_setup_overwriting=skip_setup_overwriting)
            if r is None:
                return {"result": False, "error": meter_preditor.last_error or "No result found"}
            _, _, bbox_base64 = r

            # update in watermeters table
            db = db_connection()
            cursor = db.cursor()
            cursor.execute("UPDATE watermeters SET picture_data_bbox = ? WHERE name = ?", (bbox_base64, name))
            db.commit()
            log(f"[HTTP] Re-evaluated latest picture for watermeter {name}")
            return {"result": True}

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Re-evaluation failed: {str(e)}")

    @app.post("/api/watermeters/{name}/evaluations/mark-outdated", dependencies=[Depends(authenticate)])
    def mark_evaluations_outdated(name: str):
        """Mark all evaluations as outdated for a watermeter to trigger re-evaluation."""
        db = db_connection()
        cursor = db.cursor()

        # Check if watermeter exists
        cursor.execute("SELECT name FROM watermeters WHERE name = ?", (name,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Watermeter not found")

        # Mark all evaluations as outdated
        cursor.execute("UPDATE evaluations SET outdated = 1 WHERE name = ?", (name,))
        affected = cursor.rowcount
        db.commit()

        log(f"[HTTP] Marked {affected} evaluations as outdated for watermeter {name}")
        return {"result": True, "count": affected}

    @app.post("/api/watermeters/{name}/evaluations/sample", dependencies=[Depends(authenticate)])
    @app.post("/api/watermeters/{name}/evaluations/sample/{offset}", dependencies=[Depends(authenticate)])
    def get_reevaluated_digits(name: str, offset: int = None):
        # returns a set of random digits from historic evaluations for the given watermeter, evaluated with the current settings
        # if offset is provided, returns the evaluation at that offset from the latest (0 = latest, 1 = second latest, etc.)
        # if offset is -1, returns a random evaluation
        return reevaluate_digits(config['dbfile'], name, meter_preditor, config, offset)

    # GET endpoint for retrieving evaluations
    @app.get("/api/watermeters/{name}/evals", dependencies=[Depends(authenticate)])
    def get_evals(name: str, amount: int = None, from_id: int = None):
        cursor = db_connection().cursor()
        # Check if watermeter exists
        cursor.execute("SELECT name FROM watermeters WHERE name = ?", (name,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Watermeter not found")
        # Retrieve all evaluations for the watermeter
        cursor = db_connection().cursor()

        # Build query with optional pagination
        query = """
                SELECT colored_digits, \
                       th_digits, \
                       predictions, \
                       timestamp, \
                       result, \
                       total_confidence, \
                       used_confidence, \
                       outdated, \
                       id, \
                       denied_digits, \
                       th_digits_inverted, \
                       flow_rate_m3h, \
                       delta_m3, \
                       delta_raw, \
                       time_diff_min, \
                       rejection_reason, \
                       negative_correction_applied, \
                       fallback_digit_count, \
                       digits_changed_vs_last, \
                       digits_changed_vs_top_pred, \
                       prediction_rank_used_counts, \
                       denied_digits_count, \
                       timestamp_adjusted
                FROM evaluations
                WHERE name = ? \
                """
        params = [name]

        if from_id is not None:
            query += " AND id < ?"
            params.append(from_id)

        query += " ORDER BY id DESC"

        if amount is not None:
            query += " LIMIT ?"
            params.append(amount)

        cursor.execute(query, params)
        return {"evals": [{
            "id": row[8],
            "colored_digits": json.loads(row[0]) if row[0] else None,
            "th_digits": json.loads(row[1]) if row[1] else None,
            "predictions": json.loads(row[2]) if row[2] else None,
            "timestamp": row[3],
            "result": row[4],
            "total_confidence": row[5],
            "used_confidence": row[6],
            "outdated": row[7],
            "denied_digits": json.loads(row[9]) if row[9] else None,
            "th_digits_inverted": json.loads(row[10]) if row[10] else None,
            "flow_rate_m3h": row[11],
            "delta_m3": row[12],
            "delta_raw": row[13],
            "time_diff_min": row[14],
            "rejection_reason": row[15],
            "negative_correction_applied": row[16],
            "fallback_digit_count": row[17],
            "digits_changed_vs_last": row[18],
            "digits_changed_vs_top_pred": row[19],
            "prediction_rank_used_counts": json.loads(row[20]) if row[20] else None,
            "denied_digits_count": row[21],
            "timestamp_adjusted": row[22]
        } for row in cursor.fetchall()]}

    @app.get("/api/watermeters/{name}/evals/count", dependencies=[Depends(authenticate)])
    def get_evals_count(name: str):
        """Get the number of evaluations for a watermeter."""
        cursor = db_connection().cursor()

        # Check if watermeter exists
        cursor.execute("SELECT name FROM watermeters WHERE name = ?", (name,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Watermeter not found")

        cursor.execute("SELECT COUNT(*) FROM evaluations WHERE name = ?", (name,))
        count = cursor.fetchone()[0]

        return {"count": count}

    @app.get("/api/watermeters/{name}/evals/{eval_id}", dependencies=[Depends(authenticate)])
    def get_eval_by_id(name: str, eval_id: int):
        """Get a single evaluation by ID."""
        cursor = db_connection().cursor()

        # Check if watermeter exists
        cursor.execute("SELECT name FROM watermeters WHERE name = ?", (name,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Watermeter not found")

        # Get the evaluation
        cursor.execute("""
            SELECT colored_digits,
                   th_digits,
                   predictions,
                   timestamp,
                   result,
                   total_confidence,
                   used_confidence,
                   outdated,
                   id,
                   denied_digits,
                   th_digits_inverted,
                   flow_rate_m3h,
                   delta_m3,
                   delta_raw,
                   time_diff_min,
                   rejection_reason,
                   negative_correction_applied,
                   fallback_digit_count,
                   digits_changed_vs_last,
                   digits_changed_vs_top_pred,
                   prediction_rank_used_counts,
                   denied_digits_count,
                   timestamp_adjusted
            FROM evaluations
            WHERE name = ? AND id = ?
        """, (name, eval_id))

        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Evaluation not found")

        return {
            "id": row[8],
            "colored_digits": json.loads(row[0]) if row[0] else None,
            "th_digits": json.loads(row[1]) if row[1] else None,
            "predictions": json.loads(row[2]) if row[2] else None,
            "timestamp": row[3],
            "result": row[4],
            "total_confidence": row[5],
            "used_confidence": row[6],
            "outdated": row[7],
            "denied_digits": json.loads(row[9]) if row[9] else None,
            "th_digits_inverted": json.loads(row[10]) if row[10] else None,
            "flow_rate_m3h": row[11],
            "delta_m3": row[12],
            "delta_raw": row[13],
            "time_diff_min": row[14],
            "rejection_reason": row[15],
            "negative_correction_applied": row[16],
            "fallback_digit_count": row[17],
            "digits_changed_vs_last": row[18],
            "digits_changed_vs_top_pred": row[19],
            "prediction_rank_used_counts": json.loads(row[20]) if row[20] else None,
            "denied_digits_count": row[21],
            "timestamp_adjusted": row[22]
        }

    @app.delete("/api/watermeters/{name}/evals", dependencies=[Depends(authenticate)])
    def delete_evals(name: str):
        """Delete all evaluations for a watermeter."""
        db = db_connection()
        cursor = db.cursor()

        # Check if watermeter exists
        cursor.execute("SELECT name FROM watermeters WHERE name = ?", (name,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Watermeter not found")

        # Count evaluations before deleting
        cursor.execute("SELECT COUNT(*) FROM evaluations WHERE name = ?", (name,))
        count = cursor.fetchone()[0]

        # Delete all evaluations
        cursor.execute("DELETE FROM evaluations WHERE name = ?", (name,))
        db.commit()

        log(f"[HTTP] Deleted {count} evaluations for watermeter {name}")
        return {"message": f"Deleted {count} evaluations", "count": count}

    # POST endpoint for adding an evaluation
    @app.post("/api/watermeters/{name}/evals", dependencies=[Depends(authenticate)])
    def add_eval(name: str, eval_req: EvalRequest):
        db = db_connection()
        cursor = db.cursor()
        # Check if watermeter exists
        cursor.execute("SELECT name FROM watermeters WHERE name = ?", (name,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Watermeter not found")
        # Insert the new evaluation
        cursor.execute(
            "INSERT INTO evaluations (name, eval) VALUES (?, ?)",
            (name, eval_req.eval)
        )
        db.commit()
        return {"message": "Eval added", "name": name}

    @app.post("/api/evaluate/single", dependencies=[Depends(authenticate)])
    def evaluate(
            base64str: str = Body(...),  # Changed from Query to Body for POST requests
            threshold_low: float = Body(0, ge=0, le=255),
            threshold_high: float = Body(155, ge=0, le=255),
            islanding_padding: int = Body(20, ge=0),
            invert: bool = Body(False)
    ):
            # Decode the base64 image
            image_data = base64.b64decode(base64str)

            # Get PIL image from base64
            image = Image.open(BytesIO(image_data))
            # to numpy array
            image = np.array(image)

            # Convert RGB (from PIL) to BGR for consistent OpenCV processing
            if image.ndim == 3 and image.shape[2] >= 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Apply threshold with the passed values
            base64r, digits = meter_preditor.apply_threshold(image, threshold_low, threshold_high, islanding_padding, invert=invert)

            # Return the result
            return {"base64": base64r}

    # --- Global settings endpoints ---

    @app.get("/api/global-settings", dependencies=[Depends(authenticate)])
    def get_global_settings():
        cursor = db_connection().cursor()
        cursor.execute("""
            SELECT mqtt_broker, mqtt_port, mqtt_username, mqtt_password, mqtt_topic, max_history, max_evals
            FROM global_settings WHERE id = 1
        """)
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=500, detail="Global settings not found")
        return {
            "mqtt_broker": row[0],
            "mqtt_port": row[1],
            "mqtt_username": row[2],
            "mqtt_password": row[3],
            "mqtt_topic": row[4],
            "max_history": row[5],
            "max_evals": row[6],
        }

    @app.put("/api/global-settings", dependencies=[Depends(authenticate)])
    def update_global_settings(payload: GlobalSettingsUpdate):
        db = db_connection()
        cursor = db.cursor()
        cursor.execute("""
            UPDATE global_settings SET
                mqtt_broker = COALESCE(?, mqtt_broker),
                mqtt_port = COALESCE(?, mqtt_port),
                mqtt_username = ?,
                mqtt_password = ?,
                mqtt_topic = COALESCE(?, mqtt_topic),
                max_history = COALESCE(?, max_history),
                max_evals = COALESCE(?, max_evals)
            WHERE id = 1
        """, (
            payload.mqtt_broker,
            payload.mqtt_port,
            payload.mqtt_username,
            payload.mqtt_password,
            payload.mqtt_topic,
            payload.max_history,
            payload.max_evals,
        ))
        db.commit()
        # Update in-memory config so active requests reflect changes immediately
        if payload.max_history is not None:
            config['max_history'] = payload.max_history
        if payload.max_evals is not None:
            config['max_evals'] = payload.max_evals
        log("[HTTP] Global settings updated")
        return {"result": True}

    @app.post("/api/global-settings/mqtt/restart", dependencies=[Depends(authenticate)])
    def restart_mqtt():
        if restart_mqtt_fn is None:
            raise HTTPException(status_code=501, detail="MQTT restart not available in this mode")
        try:
            restart_mqtt_fn()
            return {"result": True}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/global-settings/mqtt-supervisor")
    def get_mqtt_supervisor_credentials():
        """Fetch MQTT credentials from the HA Supervisor API (only works as HA addon)."""
        sup_token = os.environ.get('SUPERVISOR_TOKEN')
        if not sup_token:
            raise HTTPException(status_code=503, detail="Supervisor token not available")
        try:
            req = urllib.request.Request(
                "http://supervisor/services/mqtt",
                headers={"Authorization": f"Bearer {sup_token}"}
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = json.loads(resp.read().decode())
            # Supervisor wraps the payload under a "data" key
            data = body.get("data", body)
            return {
                "mqtt_broker": data.get("host"),
                "mqtt_port": data.get("port"),
                "mqtt_username": data.get("username"),
                "mqtt_password": data.get("password"),
            }
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Failed to fetch supervisor MQTT credentials: {e}")

    # --- Flash device endpoints ---

    class FlashBuildRequest(BaseModel):
        config: dict

    @app.post("/api/flash/prepare", dependencies=[Depends(authenticate)])
    async def flash_prepare():
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, clone_or_pull_firmware)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/flash/kconfig", dependencies=[Depends(authenticate)])
    def flash_kconfig():
        if not firmware_exists():
            raise HTTPException(
                status_code=404,
                detail="Firmware not cloned yet. Call /api/flash/prepare first.",
            )
        try:
            options = get_kconfig_options()
            current_values = read_current_sdkconfig()
            return {"options": options, "current_values": current_values}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/flash/build", dependencies=[Depends(authenticate)])
    async def flash_build(req: FlashBuildRequest):
        if not firmware_exists():
            raise HTTPException(status_code=404, detail="Firmware not cloned yet.")

        async def _stream():
            async for chunk in build_firmware_stream(req.config):
                yield chunk

        return StreamingResponse(
            _stream(),
            media_type="text/plain",
            headers={
                "X-Accel-Buffering": "no",   # disable Nginx/HA-ingress proxy buffering
                "Cache-Control": "no-cache",
            },
        )

    @app.get("/api/flash/flash-args", dependencies=[Depends(authenticate)])
    def flash_get_flash_args():
        try:
            return get_flash_args()
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/flash/binaries/{filepath:path}", dependencies=[Depends(authenticate)])
    def flash_binary(filepath: str):
        build_dir = os.path.normpath(os.path.join(FIRMWARE_DIR, "build"))
        target = os.path.normpath(os.path.join(build_dir, filepath))
        if not target.startswith(build_dir + os.sep) and target != build_dir:
            raise HTTPException(status_code=403, detail="Access denied")
        if not os.path.isfile(target):
            raise HTTPException(status_code=404, detail="Binary not found")
        return FileResponse(
            target,
            media_type="application/octet-stream",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/")
    async def serve_index():
        file_path = os.path.join("frontend/dist", "index.html")
        response = FileResponse(file_path)
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, proxy-revalidate"
        return response

    app.mount("/assets", StaticFiles(directory="frontend/dist/assets"), name="assets")

    log("[HTTP] Setup complete.")
    return app
