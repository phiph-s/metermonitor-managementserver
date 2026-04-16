from lib.log import log
import datetime
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
        # Use singleton instance (shared with HTTP server)
        self.meter_preditor = get_meter_predictor()
        log("[MQTT] Using shared meter predictor singleton instance.")

    # On connect, remove the alert for the frontend
    # Also publish registration messages for all known watermeters

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            log("[MQTT] Successfully connected to MQTT broker")
            remove_alert("mqtt")
        else:
            log(f"[MQTT] Connection failed with code {reason_code}")
            add_alert("mqtt", "Failed to connect to MQTT broker")
            self._reconnect()
            return

        # Re-subscribe on every connect (handles reconnect after disconnect)
        if self.topic:
            self.client.subscribe(self.topic)
            log(f"[MQTT] Subscribed to topic '{self.topic}'")

        # send registration message for all watermeters
        with sqlite3.connect(self.db_file, timeout=30) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM watermeters")
            rows = cursor.fetchall()
            for row in rows:
                publish_registration(self.client, self.config, row[0], "value")


    # On disconnect, add an alert for the frontend and try to reconnect
    def _on_disconnect(self, client, userdata, rc, properties=None, packet=None, reason=None):
        log(f"Disconnected with code {rc}")
        add_alert("mqtt", "Disconnected from MQTT broker")
        if self.should_reconnect:
            self._reconnect()

    # Reconnect with exponential backoff
    def _reconnect(self):
        """Attempts to reconnect with exponential backoff"""
        delay = 1  # Initial delay in seconds
        max_delay = 60  # Maximum delay to avoid too frequent reconnections

        add_alert("mqtt", "Reconnecting to MQTT broker")

        while self.should_reconnect:
            try:
                log(f"[MQTT] Reconnecting to MQTT broker...")
                self.client.reconnect()
                log("Reconnected successfully")
                remove_alert("mqtt")
                return  # Exit loop on success
            except Exception as e:
                log(f"[MQTT] Reconnect failed: {e}, retrying in {delay} seconds...")
                time.sleep(delay)
                delay = min(delay * 2, max_delay)  # Exponential backoff

    # Validate the incoming message
    def _on_message(self, client, userdata, msg):
        data = json.loads(msg.payload)
        self._process_message(data)

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

            # ISO timestamp
            try:
                return datetime.datetime.fromisoformat(value).isoformat()
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
    def _process_message(self, data: Dict[str, Any]):
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
            wifi_rssi = data.get('WiFi-RSSI')

            with sqlite3.connect(self.db_file, timeout=30) as conn:

                cursor = conn.cursor()
                #check if watermeter exists
                cursor.execute("SELECT picture_number FROM watermeters WHERE name = ?", (data['name'],))
                row = cursor.fetchone()
                meter_exists = row is not None

                if not meter_exists:
                    cursor.execute('''
                        INSERT INTO watermeters (name, picture_number, wifi_rssi, picture_format, picture_timestamp, picture_width, picture_height, picture_length, picture_data, setup, picture_data_bbox)
                        VALUES (?,?,?,?,?,?,?,?,?,?,NULL)
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
                        0
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

                    publish_registration(self.client, self.config, data['name'], "value")
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
                        data['name']
                    ))

                # Ensure MQTT source entry exists for this meter (unambiguous source tracking)
                cursor.execute(
                    "SELECT 1 FROM sources WHERE name = ? AND source_type = 'mqtt'",
                    (data['name'],),
                )
                if cursor.fetchone() is None:
                    cursor.execute(
                        "INSERT INTO sources (name, source_type, enabled, poll_interval_s, config_json, updated_ts) "
                        "VALUES (?, 'mqtt', 1, NULL, NULL, datetime('now'))",
                        (data['name'],),
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

    def stop(self):
        self.client.loop_stop()
        self.client.disconnect()
