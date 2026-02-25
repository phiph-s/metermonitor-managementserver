import datetime
import time

import paho.mqtt.client as mqtt
import json
import sqlite3
from typing import Dict, Any

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
        # Use singleton instance (shared with HTTP server)
        self.meter_preditor = get_meter_predictor()
        print("[MQTT] Using shared meter predictor singleton instance.")

    # On connect, remove the alert for the frontend
    # Also publish registration messages for all known watermeters

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            print("[MQTT] Successfully connected to MQTT broker")
            remove_alert("mqtt")
        else:
            print(f"[MQTT] Connection failed with code {reason_code}")
            add_alert("mqtt", "Failed to connect to MQTT broker")
            self._reconnect()
            return

        # send registration message for all watermeters
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM watermeters")
            rows = cursor.fetchall()
            for row in rows:
                publish_registration(self.client, self.config, row[0], "value")


    # On disconnect, add an alert for the frontend and try to reconnect
    def _on_disconnect(self, client, userdata, rc, properties=None, packet=None, reason=None):
        print(f"Disconnected with code {rc}")
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
                print(f"[MQTT] Reconnecting to MQTT broker...")
                self.client.reconnect()
                print("Reconnected successfully")
                remove_alert("mqtt")
                return  # Exit loop on success
            except Exception as e:
                print(f"[MQTT] Reconnect failed: {e}, retrying in {delay} seconds...")
                time.sleep(delay)
                delay = min(delay * 2, max_delay)  # Exponential backoff

    # Validate the incoming message
    def _on_message(self, client, userdata, msg):
        data = json.loads(msg.payload)
        self._process_message(data)

    def _validate_message(self, data: Dict[str, Any]) -> bool:
        # Erforderliche Top-Level Felder
        required_fields = {'name', 'picture_number', 'WiFi-RSSI', 'picture'}
        if not all(field in data for field in required_fields):
            return False

        # Erforderliche Felder im 'picture' Objekt
        required_picture_fields = {
            'timestamp',
            'format',
            'width',
            'height',
            'length',
            'data'
        }

        if not isinstance(data['picture'], dict):
            return False

        if not all(field in data['picture'] for field in required_picture_fields):
            return False

        return True

    # Process the incoming message
    def _process_message(self, data: Dict[str, Any]):
        try:
            if not self._validate_message(data):
                print(f"[MQTT] Invalid message format received at {datetime.datetime.now().isoformat()}: {data}")
                return

            print(f"[MQTT] Received message for watermeter {data['name']}")


            # Check if timestamp is 0 or null, if so set it to current time
            if not data['picture']['timestamp']or data['picture']['timestamp'] == "0":
                # current iso time
                data['picture']['timestamp'] = datetime.datetime.now().isoformat()
                print(f"[MQTT] Timestamp was missing or zero, set to current time for {data['name']} ({data['picture']['timestamp']})")

            with sqlite3.connect(self.db_file) as conn:

                cursor = conn.cursor()
                #check if watermeter exists
                cursor.execute("SELECT * FROM watermeters WHERE name = ?", (data['name'],))
                meter_exists = cursor.fetchone() is not None

                if not meter_exists:
                    cursor.execute('''
                        INSERT INTO watermeters (name, picture_number, wifi_rssi, picture_format, picture_timestamp, picture_width, picture_height, picture_length, picture_data, setup, picture_data_bbox)
                        VALUES (?,?,?,?,?,?,?,?,?,?,NULL)
                    ''', (
                        data['name'],
                        data['picture_number'],
                        data['WiFi-RSSI'],
                        data['picture']['format'],
                        data['picture']['timestamp'],
                        data['picture']['width'],
                        data['picture']['height'],
                        data['picture']['length'],
                        data['picture']['data'],
                        0
                    ))
                    cursor.execute('''
                                    INSERT OR IGNORE INTO settings
                                    (name, threshold_low, threshold_high, threshold_last_low, threshold_last_high,
                                     islanding_padding, segments, rotated_180, shrink_last_3, extended_last_digit,
                                     max_flow_rate, conf_threshold, roi_extractor, template_id, segment_mode, use_correctional_alg)
                                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
                        True
                    ))

                    publish_registration(self.client, self.config, data['name'], "value")
                else:
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
                        data['picture_number'],
                        data['WiFi-RSSI'],
                        data['picture']['format'],
                        data['picture']['timestamp'],
                        data['picture']['width'],
                        data['picture']['height'],
                        data['picture']['length'],
                        data['picture']['data'],
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
                    print(f"[MQTT] Source for watermeter {data['name']} is disabled or does not exist")
                    return
                
                print(f"[MQTT] Saved/updated metadata of {data['name']} to database.")
                _, _, boundingboxed_image = reevaluate_latest_picture(self.db_file, data['name'], self.meter_preditor,
                                                                      self.config, publish=True,
                                                                      mqtt_client=self.client)
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
                    print(f"[MQTT] Saved boundingboxed image of {data['name']} to database.")

        except Exception as e:
            print(f"[MQTT] Error processing message: {e}")
            # print traceback
            traceback.print_exc()

    # Start the MQTT client
    def start(self,
              broker: str = 'localhost',
              port: int = 1883,
              topic: str = "MeterMonitor/#",
              username: str = None,
              password: str = None):

        add_alert("mqtt", "Connecting to MQTT broker")

        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect

        if username and password:
            self.client.username_pw_set(username, password)

        try:
            self.client.connect(broker, port)
        except Exception as e:
            print(f"[MQTT] Error connecting to MQTT broker: {e}")
            add_alert("mqtt", f"Failed to connect to MQTT broker: {e}")
            return
        self.client.subscribe(topic)
        if self.forever:
            self.client.loop_forever()
        else:
            self.client.loop_start()

    def stop(self):
        self.client.loop_stop()
        self.client.disconnect()
