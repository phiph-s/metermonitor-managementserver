import time

import paho.mqtt.client as mqtt
import json
import sqlite3
from typing import Dict, Any

from lib.functions import reevaluate_latest_picture, publish_registration
from lib.meter_processing.meter_processing import MeterPredictor


class MQTTHandler:
    def __init__(self,config, db_file: str = 'watermeters.db', forever: bool = False):
        self.db_file = db_file
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.config = config
        self.forever = forever
        self.should_reconnect = True
        self.meter_preditor = MeterPredictor(
            yolo_model_path = "models/yolo-best-obb.pt",
            digit_classifier_model_path = "models/digit_classifier.pth"
        )
        print("MQTT-Handler: Loaded MQTT meter predictor.")

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            print("Successfully connected to MQTT broker")
        else:
            print(f"Connection failed with code {reason_code}")

        # send registration message for all watermeters
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM watermeters")
            rows = cursor.fetchall()
            for row in rows:
                publish_registration(self.client, self.config, row[0], "value")

    def _on_disconnect(self, client, userdata, rc, properties=None, packet=None, reason=None):
        print(f"Disconnected with code {rc}")
        if self.should_reconnect:
            self._reconnect()

    def _reconnect(self):
        """Attempts to reconnect with exponential backoff"""
        delay = 1  # Initial delay in seconds
        max_delay = 60  # Maximum delay to avoid too frequent reconnections

        while self.should_reconnect:
            try:
                print(f"Reconnecting to MQTT broker {self.broker}:{self.port}...")
                self.client.reconnect()
                print("Reconnected successfully")
                return  # Exit loop on success
            except Exception as e:
                print(f"Reconnect failed: {e}, retrying in {delay} seconds...")
                time.sleep(delay)
                delay = min(delay * 2, max_delay)  # Exponential backoff

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

    def _process_message(self, data: Dict[str, Any]):
        if not self._validate_message(data):
            print("Invalid message format")
            return

        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            #check if watermeter exists
            cursor.execute("SELECT * FROM watermeters WHERE name = ?", (data['name'],))
            if not cursor.fetchone():
                cursor.execute('''
                    INSERT INTO watermeters
                    VALUES (?,?,?,?,?,?,?,?,?,?)
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
                                VALUES (?,?,?,?,?,?,?)
                            ''', (
                    data['name'],
                    0,
                    100,
                    7,
                    False,
                    False,
                    False
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
                            picture_data = ?
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
            conn.commit()
            print(f"MQTT-Handler: Data saved for {data['name']}")
        reevaluate_latest_picture(self.db_file, data['name'], self.meter_preditor, self.config, publish=True, mqtt_client=self.client)

    def start(self,
              broker: str = 'localhost',
              port: int = 1883,
              topic: str = "MeterMonitor/#",
              username: str = None,
              password: str = None):

        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect

        if username and password:
            self.client.username_pw_set(username, password)

        self.client.connect(broker, port)
        self.client.subscribe(topic)
        if self.forever:
            self.client.loop_forever()
        else:
            self.client.loop_start()

    def stop(self):
        self.client.loop_stop()
        self.client.disconnect()