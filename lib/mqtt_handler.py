import base64
from io import BytesIO

import paho.mqtt.client as mqtt
import json
import sqlite3
import os
from typing import Dict, Any

from PIL import Image
from ultralytics import settings

from lib.meter_processing.meter_processing import MeterPredictor


class MQTTHandler:
    def __init__(self, db_file: str = 'watermeters.db', forever: bool = False):
        self.db_file = db_file
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.forever = forever
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

    def _on_disconnect(self, client, userdata, rc, properties=None, packet=None, reason=None):
        print(f"Disconnected with code {rc}")

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
            cursor.execute('''
                INSERT OR REPLACE INTO watermeters 
                VALUES (?,?,?,?,?,?,?,?,?)
            ''', (
                data['name'],
                data['picture_number'],
                data['WiFi-RSSI'],
                data['picture']['format'],
                data['picture']['timestamp'],
                data['picture']['width'],
                data['picture']['height'],
                data['picture']['length'],
                data['picture']['data']
            ))
            # also add default thresholds
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
            conn.commit()
            print(f"MQTT-Handler: Data saved for {data['name']}")
        # receive thresholds from database
        cursor.execute('''
            SELECT threshold_low, threshold_high, segments, shrink_last_3, extended_last_digit, invert
            FROM settings
            WHERE name = ?
        ''', (data['name'],))
        settings = cursor.fetchone()
        thresholds = [settings[0], settings[1]]

        image_data = base64.b64decode(data['picture']['data'])
        image = Image.open(BytesIO(image_data))
        result, digits = self.meter_preditor.predict_single_image(image, segments=settings[2], shrink_last_3=settings[3], extended_last_digit=settings[4])
        processed = []
        prediction = []
        if len(thresholds) == 0:
            print(f"MQTT-Handler: No thresholds found for {data['name']}")
        else:
            processed, digits = self.meter_preditor.apply_thresholds(digits, thresholds, invert = settings[5])
            prediction = self.meter_preditor.predict_digits(digits)
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO evaluations
                VALUES (?,?)
            ''', (
                data['name'],
                json.dumps([result,processed,prediction])
            ))
            # remove old evaluations (keep 5)
            cursor.execute('''
                DELETE FROM evaluations
                WHERE name = ?
                AND ROWID NOT IN (
                    SELECT ROWID
                    FROM evaluations
                    WHERE name = ?
                    ORDER BY ROWID DESC
                    LIMIT 5
                )
            ''', (data['name'], data['name']))

            conn.commit()
        print(f"MQTT-Handler: Prediction saved for {data['name']}")

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