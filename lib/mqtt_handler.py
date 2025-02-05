import base64
from io import BytesIO

import paho.mqtt.client as mqtt
import json
import sqlite3
import os
from typing import Dict, Any

from PIL import Image

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
        try:
            data = json.loads(msg.payload)
            self._process_message(data)
        except Exception as e:
            print(f"Error processing message: {str(e)}")

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
            conn.commit()
            print(f"MQTT-Handler: Data saved for {data['name']}")
        image_data = base64.b64decode(data['picture']['data'])
        image = Image.open(BytesIO(image_data))
        result = self.meter_preditor.predict_single_image(image)
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO evaluations
                VALUES (?,?)
            ''', (
                data['name'],
                json.dumps(result)
            ))
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