import sqlite3
import threading
from contextlib import asynccontextmanager

import uvicorn

# check if args --setup is given
import argparse
import json
from fastapi import FastAPI

from lib.http_server import prepare_setup_app
from lib.mqtt_handler import MQTTHandler

parser = argparse.ArgumentParser()
parser.add_argument("--setup", action="store_true")
args = parser.parse_args()

# parse config.yaml
with open('/data/options.json', 'r') as file:
    config = json.load(file)

    db_connection = sqlite3.connect(config['dbfile'])
    cursor = db_connection.cursor()
    cursor.execute('''
                CREATE TABLE IF NOT EXISTS watermeters (
                    name TEXT PRIMARY KEY,
                    picture_number INTEGER,
                    wifi_rssi INTEGER,
                    picture_format TEXT,
                    picture_timestamp TEXT,
                    picture_width INTEGER,
                    picture_height INTEGER,
                    picture_length INTEGER,
                    picture_data TEXT,
                    setup BOOLEAN DEFAULT 0
                )
            ''')
    cursor.execute('''
                CREATE TABLE IF NOT EXISTS settings (
                    name TEXT PRIMARY KEY,
                    threshold_low INTEGER,
                    threshold_high INTEGER,
                    segments INTEGER,
                    shrink_last_3 BOOLEAN,
                    extended_last_digit BOOLEAN,
                    invert BOOLEAN,
                    FOREIGN KEY(name) REFERENCES watermeters(name)
                )
            ''')
    # Add evaluations table
    cursor.execute('''
                CREATE TABLE IF NOT EXISTS evaluations (
                    name TEXT,
                    eval TEXT,
                    FOREIGN KEY(name) REFERENCES watermeters(name)
                )
            ''')
    cursor.execute('''
                CREATE TABLE IF NOT EXISTS history (
                    name TEXT,
                    value INTEGER,
                    timestamp TEXT,
                    manual BOOLEAN,
                    FOREIGN KEY(name) REFERENCES watermeters(name)
                )
            ''')
    db_connection.commit()

    # MQTT Config
    # example yaml:
    # mqtt:
    #   broker: "localhost"
    #   port: 1883
    #   topic: "MeterMonitor/#"
    #   username: "user"
    #   password: "password"

    MQTT_CONFIG = config['mqtt']

    if args.setup:
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            def run_mqtt():
                mqtt_handler = MQTTHandler(config, db_file=config['dbfile'], forever=True)
                mqtt_handler.start(**MQTT_CONFIG)

            thread = threading.Thread(target=run_mqtt, daemon=True)
            thread.start()
            yield
            # stop mqtt_handler
            thread.stop()

        app = prepare_setup_app(config, lifespan)

        # print routes
        print(app.routes)

        uvicorn.run(app, host="0.0.0.0", port=8070)

    else:
        print(config)
        mqtt_handler = MQTTHandler(db_file=config['dbfile'], forever=True)
        mqtt_handler.start(**MQTT_CONFIG)
