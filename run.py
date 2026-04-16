from lib.log import log, configure as configure_log
import os
import sqlite3
import threading
from contextlib import asynccontextmanager

import uvicorn
import json
from fastapi import FastAPI

from db.migrations import run_migrations
from lib.http_server import prepare_setup_app
from lib.mqtt_handler import MQTTHandler
from lib.polling_handler import PollingHandler

config = {}

# parse settings.json or options.json
# options.json is used in the addon, while settings.json is used in standalone mode
# the addon will merge the options.json with the contents of ha_default_settings.json
#
# Optional override: set METERMONITOR_SETTINGS to a custom config file path
# (useful for tests or custom deployments).

# ha_default_settings.json contains settings that should not be changed by the user when running in Home Assistant

override_path = os.environ.get("METERMONITOR_SETTINGS")
if override_path:
    log(f"[INIT] Using config override from METERMONITOR_SETTINGS: {override_path}")
    if not os.path.exists(override_path):
        raise FileNotFoundError(f"Config override not found: {override_path}")
    with open(override_path, 'r') as f:
        config = json.load(f)
else:
    path = '/data/options.json'
    if not os.path.exists(path):
        log("[INIT] Running standalone, using settings.json")
        path = 'settings.json'
        with open(path, 'r') as f:
            config = json.load(f)
    else:
        log("[INIT] Running as Home Assistant addon, using options.json and merging with ha_default_settings.json")
        #load options.json
        with open(path, 'r') as f:
            config = json.load(f)

        # merge missing options in options.json with ha_default_settings.json (deep merge)
        with open('ha_default_settings.json', 'r') as f:
            defaults = json.load(f)

        def deep_merge(target, source):
            """Deep merge source into target, preserving non-None values in target"""
            for key, value in source.items():
                if key not in target:
                    target[key] = value
                elif isinstance(value, dict) and isinstance(target.get(key), dict):
                    deep_merge(target[key], value)
                elif target[key] is None and value is not None:
                    # Replace None with default value
                    target[key] = value
            return target

        config = deep_merge(config, defaults)

configure_log(config.get('log_template', '%t %m'))
log("[INIT] Loaded config:")
# pretty print json
log(json.dumps(config, indent=4))

# Run migrations
run_migrations(config['dbfile'])

MQTT_CONFIG = config['mqtt']

# Create an outbound-only MQTT client for publishing (used by polling/http capture paths).
# The inbound MQTT handler creates its own client and also subscribes.
_publisher_mqtt_client = None
try:
    import paho.mqtt.client as mqtt

    _publisher_mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    if MQTT_CONFIG.get('username') and MQTT_CONFIG.get('password'):
        _publisher_mqtt_client.username_pw_set(MQTT_CONFIG.get('username'), MQTT_CONFIG.get('password'))
    _publisher_mqtt_client.connect(MQTT_CONFIG.get('broker', 'localhost'), int(MQTT_CONFIG.get('port', 1883)))
    _publisher_mqtt_client.loop_start()
    log("[INIT] MQTT publisher client connected (for polling/http sources)")
except Exception as e:
    log(f"[INIT] MQTT publisher client not available: {e}")
    _publisher_mqtt_client = None

# start application. if http is enabled, start the http server
# if not, start only the mqtt handler

# start polling service
polling_handler = PollingHandler(config, db_file=config['dbfile'], mqtt_client=_publisher_mqtt_client)
polling_handler.start()

if config['http']['enabled']:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        def run_mqtt():
            mqtt_handler = MQTTHandler(config, db_file=config['dbfile'], forever=True)
            mqtt_handler.start(**MQTT_CONFIG)

        thread = threading.Thread(target=run_mqtt, daemon=True)
        thread.start()
        yield

    app = prepare_setup_app(config, lifespan)
    log(f"[INIT] Started setup server on http://{config['http']['host']}:{config['http']['port']}")
    uvicorn.run(app, host=config['http']['host'], port=config['http']['port'], log_level="error")

else:
    mqtt_handler = MQTTHandler(config, db_file=config['dbfile'], forever=True)
    mqtt_handler.start(**MQTT_CONFIG)
