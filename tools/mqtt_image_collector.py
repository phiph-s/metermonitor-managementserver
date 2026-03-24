from lib.log import log
import paho.mqtt.client as mqtt
import json
import base64
import os
from datetime import datetime

# MQTT Configuration
MQTT_BROKER = "192.168.178.24"
MQTT_PORT = 1883
MQTT_USERNAME = "esp"
MQTT_PASSWORD = "esp"
MQTT_TOPIC = "MeterMonitor/Hauptzaehler/"  # Use wildcard to receive all subtopics

# Folder to save images
SAVE_FOLDER = "collected_images"
os.makedirs(SAVE_FOLDER, exist_ok=True)


# MQTT Callback when connected
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        log("✅ Connected to MQTT Broker!")
        client.subscribe(MQTT_TOPIC)  # Subscribe inside on_connect
        log(f"📡 Subscribed to topic: {MQTT_TOPIC}")
    else:
        log(f"⚠️ Connection failed with code {rc}")


# MQTT Callback when a message is received
def on_message(client, userdata, msg):
    log(f"📩 Received message on {msg.topic}")  # Debugging

    try:
        payload = json.loads(msg.payload.decode("utf-8"))

        # Extract image data
        picture_info = payload.get("picture", {})
        timestamp = picture_info.get("timestamp", "")
        image_data = picture_info.get("data", "")

        if timestamp and image_data:
            # Convert timestamp to a valid filename format
            filename = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S").strftime("%Y%m%d_%H%M%S") + ".jpeg"
            file_path = os.path.join(SAVE_FOLDER, filename)

            # Decode Base64 image data
            image_bytes = base64.b64decode(image_data)

            # Save the image
            with open(file_path, "wb") as img_file:
                img_file.write(image_bytes)

            log(f"✅ Image saved: {file_path}")

    except Exception as e:
        log(f"⚠️ Error processing message: {e}")


# MQTT Setup
client = mqtt.Client()
client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
client.on_connect = on_connect
client.on_message = on_message

# Connect and subscribe
client.connect(MQTT_BROKER, MQTT_PORT, 60)

# Start loop
log("🚀 Listening for messages...")
client.loop_forever()
