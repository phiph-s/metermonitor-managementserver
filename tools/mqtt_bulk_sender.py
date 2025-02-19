import os
import re
import time
import json
import base64
from PIL import Image
import paho.mqtt.client as mqtt

# MQTT Configuration
MQTT_BROKER = "192.168.178.24"  # Change this to your MQTT broker
MQTT_PORT = 1883
MQTT_USERNAME = "esp"  # Set your MQTT username
MQTT_PASSWORD = "esp"  # Set your MQTT password
MQTT_TOPIC = "MeterMonitor/upload"

# Image Filename Patterns
ISO_TIMESTAMP_REGEX = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"  # Matches "2024-11-02T18:54:48"
UNIX_TIMESTAMP_REGEX = r"^pic_\d{10}"  # Matches "pic_1737507953" (10-digit Unix timestamp)


def get_image_files():
    """List image files in the current directory that match timestamp patterns."""
    files = os.listdir(".")
    image_files = []

    for file in files:
        if re.match(ISO_TIMESTAMP_REGEX, file) or re.match(UNIX_TIMESTAMP_REGEX, file):
            image_files.append(file)

    return sorted(image_files)  # Sort to ensure correct order


def encode_image_to_base64(image_path):
    """Reads an image and returns its base64-encoded data and file length."""
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode('utf-8'), len(data)


def get_image_dimensions(image_path):
    """Gets the width and height of an image using Pillow."""
    with Image.open(image_path) as img:
        return img.width, img.height


def extract_timestamp(file_name):
    """Extracts a timestamp from the filename."""
    if re.match(ISO_TIMESTAMP_REGEX, file_name):
        return os.path.splitext(file_name)[0]  # Use the filename as timestamp
    elif re.match(UNIX_TIMESTAMP_REGEX, file_name):
        return file_name.split("_")[1].split(".")[0]  # Extract the Unix timestamp after 'pic_'
    return "unknown"


def build_message(file_path, picture_number):
    """Creates a JSON message for an image."""
    timestamp = extract_timestamp(os.path.basename(file_path))
    encoded_data, length = encode_image_to_base64(file_path)
    width, height = get_image_dimensions(file_path)

    # if timestamp isnt iso format, convert it
    if not re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", timestamp):
        timestamp = time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(int(timestamp)))

    return {
        "name": "HauptzaehlerStream2",
        "picture_number": picture_number,
        "WiFi-RSSI": -57,
        "picture": {
            "format": "jpeg",
            "timestamp": timestamp,
            "width": width,
            "height": height,
            "length": length,
            "data": encoded_data
        }
    }


def main():
    # Get images from the current folder
    images = get_image_files()
    if not images:
        print("No images matching the pattern were found.")
        return

    # Set up MQTT client with authentication
    client = mqtt.Client()
    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)  # Set MQTT credentials
    client.connect(MQTT_BROKER, MQTT_PORT, 60)

    print(f"Found {len(images)} images. Sending the first image immediately...")

    # Send the first image
    msg = build_message(images[0], picture_number=1)
    client.publish(MQTT_TOPIC, json.dumps(msg))
    print(f"Sent image: {images[0]}")

    # Wait for user input before sending the rest
    input("Press Enter to send the remaining images...")

    # Send remaining images with a 1-second delay
    for idx, image_file in enumerate(images[1:], start=2):
        time.sleep(1)
        msg = build_message(image_file, picture_number=idx)
        client.publish(MQTT_TOPIC, json.dumps(msg))
        print(f"Sent image: {image_file}")

    print("All images have been sent.")


if __name__ == "__main__":
    main()
