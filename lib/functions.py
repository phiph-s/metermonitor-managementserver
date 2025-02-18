import base64
import sqlite3
import json
from PIL import Image
from io import BytesIO

def reevaluate_latest_picture(db_file: str, name:str, meter_preditor):
    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()

        # get last picture
        # get latest image from watermeter
        cursor.execute("SELECT picture_data FROM watermeters WHERE name = ? ORDER BY picture_number DESC LIMIT 1", (name,))
        row = cursor.fetchone()
        if not row:
            conn.commit()
            return None
        image_data = base64.b64decode(row[0])

        cursor.execute('''
                   SELECT threshold_low, threshold_high, segments, shrink_last_3, extended_last_digit, invert
                   FROM settings
                   WHERE name = ?
               ''', (name,))
        settings = cursor.fetchone()
        thresholds = [settings[0], settings[1]]

        image = Image.open(BytesIO(image_data))
        result, digits = meter_preditor.predict_single_image(image, segments=settings[2], shrink_last_3=settings[3],
                                                                  extended_last_digit=settings[4])
        processed = []
        prediction = []
        if len(thresholds) == 0:
            print(f"MQTT-Handler: No thresholds found for {name}")
        else:
            processed, digits = meter_preditor.apply_thresholds(digits, thresholds, invert=settings[5])
            prediction = meter_preditor.predict_digits(digits)
        cursor.execute('''
                   INSERT INTO evaluations
                   VALUES (?,?)
               ''', (
            name,
            json.dumps([result, processed, prediction])
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
               ''', (name, name))

        conn.commit()
        print(f"MQTT-Handler: Prediction saved for {name}")