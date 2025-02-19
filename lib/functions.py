import base64
import sqlite3
import json
from datetime import datetime

from PIL import Image
from io import BytesIO

from tensorflow import timestamp

from lib.history_correction import correct_value


def reevaluate_latest_picture(db_file: str, name:str, meter_preditor, config):
    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()

        # get last picture
        # get latest image from watermeter
        cursor.execute("SELECT picture_data, picture_timestamp, setup FROM watermeters WHERE name = ? ORDER BY picture_number DESC LIMIT 1", (name,))
        row = cursor.fetchone()
        if not row:
            conn.commit()
            return None
        image_data = base64.b64decode(row[0])
        timestamp = row[1]
        setup = row[2] == 1

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
            print(f"Meter-Eval: No thresholds found for {name}")
        else:
            processed, digits = meter_preditor.apply_thresholds(digits, thresholds, invert=settings[5])
            prediction = meter_preditor.predict_digits(digits)

        value = None
        if setup:
            value = correct_value(db_file, name, [result, processed, prediction, timestamp])
            if value is not None:
                cursor.execute('''
                    INSERT INTO history
                    VALUES (?,?,?,?)
                ''', (
                    name,
                    value,
                    timestamp,
                    False
                ))

                # remove old entries (keep 30)
                cursor.execute('''
                    DELETE FROM history
                    WHERE name = ?
                    AND ROWID NOT IN (
                        SELECT ROWID
                        FROM history
                        WHERE name = ?
                        ORDER BY ROWID DESC
                        LIMIT ?
                    )
                ''', (name, name, config['max_history']))

        cursor.execute('''
                   INSERT INTO evaluations
                   VALUES (?,?)
               ''', (
            name,
            json.dumps([result, processed, prediction, timestamp, value])
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
                       LIMIT ?
                   )
               ''', (name, name, config['max_evals']))

        conn.commit()
        print(f"Meter-Eval: Prediction saved for {name}")

def add_history_entry(db_file: str, name: str, value: int, timestamp: str, config, manual: bool = False):
    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO history
            VALUES (?,?,?,?)
        ''', (
            name,
            value,
            timestamp,
            manual
        ))

        # remove old entries (keep 30)
        cursor.execute('''
            DELETE FROM history
            WHERE name = ?
            AND ROWID NOT IN (
                SELECT ROWID
                FROM history
                WHERE name = ?
                ORDER BY ROWID DESC
                LIMIT ?
            )
        ''', (name, name, config['max_history']))

        conn.commit()
        print(f"Meter-Eval: History entry added for {name}")