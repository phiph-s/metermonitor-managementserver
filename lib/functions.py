from lib.log import log
import base64
import sqlite3
import json

from PIL import Image
from io import BytesIO
import numpy as np
import cv2

from lib.history_correction import correct_value
from lib.meter_processing.roi_extractors.orb_extractor import ORBExtractor
from lib.realtime_events import evaluation_event_hub

def _normalize_digit_models(raw_models, digit_count: int):
    if not raw_models:
        return ["rotating"] * digit_count
    if isinstance(raw_models, str):
        try:
            parsed = json.loads(raw_models)
        except Exception:
            parsed = []
    else:
        parsed = raw_models
    if not isinstance(parsed, list):
        parsed = []
    models = []
    for i in range(digit_count):
        model = parsed[i] if i < len(parsed) else None
        if model not in {"rotating", "segment"}:
            model = "rotating"
        models.append(model)
    return models


def reevaluate_digits(db_file: str, name: str, meter_preditor, config, offset: int = None):
    with sqlite3.connect(db_file, timeout=30) as conn:
        cursor = conn.cursor()

        # Get eval from the database - either by offset or last
        if offset == -1:
            cursor.execute('''
                SELECT colored_digits FROM evaluations
                WHERE name = ?
                ORDER BY RANDOM()
                LIMIT 1
            ''', (name,))
        elif offset is not None:
            cursor.execute('''
                SELECT colored_digits FROM evaluations
                WHERE name = ?
                ORDER BY id DESC
                LIMIT 1 OFFSET ?
            ''', (name, offset))
        else:
            cursor.execute('''
                SELECT colored_digits FROM evaluations
                WHERE name = ?
                ORDER BY id DESC
                LIMIT 1
            ''', (name,))
        row = cursor.fetchone()
        if not row:
            return {"error": "No evaluations found"}

        # parse the eval (json)
        raw_images = json.loads(row[0])

        # convert to np arrays (from base64)
        digits = []
        for raw_image in raw_images:
            image_data = base64.b64decode(raw_image)
            image = Image.open(BytesIO(image_data))
            # Convert from RGB (stored in DB) to BGR (expected by apply_threshold)
            digit_array = np.array(image)
            if len(digit_array.shape) == 3:
                digit_array = cv2.cvtColor(digit_array, cv2.COLOR_RGB2BGR)
            digits.append(digit_array)

        # Get current settings for the watermeter
        cursor.execute('''
                       SELECT threshold_low,
                              threshold_high,
                              threshold_last_low,
                              threshold_last_high,
                              islanding_padding,
                              segments,
                              shrink_last_3,
                              extended_last_digit,
                              max_flow_rate,
                              rotated_180,
                              digit_models,
                              decimals
                       FROM settings
                       WHERE name = ?
                       ''', (name,))
        settings = cursor.fetchone()
        if not settings:
            return {"error": "Error fetching settings"}
        thresholds = [settings[0], settings[1]]
        thresholds_last = [settings[2], settings[3]]
        islanding_padding = settings[4]
        digit_models = settings[10] if len(settings) > 10 else None
        decimals = settings[11] if len(settings) > 11 and settings[11] is not None else 3
        model_types = _normalize_digit_models(digit_models, len(digits))

        if len(thresholds) == 0:
            log(f"[Eval ({name})] No thresholds found for {name}")
            return {"error": "No thresholds found"}
        else:
            processed, digits, digits_inverted = meter_preditor.apply_thresholds(digits, thresholds, thresholds_last, islanding_padding, decimals=decimals)
            prediction = meter_preditor.predict_digits(digits, model_types=model_types)

        return {
            "processed_images": digits_inverted,
            "predictions": prediction
        }




# This file reevaluates the latest picture of a watermeter and saves the result in the database.
def reevaluate_latest_picture(db_file: str, name:str, meter_preditor, config, publish: bool = False, skip_setup_overwriting = True, mqtt_client = None, notify_realtime: bool = False):
    with sqlite3.connect(db_file, timeout=30) as conn:
        cursor = conn.cursor()
        meter_preditor.last_error = None

        # get latest image from watermeter
        cursor.execute("SELECT picture_data, picture_timestamp, setup FROM watermeters WHERE name = ? ORDER BY picture_number DESC LIMIT 1", (name,))
        row = cursor.fetchone()
        if not row:
            conn.commit()
            log(f"[Eval ({name})] No picture found for {name}")
            return None
        image_data = base64.b64decode(row[0])
        timestamp = row[1]
        setup = row[2] == 1

        # Get current settings for the watermeter
        cursor.execute('''
                   SELECT threshold_low, threshold_high, threshold_last_low, threshold_last_high, islanding_padding,
                    segments, shrink_last_3, extended_last_digit, max_flow_rate, rotated_180, conf_threshold, roi_extractor, template_id, segment_mode, digit_models, decimals, use_correctional_alg
                   FROM settings
                   WHERE name = ?
               ''', (name,))
        settings = cursor.fetchone()
        thresholds = [settings[0], settings[1]]
        thresholds_last = [settings[2], settings[3]]
        islanding_padding = settings[4]
        segments = settings[5]
        shrink_last_3 = settings[6]
        extended_last_digit = settings[7]
        max_flow_rate = settings[8]
        rotated_180 = settings[9]
        conf_threshold = settings[10] if settings[10] else 0.0
        roi_extractor = settings[11] if settings[11] else "yolo"
        template_id = settings[12] if settings[12] else None
        segment_mode = settings[13] if settings[13] else "display"
        digit_models = settings[14] if len(settings) > 14 else None
        decimals = settings[15] if len(settings) > 15 and settings[15] is not None else 3
        use_correctional_alg = bool(settings[16]) if settings[16] is not None else True

        # Get the target_brightness from the last history entry
        cursor.execute("SELECT target_brightness FROM history WHERE name = ? ORDER BY ROWID DESC LIMIT 1", (name,))
        row = cursor.fetchone()
        target_brightness = None
        if row:
            target_brightness = row[0]
        conn.commit()
        image = Image.open(BytesIO(image_data))

        # Use the meter predictor to extract the digits from the image
        extractor_instance = None
        if roi_extractor in {"orb", "static_rect"}:
            if not template_id:
                log(f"[Eval ({name})] Template required for extractor '{roi_extractor}' but none is set.")
                meter_preditor.last_error = f"Template required for extractor '{roi_extractor}'."
                return None
            try:
                if roi_extractor == "orb":
                    extractor_instance = ORBExtractor.from_database(conn, template_id)
                elif roi_extractor == "static_rect":
                    from lib.meter_processing.roi_extractors.static_rect_extractor import StaticRectExtractor
                    extractor_instance = StaticRectExtractor.from_database(conn, template_id)
            except Exception as e:
                log(f"[Eval ({name})] Failed to load template '{template_id}': {e}")
                meter_preditor.last_error = f"Failed to load template: {e}"
                return None

        result, digits, target_brightness, boundingboxed_image = meter_preditor.extract_display_and_segment(
            image,
            segments=segments,
            shrink_last_3=shrink_last_3,
            extended_last_digit=extended_last_digit,
            rotated_180=rotated_180,
            target_brightness=target_brightness,
            roi_extractor=roi_extractor,
            extractor_instance=extractor_instance,
            segment_mode=segment_mode
        )

        if not result or len(result) == 0:
            log(f"[Eval ({name})] No result found")
            if not meter_preditor.last_error:
                meter_preditor.last_error = "No result found"
            return None

        # Apply thresholds and extract the digits
        processed = []
        prediction = []
        digits_inverted = []
        if len(thresholds) == 0:
            log(f"[Eval ({name})] No thresholds found for {name}")
        else:
            processed, digits, digits_inverted = meter_preditor.apply_thresholds(digits, thresholds, thresholds_last, islanding_padding, decimals=decimals)
            model_types = _normalize_digit_models(digit_models, len(digits))
            prediction = meter_preditor.predict_digits(digits, model_types=model_types)

        # check for each digit if its highest conf is above conf_threshold, otherwise mark it as denied
        denied_digits = []
        for digit_predictions in prediction:
            if len(digit_predictions) == 0 or digit_predictions[0][1]*100 < conf_threshold:
                denied_digits.append(True)
            else:
                denied_digits.append(False)

        # If the setup is finished, try to correct the value and save the result
        value = None
        confidence, used_confidence = 0, 0
        correction_meta = {
            "flow_rate_m3h": None,
            "delta_m3": None,
            "delta_raw": None,
            "time_diff_min": None,
            "rejection_reason": None,
            "negative_correction_applied": None,
            "fallback_digit_count": None,
            "digits_changed_vs_last": None,
            "digits_changed_vs_top_pred": None,
            "prediction_rank_used_counts": None,
            "denied_digits_count": None,
            "timestamp_adjusted": None
        }
        if setup:
            correction = correct_value(db_file, name, [result, processed, prediction, timestamp, denied_digits], allow_negative_correction=config["allow_negative_correction"], max_flow_rate=max_flow_rate, use_full_correction=use_correctional_alg, decimals=decimals)
            correction_meta = {k: correction.get(k) for k in correction_meta.keys()}
            confidence = correction.get("total_confidence", 0.0)
            used_confidence = correction.get("used_confidence", 0.0)
            if correction.get("accepted"):
                value = correction.get("value")
                cursor.execute('''
                    INSERT INTO history (name, value, confidence, used_confidence, target_brightness, timestamp, manual)
                    VALUES (?,?,?,?,?,?,?)
                ''', (
                    name,
                    value,
                    confidence,
                    used_confidence,
                    target_brightness,
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

                if publish and mqtt_client:
                    publish_value(mqtt_client, config, name, value, decimals=decimals)

        curser = cursor.execute('''
            SELECT COUNT(*) FROM evaluations  
            WHERE name = ?
        ''', (name,))
        count = curser.fetchone()[0]


        if not skip_setup_overwriting and count > 0:
            # find id of last evaluation
            cursor.execute('''
                SELECT id FROM evaluations
                WHERE name = ?
                ORDER BY id DESC
                LIMIT 1
            ''', (name,))
            row = cursor.fetchone()
            eval_id = row[0]

            # replace the last evaluation if setup is not finished instead of adding a new one
            cursor.execute('''
                           UPDATE evaluations
                           SET colored_digits = ?,
                               th_digits = ?,
                               predictions = ?,
                               timestamp = ?,
                               result = ?,
                               total_confidence = ?,
                               used_confidence = ?,
                               th_digits_inverted = ?,
                               flow_rate_m3h = ?,
                               delta_m3 = ?,
                               delta_raw = ?,
                               time_diff_min = ?,
                               rejection_reason = ?,
                               negative_correction_applied = ?,
                               fallback_digit_count = ?,
                               digits_changed_vs_last = ?,
                               digits_changed_vs_top_pred = ?,
                               prediction_rank_used_counts = ?,
                               denied_digits_count = ?,
                               timestamp_adjusted = ?
                           WHERE name = ? AND id = ?
                           ''', (
                               json.dumps(result) if result is not None else None,
                               json.dumps(processed) if processed is not None else None,
                               json.dumps(prediction) if prediction is not None else None,
                               timestamp if isinstance(timestamp, str) and timestamp.strip() else None,
                               value if value is not None else None,
                               float(confidence) if confidence is not None else None,
                               float(used_confidence) if used_confidence is not None else None,
                               json.dumps(digits_inverted),
                               correction_meta["flow_rate_m3h"],
                               correction_meta["delta_m3"],
                               correction_meta["delta_raw"],
                               correction_meta["time_diff_min"],
                               correction_meta["rejection_reason"],
                               correction_meta["negative_correction_applied"],
                               correction_meta["fallback_digit_count"],
                               correction_meta["digits_changed_vs_last"],
                               correction_meta["digits_changed_vs_top_pred"],
                               json.dumps(correction_meta["prediction_rank_used_counts"]) if correction_meta["prediction_rank_used_counts"] is not None else None,
                               correction_meta["denied_digits_count"],
                               correction_meta["timestamp_adjusted"],
                               name,
                               eval_id
                           ))
        else:
            cursor.execute('''
                           INSERT INTO evaluations
                           (name, colored_digits, th_digits, predictions, timestamp, result, total_confidence, used_confidence, denied_digits, th_digits_inverted,
                            flow_rate_m3h, delta_m3, delta_raw, time_diff_min, rejection_reason, negative_correction_applied, fallback_digit_count,
                            digits_changed_vs_last, digits_changed_vs_top_pred, prediction_rank_used_counts, denied_digits_count, timestamp_adjusted)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                           ''', (
                               name,
                               json.dumps(result) if result is not None else None,
                               json.dumps(processed) if processed is not None else None,
                               json.dumps(prediction) if prediction is not None else None,
                               timestamp if isinstance(timestamp, str) and timestamp.strip() else None,
                               value if value is not None else None,
                               float(confidence) if confidence is not None else None,
                               float(used_confidence) if used_confidence is not None else None,
                               json.dumps(denied_digits),
                               json.dumps(digits_inverted),
                               correction_meta["flow_rate_m3h"],
                               correction_meta["delta_m3"],
                               correction_meta["delta_raw"],
                               correction_meta["time_diff_min"],
                               correction_meta["rejection_reason"],
                               correction_meta["negative_correction_applied"],
                               correction_meta["fallback_digit_count"],
                               correction_meta["digits_changed_vs_last"],
                               correction_meta["digits_changed_vs_top_pred"],
                               json.dumps(correction_meta["prediction_rank_used_counts"]) if correction_meta["prediction_rank_used_counts"] is not None else None,
                               correction_meta["denied_digits_count"],
                               correction_meta["timestamp_adjusted"]
                           ))

        # remove old evaluations
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

        if notify_realtime and setup:
            evaluation_event_hub.notify({
                "type": "evaluation_created",
                "name": name,
                "timestamp": timestamp if isinstance(timestamp, str) else None
            })

        log(f"[Eval ({name})] Prediction saved")
        return target_brightness, confidence, boundingboxed_image

# Function to publish the value to the MQTT broker, compatible with Home Assistant
def publish_value(mqtt_client, config, name, value, decimals=3):
    # publish to topic
    topic = config["publish_to"].replace("{device}", name) + "value"
    scale = 10 ** int(decimals or 0)
    dict = {
        "value": int(value) / float(scale),
    }
    mqtt_client.publish(topic, json.dumps(dict), qos=1, retain=True)
    log(f"[Eval/MQTT ({name})] Value published ({value} m³)")

# Function to publish the registration to the MQTT broker, compatible with Home Assistant
def publish_registration(mqtt_client, config, name, type):
    # publish to topic
    topic = config["publish_to"].replace("{device}", name) + "config"
    dict = {
      "name": "Water usage",
      "state_topic": config["publish_to"].replace("{device}", name) + type,
      "unit_of_measurement": "m³",
      "device_class": "water",
      "unique_id": "watermeter_" + name,
      "value_template": "{{ value_json.value }}",
      "device": {
        "identifiers": ["watermeter_" + name],
        "name": name,
        "manufacturer": "DIY",
        "model": "WM-1",
        "sw_version": "1.0"
      }
    }
    mqtt_client.publish(topic, json.dumps(dict), qos=1, retain=True)
    log(f"[Eval/MQTT ({name})] HA compatible Registration published")

# Function to add a history entry to the database, removing old entries
def add_history_entry(db_file: str, name: str, value: int, confidence:int, target_brightness: float, timestamp: str, config, manual: bool = False):
    with sqlite3.connect(db_file, timeout=30) as conn:
        cursor = conn.cursor()

        # Before inserting, check if the previous history entry is from yesterday —
        # if so, snapshot it into daily_history (one row per day, never pruned).
        today = timestamp[:10]  # "YYYY-MM-DD"
        cursor.execute('''
            SELECT value, timestamp FROM history
            WHERE name = ?
            ORDER BY ROWID DESC LIMIT 1
        ''', (name,))
        prev = cursor.fetchone()
        if prev:
            prev_value, prev_ts = prev
            prev_date = prev_ts[:10] if prev_ts else None
            if prev_date and prev_date < today:
                cursor.execute('''
                    INSERT OR IGNORE INTO daily_history (name, value, date)
                    VALUES (?, ?, ?)
                ''', (name, prev_value, prev_date))

        cursor.execute('''
            INSERT INTO history (name, value, confidence, target_brightness, timestamp, manual)
            VALUES (?,?,?,?,?,?)
        ''', (
            name,
            value,
            confidence,
            target_brightness,
            timestamp,
            manual
        ))

        # Remove old detailed entries beyond max_history limit
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
        log(f"[Eval ({name})] History entry added")
