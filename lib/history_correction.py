import sqlite3
from datetime import datetime

def correct_value(db_file: str, name: str, new_eval, allow_negative_correction = False, max_flow_rate = 1.0, use_full_correction = True, decimals: int = 3):
    # get last evaluation
    reject = False
    metadata = {
        "flow_rate_m3h": None,
        "delta_m3": None,
        "delta_raw": None,
        "time_diff_min": None,
        "rejection_reason": None,
        "negative_correction_applied": False,
        "fallback_digit_count": 0,
        "digits_changed_vs_last": 0,
        "digits_changed_vs_top_pred": 0,
        "prediction_rank_used_counts": [0, 0, 0],
        "denied_digits_count": 0,
        "timestamp_adjusted": False
    }
    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        segments = len(new_eval[2])
        # get last history entry
        cursor.execute("SELECT value, timestamp, confidence FROM history WHERE name = ? ORDER BY ROWID DESC LIMIT 2", (name,))
        rows = cursor.fetchall()
        if len(rows) == 0:
            denied_digits = new_eval[4]
            metadata["denied_digits_count"] = sum(denied_digits) if denied_digits is not None and len(denied_digits) > 0 else 0
            metadata["rejection_reason"] = "no_history"
            conn.commit()
            return {
                "accepted": False,
                "value": None,
                "total_confidence": 0.0,
                "used_confidence": 0.0,
                **metadata
            }
        row = rows[0]

        second_row = rows[1] if len(rows) > 1 else None

        last_value = str(row[0]).zfill(segments)
        if len(last_value) > segments:
            last_value = last_value[-segments:]
        last_time = datetime.fromisoformat(row[1])
        last_confidence = row[2]
        try:
            new_time = datetime.fromisoformat(new_eval[3])
        except Exception as e:
            print(f"[CorrectionAlg ({name})] Error parsing new evaluation time (assuming current): {e}")
            new_time = datetime.now()

        new_results = new_eval[2]
        denied_digits = new_eval[4]
        metadata["denied_digits_count"] = sum(denied_digits) if denied_digits is not None and len(denied_digits) > 0 else 0

        if last_time >= new_time:
            print(f"[CorrectionAlg ({name})] Time difference to last message is negative, assuming current time for correction")
            new_time = datetime.now()
            metadata["timestamp_adjusted"] = True

        max_flow_rate /= 60.0
        # get the time difference in minutes
        time_diff = (new_time - last_time).seconds / 60.0
        metadata["time_diff_min"] = time_diff

        if time_diff <= 0:
            metadata["rejection_reason"] = "time_diff_zero"
            return {
                "accepted": False,
                "value": None,
                "total_confidence": 0.0,
                "used_confidence": 0.0,
                **metadata
            }

        correctedValue = ""
        totalConfidence = 1.0
        # used confidence tracks the confidence of digits that actually are being funneled into the correction
        usedConfidence = 1.0
        negative_corrected = False
        digits_changed_vs_last = 0
        digits_changed_vs_top_pred = 0
        fallback_digit_count = 0
        prediction_rank_used_counts = [0, 0, 0]

        scale = 10 ** int(decimals or 0)

        # Light correction mode: only replace r/rotation and denied digits with last value
        if not use_full_correction:
            for i, lastChar in enumerate(last_value):
                predictions = new_results[i]
                if len(predictions) == 0:
                    correctedValue += lastChar
                    reject = True
                    fallback_digit_count += 1
                    continue

                prediction = predictions[0]
                prefix_increased = i > 0 and int(correctedValue) > int(last_value[:i])

                # Replace rotation class or denied digits with last value
                if prediction[0] == 'r' or denied_digits[i]:
                    # check if the digit before has changed upwards, set the digit to 0
                    if prefix_increased:
                        chosen_digit = '0'
                    else:
                        chosen_digit = lastChar
                    totalConfidence *= prediction[1]
                    if not denied_digits[i]:
                        usedConfidence *= prediction[1]
                    correctedValue += chosen_digit
                    if chosen_digit != lastChar:
                        digits_changed_vs_last += 1
                    if chosen_digit != prediction[0]:
                        digits_changed_vs_top_pred += 1
                    prediction_rank_used_counts[0] += 1
                else:
                    # Use predicted digit directly
                    chosen_digit = prediction[0]
                    totalConfidence *= prediction[1]
                    usedConfidence *= prediction[1]
                    correctedValue += chosen_digit
                    if chosen_digit != lastChar:
                        digits_changed_vs_last += 1
                    prediction_rank_used_counts[0] += 1

            # No flow rate or positive flow checks in light mode
            delta_raw = int(correctedValue) - int(last_value)
            delta_m3 = delta_raw / float(scale)
            flow_rate_m3min = delta_m3 / time_diff
            flow_rate_m3h = flow_rate_m3min * 60.0
            metadata["delta_raw"] = delta_raw
            metadata["delta_m3"] = delta_m3
            metadata["flow_rate_m3h"] = flow_rate_m3h
            metadata["negative_correction_applied"] = False
            metadata["fallback_digit_count"] = fallback_digit_count
            metadata["digits_changed_vs_last"] = digits_changed_vs_last
            metadata["digits_changed_vs_top_pred"] = digits_changed_vs_top_pred
            metadata["prediction_rank_used_counts"] = prediction_rank_used_counts

            if reject:
                metadata["rejection_reason"] = "fallback_digit"
                return {
                    "accepted": False,
                    "value": None,
                    "total_confidence": 0.0,
                    "used_confidence": 0.0,
                    **metadata
                }

            print(f"[CorrectionAlg LIGHT ({name})] Value accepted for time", new_time, "value", correctedValue)
            return {
                "accepted": True,
                "value": int(correctedValue),
                "total_confidence": totalConfidence,
                "used_confidence": usedConfidence,
                **metadata
            }

        # Full correction mode (original algorithm)
        for i, lastChar in enumerate(last_value):

            predictions = new_results[i]
            digit_appended = False
            prefix_increased = i > 0 and int(correctedValue) > int(last_value[:i])
            for prediction_index, prediction in enumerate(predictions):

                tempValue = correctedValue
                tempConfidence = totalConfidence

                # replacement of the rotation class
                if prediction[0] == 'r' or denied_digits[i]:
                    # check if the digit before has changed upwards, set the digit to 0
                    if prefix_increased:
                        chosen_digit = '0'
                        tempConfidence *= prediction[1]
                    else:
                        chosen_digit = lastChar
                        tempConfidence *= prediction[1]
                else:
                    chosen_digit = prediction[0]
                    tempConfidence *= prediction[1]
                tempValue += chosen_digit

                # check if the new value is higher than the last value (positive flow)
                if int(tempValue) >= int(last_value[:i+1]) or negative_corrected and tempConfidence > 0.15:
                    correctedValue = tempValue
                    totalConfidence = tempConfidence
                    if not denied_digits[i]: usedConfidence *= prediction[1]
                    digit_appended = True
                    if prediction_index < len(prediction_rank_used_counts):
                        prediction_rank_used_counts[prediction_index] += 1
                    if chosen_digit != lastChar:
                        digits_changed_vs_last += 1
                    if predictions is not None and len(predictions) > 0 and chosen_digit != predictions[0][0]:
                        digits_changed_vs_top_pred += 1
                    break

                # check conditions for negative correction
                elif allow_negative_correction:
                    if second_row:
                        pre_last_value = str(second_row[0]).zfill(segments)
                        if len(pre_last_value) > segments:
                            pre_last_value = pre_last_value[-segments:]
                        # if last history entry has a very low confidence, but current confidence is high enough
                        # compare with the second last entry
                        if last_confidence < 0.2 and tempConfidence > 0.50 and \
                                int(tempValue) >= int(pre_last_value[:i+1]):
                            correctedValue = tempValue
                            totalConfidence = tempConfidence
                            usedConfidence *= prediction[1]
                            digit_appended = True
                            negative_corrected = True
                            if prediction_index < len(prediction_rank_used_counts):
                                prediction_rank_used_counts[prediction_index] += 1
                            if chosen_digit != lastChar:
                                digits_changed_vs_last += 1
                            if predictions is not None and len(predictions) > 0 and chosen_digit != predictions[0][0]:
                                digits_changed_vs_top_pred += 1
                            print(f"[CorrectionAlg ({name})] Negative correction accepted")
                            break

            # if no digit was appended, append the original digit but reject the value
            if not digit_appended:
                correctedValue += lastChar
                reject = True
                fallback_digit_count += 1
                if predictions is not None and len(predictions) > 0 and lastChar != predictions[0][0]:
                    digits_changed_vs_top_pred += 1
                print(f"[CorrectionAlg ({name})] Fallback: appending original digit", lastChar)

        # get the flow rate and check if it is within the limits
        delta_raw = int(correctedValue) - int(last_value)
        delta_m3 = delta_raw / float(scale)
        flow_rate_m3min = delta_m3 / time_diff
        flow_rate_m3h = flow_rate_m3min * 60.0
        metadata["delta_raw"] = delta_raw
        metadata["delta_m3"] = delta_m3
        metadata["flow_rate_m3h"] = flow_rate_m3h
        metadata["negative_correction_applied"] = negative_corrected
        metadata["fallback_digit_count"] = fallback_digit_count
        metadata["digits_changed_vs_last"] = digits_changed_vs_last
        metadata["digits_changed_vs_top_pred"] = digits_changed_vs_top_pred
        metadata["prediction_rank_used_counts"] = prediction_rank_used_counts

        if flow_rate_m3min > max_flow_rate :
            metadata["rejection_reason"] = "flow_rate_high"
        elif flow_rate_m3min < 0 and not allow_negative_correction:
            metadata["rejection_reason"] = "negative_flow"
        elif reject:
            metadata["rejection_reason"] = "fallback_digit"

        if metadata["rejection_reason"]:
            print(f"[CorrectionAlg ({name})] Flow rate is too high or negative")
            return {
                "accepted": False,
                "value": None,
                "total_confidence": 0.0,
                "used_confidence": 0.0,
                **metadata
            }

        print (f"[CorrectionAlg ({name})] Value accepted for time", new_time, "flow rate", flow_rate_m3min, "value", correctedValue)
        return {
            "accepted": True,
            "value": int(correctedValue),
            "total_confidence": totalConfidence,
            "used_confidence": usedConfidence,
            **metadata
        }
