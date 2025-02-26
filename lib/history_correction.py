import json
import sqlite3
from datetime import datetime, timezone
from itertools import product

from dateutil.tz import tzlocal

def correct_value(db_file:str, name: str, new_eval, max_flow = 0.7, allow_negative_correction = False):
    # get last evaluation
    reject = False
    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        segments = len(new_eval[2])
        print("segments", segments)
        # get last history entry
        cursor.execute("SELECT value, timestamp FROM history WHERE name = ? ORDER BY ROWID DESC LIMIT 1", (name,))
        row = cursor.fetchone()
        if not row:
            conn.commit()
            return None
        last_value = str(row[0]).zfill(segments)
        last_time = datetime.fromisoformat(row[1])
        new_time = datetime.fromisoformat(new_eval[3])
        new_results = new_eval[2]
        new_tesseract_results = new_eval[4]

        max_flow /= 60.0
        # get the time difference in minutes
        time_diff = (new_time - last_time).seconds / 60.0
        if time_diff <= 0:
            conn.commit()
            print("Time difference is 0 or negative")
            return None

        correctedValue = ""
        totalConfidence = 1.0
        negative_corrected = False
        for i, lastChar in enumerate(last_value):

            if new_tesseract_results[i]:
                tempValue = correctedValue +  new_tesseract_results[i][0]
                tempConfidence = totalConfidence * new_tesseract_results[i][1] / 100.0
                if int(tempValue) >= int(last_value[:i + 1]):
                    correctedValue = tempValue
                    totalConfidence = tempConfidence
                    continue
            predictions = new_results[i]

            digit_appended = False
            for prediction in predictions:
                tempValue = correctedValue
                tempConfidence = totalConfidence
                if prediction[0] == 'r':
                    # check if the digit before has changed upwards, set the digit to 0
                    if len(correctedValue) > 1 and correctedValue[-1] != last_value[i-1]:
                        tempValue += '0'
                        tempConfidence *= prediction[1]
                    else:
                        tempValue += lastChar
                        tempConfidence *= prediction[1]
                else:
                    tempValue += prediction[0]
                    tempConfidence *= prediction[1]

                if int(tempValue) >= int(last_value[:i+1]) or negative_corrected:
                    correctedValue = tempValue
                    totalConfidence = tempConfidence
                    digit_appended = True
                    break
                elif allow_negative_correction and new_tesseract_results[i]:
                    if new_tesseract_results[i][0] == prediction[0] and \
                            new_tesseract_results[i][1] >= 95:
                        correctedValue = correctedValue + new_tesseract_results[i][0]
                        totalConfidence = totalConfidence * new_tesseract_results[i][1] / 100.0
                        digit_appended = True
                        negative_corrected = True
                        print("Negative correction accepted")
                        break

            if not digit_appended:
                correctedValue += lastChar
                reject = True
                print("Fallback: appending original digit", lastChar)

        # get the flow rate
        flow_rate = (int(correctedValue) - int(last_value)) / 1000.0 / time_diff
        if flow_rate > max_flow or (flow_rate < 0 and not allow_negative_correction) or reject:
            print("Flow rate is too high or negative")
            return None
        print ("Value accepted for time", new_time, "flow rate", flow_rate, "value", correctedValue)
        return int(correctedValue), totalConfidence