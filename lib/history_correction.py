import json
import sqlite3
from datetime import datetime, timezone
from itertools import product

from dateutil.tz import tzlocal


def generate_number_variations(data):
    # Keep last two digits but ignore their alternatives
    core_data = data[:-2]
    last_two_digits = [max(data[-2], key=lambda x: x[1])[0], max(data[-1], key=lambda x: x[1])[0]]

    # Filter out values with confidence < 3%
    filtered_data = [
        [(char, conf) for char, conf in predictions if conf >= 0.075]
        for predictions in core_data
    ]

    # Generate all possible combinations
    possible_combinations = list(product(*filtered_data))

    # Compute total confidence for each combination
    results = []
    for combination in possible_combinations:
        number = "".join([char for char, _ in combination]) + "".join(last_two_digits)
        total_confidence = round(
            sum([conf for _, conf in combination]) / len(combination), 6
        )  # Averaging confidence
        results.append([number, total_confidence])

    return results


def replace_rotation(value, r_last_value, segments=8):
    r_indices = [pos for pos, char in enumerate(value) if char == 'r']
    # replace the r with the digit from the last value
    nstr = value
    for r_index in r_indices:
        nstr = (nstr[:r_index] + str(r_last_value).zfill(segments)[r_index]
                + nstr[r_index + 1:])

    return nstr


def correct_value(db_file:str, name: str, new_eval, max_flow = 0.7):

    # get last evaluation
    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        segments = len(new_eval[2])
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

        max_flow /= 60.0
        # get the time difference in minutes
        time_diff = (new_time - last_time).seconds / 60.0
        if time_diff <= 0:
            conn.commit()
            print("Time difference is 0 or negative")
            return None

        correctedValue = ""
        totalConfidence = 1.0
        for i, lastChar in enumerate(last_value):
            predictions = new_results[i]

            for prediction in predictions:
                tempValue = correctedValue
                tempConfidence = totalConfidence
                if prediction[0] == 'r':
                    tempValue += lastChar
                    tempConfidence *= prediction[1]
                else:
                    tempValue += prediction[0]
                    tempConfidence *= prediction[1]

                if int(tempValue) >= int(last_value[:i+1]):
                    correctedValue = tempValue
                    totalConfidence = tempConfidence
                    break

        # get the flow rate
        flow_rate = (int(correctedValue) - int(last_value)) / 1000.0 / time_diff
        if flow_rate > max_flow or flow_rate < 0:
            print("Flow rate is too high or negative")
            return None
        print ("Value accepted for time", new_time, "flow rate", flow_rate)
        return int(correctedValue)

def correct_value_legacy(db_file:str, name: str, new_eval, max_flow = 0.7):

    # get last evaluation
    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        segments = len(new_eval[2])
        # get last history entry
        cursor.execute("SELECT value, timestamp FROM history WHERE name = ? ORDER BY ROWID DESC LIMIT 1", (name,))
        row = cursor.fetchone()
        if not row:
            conn.commit()
            return None
        last_value = int(row[0]) / 1000.0
        last_time = datetime.fromisoformat(row[1])
        new_time = datetime.fromisoformat(new_eval[3])

        variations = generate_number_variations(new_eval[2])
        print (variations)
        for i, variation in enumerate(variations[:]):
            print (variation , last_value, segments)
            variations[i][0] = replace_rotation(variation[0], int(last_value * 1000), segments)
            print (variations[i][0], last_value)

        max_flow /= 60.0
        # get the time difference in minutes
        time_diff = (new_time - last_time).seconds / 60.0

        if time_diff == 0:
            conn.commit()
            return None

        variations = sorted(variations, key=lambda x: x[1], reverse=True)

        value = int(variations[0][0]) / 1000.0
        flow_rate = (value - last_value) / time_diff
        if flow_rate > max_flow or flow_rate < 0:
            # search a mutation that fits the flow rate
            for i in range(1, len(variations)):
                value = int(variations[i][0]) / 1000.0
                print(value, last_value)
                flow_rate = (value - last_value) / time_diff
                if flow_rate <= max_flow and flow_rate >= 0:
                    print("Alternative value found for time", new_time, "flow rate", flow_rate)
                    return int(value * 1000)
            print("No mutation found for time", new_time, "flow rate", flow_rate)
            return None
        else:
            print ("Value accepted for time", new_time, "flow rate", flow_rate)
            return int(value * 1000)