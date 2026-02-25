import sqlite3
import json
from datetime import datetime

def run_migrations(db_file):
    with sqlite3.connect(db_file) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS watermeters
                       (
                           name              TEXT PRIMARY KEY,
                           picture_number    INTEGER,
                           wifi_rssi         INTEGER,
                           picture_format    TEXT,
                           picture_timestamp TEXT,
                           picture_width     INTEGER,
                           picture_height    INTEGER,
                           picture_length    INTEGER,
                           picture_data      TEXT,
                           setup             BOOLEAN DEFAULT 0,
                           picture_data_bbox BLOB
                       )
                       ''')
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS settings
                       (
                           name                TEXT PRIMARY KEY,
                           threshold_low       INTEGER,
                           threshold_high      INTEGER,
                           threshold_last_low  INTEGER,
                           threshold_last_high INTEGER,
                           islanding_padding   INTEGER,
                           segments            INTEGER,
                           rotated_180         BOOLEAN,
                           shrink_last_3       BOOLEAN,
                           extended_last_digit BOOLEAN,
                           max_flow_rate       FLOAT,
                           conf_threshold      REAL DEFAULT NULL,
                           roi_extractor       TEXT DEFAULT 'yolo',
                           template_id         TEXT DEFAULT NULL,
                           segment_mode        TEXT DEFAULT 'display',
                           use_correctional_alg BOOLEAN DEFAULT true,
                           FOREIGN KEY (name) REFERENCES watermeters (name)
                       )
                       ''')
        # Add evaluations table (latest schema)
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS evaluations
                       (
                           id INTEGER PRIMARY KEY AUTOINCREMENT,
                           name TEXT NOT NULL,
                           colored_digits TEXT,
                           th_digits TEXT,
                           predictions TEXT,
                           timestamp DATETIME,
                           result INTEGER,
                           total_confidence REAL,
                           outdated BOOLEAN DEFAULT 0,
                           denied_digits TEXT,
                           th_digits_inverted TEXT,
                           used_confidence REAL DEFAULT -1.0,
                           flow_rate_m3h REAL,
                           delta_m3 REAL,
                           delta_raw INTEGER,
                           time_diff_min REAL,
                           rejection_reason TEXT,
                           negative_correction_applied BOOLEAN,
                           fallback_digit_count INTEGER,
                           digits_changed_vs_last INTEGER,
                           digits_changed_vs_top_pred INTEGER,
                           prediction_rank_used_counts TEXT,
                           denied_digits_count INTEGER,
                           timestamp_adjusted BOOLEAN,
                           FOREIGN KEY (name) REFERENCES watermeters (name)
                       )
                       ''')
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS history
                       (
                           id INTEGER PRIMARY KEY AUTOINCREMENT,
                           name TEXT NOT NULL,
                           value INTEGER,
                           confidence REAL,
                           used_confidence REAL DEFAULT -1.0,
                           target_brightness REAL,
                           timestamp TEXT,
                           manual BOOLEAN,
                           FOREIGN KEY (name) REFERENCES watermeters (name)
                       )
                       ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                source_type TEXT NOT NULL,
                enabled BOOLEAN DEFAULT 1,
                poll_interval_s INTEGER,
                config_json TEXT,
                last_success_ts TEXT,
                last_error TEXT,
                created_ts TEXT DEFAULT (datetime('now')),
                updated_ts TEXT DEFAULT (datetime('now'))
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS templates (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                reference_image_base64 TEXT NOT NULL,
                image_width INTEGER NOT NULL,
                image_height INTEGER NOT NULL,
                config_json TEXT NOT NULL,
                precomputed_data_base64 TEXT
            )
        ''')
        # For <= 1.2.3: Add outdated bool to evaluations table if it doesn't exist yet
        cursor.execute("PRAGMA table_info(evaluations)")
        columns = [info[1] for info in cursor.fetchall()]
        if 'outdated' not in columns:
            cursor.execute('''
                ALTER TABLE evaluations
                ADD COLUMN outdated BOOLEAN DEFAULT 0
            ''')
            print("[MIGRATION] Added 'outdated' column to 'evaluations' table")
            # re-read columns after altering
            cursor.execute("PRAGMA table_info(evaluations)")
            columns = [info[1] for info in cursor.fetchall()]

        # Migrate from old schema (name, eval JSON string) to new explicit columns
        # Only run if the legacy 'eval' column exists and new columns are not present yet
        new_cols_needed = {'colored_digits', 'th_digits', 'predictions', 'timestamp', 'result', 'total_confidence'}
        if 'eval' in columns and not new_cols_needed.intersection(set(columns)):
            print("[MIGRATION] Starting evaluations table migration (eval JSON -> explicit columns)")

            # Create new table with desired schema (assumes original table had only 'name' and 'eval')
            cursor.execute('''
                CREATE TABLE evaluations_new (
                    name TEXT,
                    colored_digits TEXT,
                    th_digits TEXT,
                    predictions TEXT,
                    timestamp DATETIME,
                    result INTEGER,
                    total_confidence REAL,
                    outdated BOOLEAN DEFAULT 0
                )
            ''')

            # Copy and transform rows
            cursor.execute("SELECT name, eval, outdated FROM evaluations")
            rows = cursor.fetchall()
            for row in rows:
                name = row["name"]
                eval_str = row["eval"]
                outdated_val = row["outdated"] if "outdated" in row.keys() else 0

                # Default values
                colored_json = None
                th_json = None
                predictions_json = None
                timestamp_val = None
                result_val = None
                total_conf_val = None

                # Parse eval JSON safely
                try:
                    parsed = json.loads(eval_str) if eval_str else []
                except Exception:
                    parsed = []

                # Map indices to new columns
                if isinstance(parsed, list):
                    if len(parsed) > 0 and parsed[0] is not None:
                        # store as JSON string (array of base64 strings)
                        try:
                            colored_json = json.dumps(parsed[0])
                        except Exception:
                            colored_json = None
                    if len(parsed) > 1 and parsed[1] is not None:
                        try:
                            th_json = json.dumps(parsed[1])
                        except Exception:
                            th_json = None
                    if len(parsed) > 2 and parsed[2] is not None:
                        try:
                            predictions_json = json.dumps(parsed[2])
                        except Exception:
                            predictions_json = None
                    if len(parsed) > 3 and parsed[3]:
                        # Try to parse ISO timestamp; store as ISO string if valid
                        try:
                            dt = datetime.fromisoformat(parsed[3])
                            timestamp_val = dt.isoformat(sep='T')
                        except Exception:
                            # If parsing fails, attempt to store raw string; if empty, keep None
                            timestamp_val = parsed[3] if isinstance(parsed[3], str) and parsed[3].strip() else None
                    if len(parsed) > 4:
                        # result may be null
                        result_val = parsed[4] if parsed[4] is not None else None
                    if len(parsed) > 6:
                        try:
                            total_conf_val = float(parsed[6]) if parsed[6] is not None else None
                        except Exception:
                            total_conf_val = None

                cursor.execute('''
                    INSERT INTO evaluations_new
                    (name, colored_digits, th_digits, predictions, timestamp, result, total_confidence, outdated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (name, colored_json, th_json, predictions_json, timestamp_val, result_val, total_conf_val, outdated_val))

            # Replace old table
            cursor.execute("DROP TABLE evaluations")
            cursor.execute("ALTER TABLE evaluations_new RENAME TO evaluations")
            conn.commit()
            print("[MIGRATION] Completed evaluations table migration")

        # Add auto-incrementing ID primary key to evaluations table
        cursor.execute("PRAGMA table_info(evaluations)")
        columns = [info[1] for info in cursor.fetchall()]
        if 'id' not in columns:
            print("[MIGRATION] Adding auto-incrementing ID primary key to evaluations table")

            # Create new table with id as primary key
            cursor.execute('''
                CREATE TABLE evaluations_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    colored_digits TEXT,
                    th_digits TEXT,
                    predictions TEXT,
                    timestamp DATETIME,
                    result INTEGER,
                    total_confidence REAL,
                    outdated BOOLEAN DEFAULT 0,
                    FOREIGN KEY(name) REFERENCES watermeters(name)
                )
            ''')

            # Copy all data from old table
            cursor.execute('''
                INSERT INTO evaluations_new 
                (name, colored_digits, th_digits, predictions, timestamp, result, total_confidence, outdated)
                SELECT name, colored_digits, th_digits, predictions, timestamp, result, total_confidence, outdated
                FROM evaluations
            ''')

            # Replace old table
            cursor.execute("DROP TABLE evaluations")
            cursor.execute("ALTER TABLE evaluations_new RENAME TO evaluations")
            conn.commit()
            print("[MIGRATION] Added ID primary key to evaluations table")
        # Add auto-incrementing ID primary key to history table
        cursor.execute("PRAGMA table_info(history)")
        columns = [info[1] for info in cursor.fetchall()]
        if 'id' not in columns:
            print("[MIGRATION] Adding auto-incrementing ID primary key to history table")

            # Create new table with id as primary key
            cursor.execute('''
                CREATE TABLE history_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    value INTEGER,
                    confidence REAL,
                    target_brightness REAL,
                    timestamp TEXT,
                    manual BOOLEAN,
                    FOREIGN KEY(name) REFERENCES watermeters(name)
                )
            ''')

            # Copy all data from old table
            cursor.execute('''
                INSERT INTO history_new 
                (name, value, confidence, target_brightness, timestamp, manual)
                SELECT name, value, confidence, target_brightness, timestamp, manual
                FROM history
            ''')

            # Replace old table
            cursor.execute("DROP TABLE history")
            cursor.execute("ALTER TABLE history_new RENAME TO history")
            conn.commit()
            print("[MIGRATION] Added ID primary key to history table")

        # add column picture_data_bbox to watermeters table if it doesn't exist yet
        cursor.execute("PRAGMA table_info(watermeters)")
        columns = [info[1] for info in cursor.fetchall()]
        if 'picture_data_bbox' not in columns:
            cursor.execute('''
                ALTER TABLE watermeters
                ADD COLUMN picture_data_bbox BLOB
            ''')
            print("[MIGRATION] Added 'picture_data_bbox' column to 'watermeters' table")

        # add settings column conf_threshold to settings table if it doesn't exist yet
        cursor.execute("PRAGMA table_info(settings)")
        columns = [info[1] for info in cursor.fetchall()]
        if 'conf_threshold' not in columns:
            cursor.execute('''
                ALTER TABLE settings
                ADD COLUMN conf_threshold REAL DEFAULT NULL
            ''')
            print("[MIGRATION] Added 'conf_threshold' column to 'settings' table")

        cursor.execute("PRAGMA table_info(settings)")
        columns = [info[1] for info in cursor.fetchall()]
        if 'roi_extractor' not in columns:
            cursor.execute('''
                ALTER TABLE settings
                ADD COLUMN roi_extractor TEXT DEFAULT 'yolo'
            ''')
            print("[MIGRATION] Added 'roi_extractor' column to 'settings' table")

        cursor.execute("PRAGMA table_info(settings)")
        columns = [info[1] for info in cursor.fetchall()]
        if 'template_id' not in columns:
            cursor.execute('''
                ALTER TABLE settings
                ADD COLUMN template_id TEXT DEFAULT NULL
            ''')
            print("[MIGRATION] Added 'template_id' column to 'settings' table")

        cursor.execute("PRAGMA table_info(settings)")
        columns = [info[1] for info in cursor.fetchall()]
        if 'segment_mode' not in columns:
            cursor.execute('''
                ALTER TABLE settings
                ADD COLUMN segment_mode TEXT DEFAULT 'display'
            ''')
            print("[MIGRATION] Added 'segment_mode' column to 'settings' table")

        # add a column "denied_digits" to the evaluations table ([FALSE, FALSE, ...] as JSON string with length of digits as default)
        cursor.execute("PRAGMA table_info(evaluations)")
        columns = [info[1] for info in cursor.fetchall()]
        if 'denied_digits' not in columns:
            cursor.execute('''
                ALTER TABLE evaluations
                ADD COLUMN denied_digits TEXT
            ''')
            # set default value for existing rows
            cursor.execute("SELECT name, th_digits FROM evaluations")
            rows = cursor.fetchall()
            for row in rows:
                name = row["name"]
                th_digits_json = row["th_digits"]
                length = 0
                try:
                    th_digits = json.loads(th_digits_json) if th_digits_json else []
                    length = len(th_digits)
                except Exception:
                    length = 0
                denied_list = [False] * length
                denied_json = json.dumps(denied_list)
                cursor.execute("UPDATE evaluations SET denied_digits = ? WHERE name = ?", (denied_json, name))
            print("[MIGRATION] Added 'denied_digits' column to 'evaluations' table and set default values")


        # add column th_digits_inverted to evaluations table if it doesn't exist yet
        # invert the th_digits and store in th_digits_inverted
        cursor.execute("PRAGMA table_info(evaluations)")
        columns = [info[1] for info in cursor.fetchall()]
        if 'th_digits_inverted' not in columns:
            cursor.execute('''
                ALTER TABLE evaluations
                ADD COLUMN th_digits_inverted TEXT
            ''')
            # load all images, invert colors of th_digits and store in th_digits_inverted
            cursor.execute("SELECT id, th_digits FROM evaluations")
            rows = cursor.fetchall()
            for row in rows:
                row_id = row["id"]
                th_digits_json = row["th_digits"]
                inverted_list = []
                try:
                    th_digits = json.loads(th_digits_json) if th_digits_json else []
                    for digit_b64 in th_digits:
                        # decode base64
                        import base64
                        from io import BytesIO
                        from PIL import Image, ImageOps

                        digit_data = base64.b64decode(digit_b64)
                        digit_image = Image.open(BytesIO(digit_data)).convert("L")  # convert to grayscale
                        inverted_image = ImageOps.invert(digit_image)
                        buffered = BytesIO()
                        inverted_image.save(buffered, format="PNG")
                        inverted_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                        inverted_list.append(inverted_b64)
                except Exception:
                    inverted_list = []
                inverted_json = json.dumps(inverted_list)
                cursor.execute("UPDATE evaluations SET th_digits_inverted = ? WHERE id = ?", (inverted_json, row_id))
            print("[MIGRATION] Added 'th_digits_inverted' column to 'evaluations' table and populated values")

        # add column "used_confidence" to history
        cursor.execute("PRAGMA table_info(history)")
        columns = [info[1] for info in cursor.fetchall()]
        if 'used_confidence' not in columns:
            cursor.execute("ALTER TABLE history ADD COLUMN used_confidence REAL DEFAULT -1.0")
            print("[MIGRATION] Added 'used_confidence' column to 'history' table")

        # add column "used_confidence" to evaluations
        cursor.execute("PRAGMA table_info(evaluations)")
        columns = [info[1] for info in cursor.fetchall()]
        if 'used_confidence' not in columns:
            cursor.execute("ALTER TABLE evaluations ADD COLUMN used_confidence REAL DEFAULT -1.0")
            print("[MIGRATION] Added 'used_confidence' column to 'evaluations' table")

        # add correction metadata columns to evaluations
        cursor.execute("PRAGMA table_info(evaluations)")
        columns = [info[1] for info in cursor.fetchall()]
        if 'flow_rate_m3h' not in columns:
            cursor.execute("ALTER TABLE evaluations ADD COLUMN flow_rate_m3h REAL")
            print("[MIGRATION] Added 'flow_rate_m3h' column to 'evaluations' table")
        if 'delta_m3' not in columns:
            cursor.execute("ALTER TABLE evaluations ADD COLUMN delta_m3 REAL")
            print("[MIGRATION] Added 'delta_m3' column to 'evaluations' table")
        if 'delta_raw' not in columns:
            cursor.execute("ALTER TABLE evaluations ADD COLUMN delta_raw INTEGER")
            print("[MIGRATION] Added 'delta_raw' column to 'evaluations' table")
        if 'time_diff_min' not in columns:
            cursor.execute("ALTER TABLE evaluations ADD COLUMN time_diff_min REAL")
            print("[MIGRATION] Added 'time_diff_min' column to 'evaluations' table")
        if 'rejection_reason' not in columns:
            cursor.execute("ALTER TABLE evaluations ADD COLUMN rejection_reason TEXT")
            print("[MIGRATION] Added 'rejection_reason' column to 'evaluations' table")
        if 'negative_correction_applied' not in columns:
            cursor.execute("ALTER TABLE evaluations ADD COLUMN negative_correction_applied BOOLEAN")
            print("[MIGRATION] Added 'negative_correction_applied' column to 'evaluations' table")
        if 'fallback_digit_count' not in columns:
            cursor.execute("ALTER TABLE evaluations ADD COLUMN fallback_digit_count INTEGER")
            print("[MIGRATION] Added 'fallback_digit_count' column to 'evaluations' table")
        if 'digits_changed_vs_last' not in columns:
            cursor.execute("ALTER TABLE evaluations ADD COLUMN digits_changed_vs_last INTEGER")
            print("[MIGRATION] Added 'digits_changed_vs_last' column to 'evaluations' table")
        if 'digits_changed_vs_top_pred' not in columns:
            cursor.execute("ALTER TABLE evaluations ADD COLUMN digits_changed_vs_top_pred INTEGER")
            print("[MIGRATION] Added 'digits_changed_vs_top_pred' column to 'evaluations' table")
        if 'prediction_rank_used_counts' not in columns:
            cursor.execute("ALTER TABLE evaluations ADD COLUMN prediction_rank_used_counts TEXT")
            print("[MIGRATION] Added 'prediction_rank_used_counts' column to 'evaluations' table")
        if 'denied_digits_count' not in columns:
            cursor.execute("ALTER TABLE evaluations ADD COLUMN denied_digits_count INTEGER")
            print("[MIGRATION] Added 'denied_digits_count' column to 'evaluations' table")
        if 'timestamp_adjusted' not in columns:
            cursor.execute("ALTER TABLE evaluations ADD COLUMN timestamp_adjusted BOOLEAN")
            print("[MIGRATION] Added 'timestamp_adjusted' column to 'evaluations' table")

        # add column "used_confidence" to evaluations
        cursor.execute("PRAGMA table_info(settings)")
        columns = [info[1] for info in cursor.fetchall()]
        if 'use_correctional_alg' not in columns:
            cursor.execute("ALTER TABLE settings ADD COLUMN use_correctional_alg BOOLEAN DEFAULT true")
            print("[MIGRATION] Added 'use_correctional_alg' column to 'settings' table")
