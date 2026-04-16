from lib.log import log
import datetime
import time
import threading
import sqlite3
import json
from typing import Dict, Any

from lib.capture_utils import capture_and_process_source
from lib.model_singleton import get_meter_predictor
from lib.global_alerts import add_alert, remove_alert
import traceback

class PollingHandler:

    def __init__(self, config, db_file: str = 'watermeters.db', mqtt_client=None):
        self.db_file = db_file
        self.config = config
        self.meter_predictor = get_meter_predictor()
        self.mqtt_client = mqtt_client
        self.stop_event = threading.Event()
        log("[POLLING] Using shared meter predictor singleton instance.")

    def _process_capture(self, source_row):
        source_id = source_row['id']
        source_name = source_row['name']
        now = datetime.datetime.now().isoformat()
        alert_key = f'polling_{source_name}'

        try:
            capture_and_process_source(self.config, self.db_file, source_row, self.meter_predictor, mqtt_client=self.mqtt_client)
            # On success, update last_success_ts and clear error
            with sqlite3.connect(self.db_file, timeout=30) as conn:
                cursor = conn.cursor()
                cursor.execute("UPDATE sources SET last_success_ts = ?, last_error = NULL WHERE id = ?", (now, source_id))
                conn.commit()
            log(f"[POLLING] Successfully captured from source '{source_name}'")
            remove_alert(alert_key)
        except Exception as e:
            # On failure, update last_success_ts to now to prevent immediate retry
            error_msg = str(e)
            log(f"[POLLING] Error capturing from source '{source_name}': {error_msg}")
            traceback.print_exc()
            with sqlite3.connect(self.db_file, timeout=30) as conn:
                cursor = conn.cursor()
                cursor.execute("UPDATE sources SET last_success_ts = ?, last_error = ? WHERE id = ?", (now, error_msg, source_id))
                conn.commit()
            add_alert(alert_key, f"Polling failed for source '{source_name}': {error_msg}")

    def _polling_loop(self):
        while not self.stop_event.is_set():
            try:
                with sqlite3.connect(self.db_file, timeout=30) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT id, name, source_type, poll_interval_s, config_json, last_success_ts
                        FROM sources
                        WHERE enabled = 1 AND poll_interval_s > 0 AND source_type IN ('ha_camera', 'http')
                    """)
                    sources = cursor.fetchall()

                for source in sources:
                    # Check if it's time to poll
                    last_ts = source['last_success_ts']
                    interval = source['poll_interval_s']
                    now = datetime.datetime.now()
                    if last_ts:
                        last_dt = datetime.datetime.fromisoformat(last_ts)
                        if (now - last_dt).total_seconds() < interval:
                            continue
                    # Capture
                    self._process_capture(source)

            except Exception as e:
                log(f"[POLLING] Error in polling loop: {e}")
                traceback.print_exc()

            # Sleep for a short time before next check
            self.stop_event.wait(10)  # Check every 10 seconds

    def start(self):
        self.thread = threading.Thread(target=self._polling_loop, daemon=True)
        self.thread.start()
        log("[POLLING] Polling handler started")

    def stop(self):
        self.stop_event.set()
        if hasattr(self, 'thread'):
            self.thread.join()
        log("[POLLING] Polling handler stopped")
