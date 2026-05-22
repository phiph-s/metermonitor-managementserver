from threading import Lock

# This is a global variable that is used to store alerts
# to be shown in the frontend.

# This provides a thread safe way to add, remove and get alerts.
# For communication between frontend and mqtt_handler.py

alerts = {}
alerts_lock = Lock()
_change_callback = None


def set_change_callback(callback):
    global _change_callback
    _change_callback = callback


def _notify_change():
    if _change_callback:
        try:
            _change_callback(get_alerts())
        except Exception:
            pass


def add_alert(key, alert):
    with alerts_lock:
        alerts[key] = alert
    _notify_change()


def remove_alert(key):
    changed = False
    with alerts_lock:
        if key in alerts:
            del alerts[key]
            changed = True
    if changed:
        _notify_change()


def get_alerts():
    with alerts_lock:
        return dict(alerts)


def clear_alerts():
    with alerts_lock:
        alerts.clear()
    _notify_change()
