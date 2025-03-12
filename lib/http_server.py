from io import BytesIO

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException, Body, Header, Depends
from fastapi.openapi.models import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import base64
import sqlite3

from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request

from lib.functions import reevaluate_latest_picture, add_history_entry
from lib.meter_processing.meter_processing import MeterPredictor


# http server class
# that serves the json api endpoints:

# GET /discovery - Returns watermeters that are not setup
# GET /evaluate?base64=... - Returns the evaluation of the base64 encoded image
#                            (Uses MeterPredictor class to evaluate the image)

# GET /watermeters - Returns all watermeters
# GET /watermeters/:name - Returns the watermeter with the given name, including the evaluation results

# POST /setup - Sets up the watermeter with the given name
# POST /thresholds - Sets the thresholds for the watermeter with the given name (completes the setup)

def prepare_setup_app(config, lifespan):
    app = FastAPI(lifespan=lifespan)
    SECRET_KEY = config['secret_key']
    db_connection = lambda: sqlite3.connect(config['dbfile'])

    meter_preditor = MeterPredictor(
        yolo_model_path="models/yolo-best-obb-2.pt",
        digit_classifier_model_path="models/digit_classifier.pth"
    )
    print("HTTP-Server: Loaded HTTP meter predictor.")

    # CORS Konfiguration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Authentication
    def authenticate(secret: str = Header(None)):
        print (secret, SECRET_KEY)
        if secret != SECRET_KEY:
            raise HTTPException(status_code=401, detail="Unauthorized")

    # Models
    class PictureData(BaseModel):
        format: str
        timestamp: str
        width: int
        height: int
        length: int
        data: str

    class ConfigRequest(BaseModel):
        name: str
        picture_number: int
        WiFi_RSSI: int
        picture: PictureData

    class SettingsRequest(BaseModel):
        name: str
        threshold_low: int
        threshold_high: int
        threshold_last_low: int
        threshold_last_high: int
        islanding_padding: int
        segments: int
        rotated_180: bool
        shrink_last_3: bool
        extended_last_digit: bool
        invert: bool

    class EvalRequest(BaseModel):
        eval: str

    class SetupData(BaseModel):
        value: int
        timestamp: str

    @app.get("/api/discovery", dependencies=[Depends(authenticate)])
    def get_discovery():
        cursor = db_connection().cursor()
        cursor.execute("SELECT name, picture_timestamp FROM watermeters WHERE setup = 0")
        return {"watermeters": [row for row in cursor.fetchall()]}

    @app.post("/api/evaluate/single", dependencies=[Depends(authenticate)])
    def evaluate(
            base64str: str = Body(...),  # Changed from Query to Body for POST requests
            threshold_low: float = Body(0, ge=0, le=255),
            threshold_high: float = Body(155, ge=0, le=255),
            islanding_padding: int = Body(20, ge=0),
            invert: bool = Body(False)
    ):
            # Decode the base64 image
            image_data = base64.b64decode(base64str)

            # Get PIL image from base64
            image = Image.open(BytesIO(image_data))
            # to numpy array
            image = np.array(image)

            # Apply threshold with the passed values
            base64r, digits = meter_preditor.apply_threshold(image, threshold_low, threshold_high, islanding_padding,  invert = invert)

            # Return the result
            return {"base64": base64r}

    @app.get("/api/watermeters", dependencies=[Depends(authenticate)])
    def get_watermeters():
        cursor = db_connection().cursor()
        cursor.execute("SELECT name, picture_timestamp FROM watermeters WHERE setup = 1")
        return {"watermeters": [row for row in cursor.fetchall()]}

    @app.post("/api/setup/{name}/finish", dependencies=[Depends(authenticate)])
    def post_setup_finished(name: str, data: SetupData):
        db = db_connection()
        cursor = db.cursor()
        cursor.execute("UPDATE watermeters SET setup = 1 WHERE name = ?", (name,))
        db.commit()
        target_brightness, confidence = reevaluate_latest_picture(config['dbfile'], name, meter_preditor, config)
        add_history_entry(config['dbfile'], name, data.value, confidence, target_brightness, data.timestamp, config, manual=True)
        return {"message": "Setup completed"}

    @app.post("/api/setup/{name}/reset", dependencies=[Depends(authenticate)])
    def post_setup_reset(name: str):
        db = db_connection()
        cursor = db.cursor()
        cursor.execute("UPDATE watermeters SET setup = 0 WHERE name = ?", (name,))

        # clear history
        cursor.execute("DELETE FROM history WHERE name = ?", (name,))

        db.commit()
        return {"message": "Setup completed"}

    @app.get("/api/watermeters/{name}/history", dependencies=[Depends(authenticate)])
    def get_watermeter(name: str):
        cursor = db_connection().cursor()
        cursor.execute("SELECT value, timestamp, confidence, manual FROM history WHERE name = ?", (name,))
        return {"history": [row for row in cursor.fetchall()]}

    @app.get("/api/watermeters/{name}", dependencies=[Depends(authenticate)])
    def get_watermeter(name: str):
        cursor = db_connection().cursor()
        cursor.execute("SELECT * FROM watermeters WHERE name = ?", (name,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Watermeter not found")
        return {
            "name": row[0],
            "picture_number": row[1],
            "WiFi-RSSI": row[2],
            "picture": {
                "format": row[3],
                "timestamp": row[4],
                "width": row[5],
                "height": row[6],
                "length": row[7],
                "data": row[8]
            }
        }

    @app.delete("/api/watermeters/{name}", dependencies=[Depends(authenticate)])
    def delete_watermeter(name: str):
        db = db_connection()
        cursor = db.cursor()
        cursor.execute("DELETE FROM watermeters WHERE name = ?", (name,))
        cursor.execute("DELETE FROM evaluations WHERE name = ?", (name,))
        cursor.execute("DELETE FROM history WHERE name = ?", (name,))
        cursor.execute("DELETE FROM settings WHERE name = ?", (name,))
        db.commit()
        return {"message": "Watermeter deleted", "name": name}

    @app.post("/api/setup", dependencies=[Depends(authenticate)])
    def setup_watermeter(config: ConfigRequest):
        db = db_connection()
        cursor = db.cursor()
        cursor.execute(
            """
            INSERT INTO watermeters (name, picture_number, wifi_rssi, picture_format, 
            picture_timestamp, picture_width, picture_height, picture_length, picture_data) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (
                config.name,
                config.picture_number,
                config.WiFi_RSSI,
                config.picture.format,
                config.picture.timestamp,
                config.picture.width,
                config.picture.height,
                config.picture.length,
                config.picture.data
            )
        )
        db_connection.commit()
        return {"message": "Watermeter configured", "name": config.name}

    @app.get("/api/settings/{name}", dependencies=[Depends(authenticate)])
    def get_settings(name: str):
        cursor = db_connection().cursor()
        cursor.execute("SELECT threshold_low, threshold_high, threshold_last_low, threshold_last_high, islanding_padding, segments, shrink_last_3, extended_last_digit, invert, rotated_180 FROM settings WHERE name = ?", (name,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Thresholds not found")
        return {
            "threshold_low": row[0],
            "threshold_high": row[1],
            "threshold_last_low": row[2],
            "threshold_last_high": row[3],
            "islanding_padding": row[4],
            "segments": row[5],
            "shrink_last_3": row[6],
            "extended_last_digit": row[7],
            "invert": row[8],
            "rotated_180": row[9]
        }

    @app.post("/api/settings", dependencies=[Depends(authenticate)])
    def set_settings(settings: SettingsRequest):
        db = db_connection()
        cursor = db.cursor()
        cursor.execute(
            """
            INSERT INTO settings (name, threshold_low, threshold_high, threshold_last_low, threshold_last_high, islanding_padding, segments, shrink_last_3, extended_last_digit, invert, rotated_180) 
            VALUES (?, ?, ?, ?,?,?, ?, ? , ?, ?, ?) ON CONFLICT(name) DO UPDATE SET 
            threshold_low=excluded.threshold_low, threshold_high=excluded.threshold_high, threshold_last_low=excluded.threshold_last_low, threshold_last_high=excluded.threshold_last_high,
            islanding_padding=excluded.islanding_padding,
            segments=excluded.segments, shrink_last_3=excluded.shrink_last_3, extended_last_digit=excluded.extended_last_digit, invert=excluded.invert, rotated_180=excluded.rotated_180
            """,
            (settings.name, settings.threshold_low, settings.threshold_high, settings.threshold_last_low, settings.threshold_last_high, settings.islanding_padding,
             settings.segments, settings.shrink_last_3, settings.extended_last_digit, settings.invert, settings.rotated_180)
        )
        db.commit()
        return {"message": "Thresholds set", "name": settings.name}

    @app.get("/api/reevaluate_latest/{name}", dependencies=[Depends(authenticate)])
    def reevaluate_latest(name: str):
        reevaluate_latest_picture(config['dbfile'], name, meter_preditor, config)

    # GET endpoint for retrieving evaluations
    @app.get("/api/watermeters/{name}/evals", dependencies=[Depends(authenticate)])
    def get_evals(name: str):
        cursor = db_connection().cursor()
        # Check if watermeter exists
        cursor.execute("SELECT name FROM watermeters WHERE name = ?", (name,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Watermeter not found")
        # Retrieve all evaluations for the watermeter
        cursor.execute("SELECT eval FROM evaluations WHERE name = ?", (name,))
        evals = [row[0] for row in cursor.fetchall()]
        return {"evals": evals}

    # POST endpoint for adding an evaluation
    @app.post("/api/watermeters/{name}/evals", dependencies=[Depends(authenticate)])
    def add_eval(name: str, eval_req: EvalRequest):
        db = db_connection()
        cursor = db.cursor()
        # Check if watermeter exists
        cursor.execute("SELECT name FROM watermeters WHERE name = ?", (name,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Watermeter not found")
        # Insert the new evaluation
        cursor.execute(
            "INSERT INTO evaluations (name, eval) VALUES (?, ?)",
            (name, eval_req.eval)
        )
        db.commit()
        return {"message": "Eval added", "name": name}

    # Serve Vue Frontend from dist directory
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="static")

    print("HTTP-Server: Setup complete.")
    return app
