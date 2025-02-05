from io import BytesIO

from PIL import Image
from fastapi import FastAPI, HTTPException, Query, Header, Depends
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import base64
import sqlite3
from starlette.middleware.cors import CORSMiddleware
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
        yolo_model_path="models/yolo-best-obb.pt",
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

    class ThresholdRequest(BaseModel):
        name: str
        threshold_low: int
        threshold_high: int

    class EvalRequest(BaseModel):
        eval: str

    @app.get("/api/discovery", dependencies=[Depends(authenticate)])
    def get_discovery():
        db = db_connection()
        cursor = db.cursor()
        cursor.execute("SELECT name FROM watermeters")
        existing_meters = {row[0] for row in cursor.fetchall()}
        return list(existing_meters)

    @app.get("/api/evaluate", dependencies=[Depends(authenticate)])
    def evaluate(base64_image: str = Query(...)):
        try:
            image_data = base64.b64decode(base64_image)
            # get pil image from base64
            image = Image.open(BytesIO(image_data))
            result = meter_preditor.predict_single_image(image)
            return result
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image")

    @app.get("/api/watermeters", dependencies=[Depends(authenticate)])
    def get_watermeters():
        print ("get_watermeters")
        cursor = db_connection().cursor()
        cursor.execute("SELECT name FROM watermeters")
        return {"watermeters": [row[0] for row in cursor.fetchall()]}

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

    @app.post("/api/setup", dependencies=[Depends(authenticate)])
    def setup_watermeter(config: ConfigRequest):
        db = db_connection()
        cursor = db.cursor()
        cursor.execute(
            """
            INSERT INTO watermeters (name, picture_number, wifi_rssi, picture_format, 
            picture_timestamp, picture_width, picture_height, picture_length, picture_data) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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

    @app.get("/api/thresholds/{name}", dependencies=[Depends(authenticate)])
    def get_thresholds(name: str):
        cursor = db_connection().cursor()
        cursor.execute("SELECT threshold_low, threshold_high FROM thresholds WHERE name = ?", (name,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Thresholds not found")
        return {"threshold_low": row[0], "threshold_high": row[1]}

    @app.post("/api/thresholds", dependencies=[Depends(authenticate)])
    def set_thresholds(thresholds: ThresholdRequest):
        db = db_connection()
        cursor = db.cursor()
        cursor.execute(
            """
            INSERT INTO thresholds (name, threshold_low, threshold_high) 
            VALUES (?, ?, ?) ON CONFLICT(name) DO UPDATE SET 
            threshold_low=excluded.threshold_low, threshold_high=excluded.threshold_high
            """,
            (thresholds.name, thresholds.threshold_low, thresholds.threshold_high)
        )
        db.commit()
        return {"message": "Thresholds set", "name": thresholds.name}

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
