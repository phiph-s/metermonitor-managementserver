import base64
import datetime
from io import BytesIO
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.meter_processing.meter_processing import MeterPredictor
from lib.meter_processing.roi_extractors.orb_extractor import ORBExtractor
from lib.history_correction import correct_value


class TestFullPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[2]
        img_path = root / "test" / "img" / "img.png"
        reference = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if reference is None:
            raise AssertionError("Failed to load test image for pipeline test")

        config = {
            "display_corners": [
                [262, 282],
                [433, 152],
                [454, 183],
                [286, 312],
            ],
            "target_width": 220,
            "target_height": 90,
            "target_width_ext": 240,
            "target_height_ext": 108,
            "min_inliers": 4,
            "inlier_ratio_threshold": 0.1,
            "max_reprojection_error": 10.0,
            "matching_mask_padding": 4,
            "lowe_ratio": 0.9,
            "orb_nfeatures": 1500,
        }

        extractor = ORBExtractor(reference, config)
        predictor = MeterPredictor()

        input_image = Image.open(img_path)
        base64s, digits, target_brightness, bbox = predictor.extract_display_and_segment(
            input_image,
            segments=7,
            shrink_last_3=False,
            extended_last_digit=False,
            rotated_180=True,
            target_brightness=None,
            roi_extractor="orb",
            extractor_instance=extractor,
        )

        cls.predictor = predictor
        cls.base64s = base64s
        cls.digits = digits
        cls.target_brightness = target_brightness
        cls.bbox = bbox
        cls.thresholds = [0, 117]
        cls.thresholds_last = [0, 120]
        cls.islanding_padding = 20

        _, thresholded, _ = predictor.apply_thresholds(
            digits, cls.thresholds, cls.thresholds_last, islanding_padding=cls.islanding_padding
        )
        cls.predictions = predictor.predict_digits(thresholded)
        cls.top1 = "".join(p[0][0] for p in cls.predictions)

    def test_full_pipeline_orb_with_onnx(self):
        self.assertEqual(len(self.base64s), 7)
        self.assertEqual(len(self.digits), 7)
        self.assertIsNotNone(self.target_brightness)
        self.assertIsNotNone(self.bbox)

        bbox_img = Image.open(BytesIO(base64.b64decode(self.bbox)))
        self.assertGreater(bbox_img.size[0], 0)
        self.assertGreater(bbox_img.size[1], 0)

        self.assertEqual(len(self.predictions), 7)
        self.assertEqual(self.top1, "004330r")

        for pred in self.predictions:
            self.assertGreaterEqual(len(pred), 1)
            self.assertIsInstance(pred[0][0], str)
            self.assertIsInstance(pred[0][1], float)

    def test_correction_algorithm_with_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/test.db"
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "CREATE TABLE history (name TEXT, value INTEGER, confidence REAL, target_brightness REAL, timestamp TEXT, manual BOOLEAN)"
                )
                last_ts = (datetime.datetime.now() - datetime.timedelta(minutes=10)).isoformat()
                cursor.execute(
                    "INSERT INTO history (name, value, confidence, target_brightness, timestamp, manual) VALUES (?,?,?,?,?,?)",
                    ("meter-1", 43290, 0.9, None, last_ts, False),
                )
                conn.commit()

            denied_digits = [False] * len(self.predictions)
            new_eval = [None, None, self.predictions, datetime.datetime.now().isoformat(), denied_digits]
            result = correct_value(
                db_path,
                "meter-1",
                new_eval,
                allow_negative_correction=False,
                max_flow_rate=1000.0,
                use_full_correction=True,
            )

        self.assertTrue(result["accepted"])
        self.assertEqual(result["value"], 43300)
        self.assertIsNone(result["rejection_reason"])


if __name__ == "__main__":
    unittest.main()
