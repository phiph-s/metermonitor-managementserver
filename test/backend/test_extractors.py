import base64
from io import BytesIO
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.meter_processing.roi_extractors.bypass_extractor import BypassExtractor
from lib.meter_processing.roi_extractors.orb_extractor import ORBExtractor
from lib.meter_processing.roi_extractors.yolo_extractor import YOLOExtractor


class FakeYoloSession:
    def __init__(self, output):
        self.output = output

    def run(self, *_args, **_kwargs):
        return [self.output]


class TestExtractors(unittest.TestCase):
    def test_bypass_extractor_returns_image_and_bbox(self):
        img = Image.new("RGB", (10, 12), (10, 20, 30))
        extractor = BypassExtractor()

        cropped, cropped_ext, bbox = extractor.extract(img)

        self.assertIsNone(cropped_ext)
        self.assertEqual(cropped.shape[:2], (12, 10))
        self.assertEqual(cropped[0, 0].tolist(), [30, 20, 10])

        bbox_img = Image.open(BytesIO(base64.b64decode(bbox)))
        self.assertEqual(bbox_img.size, (10, 12))

    def test_yolo_extractor_with_fake_session(self):
        output = np.zeros((1, 10, 6), dtype=np.float32)
        output[0, 0] = [0.5, 0.5, 0.4, 0.2, 0.9, 0.1]

        session = FakeYoloSession(output)
        extractor = YOLOExtractor(session, "input", extended_last_digit=True)
        img = Image.new("RGB", (640, 480), (200, 200, 200))

        cropped, cropped_ext, bbox = extractor.extract(img)

        self.assertIsNotNone(cropped)
        self.assertIsNotNone(cropped_ext)
        self.assertGreater(cropped.shape[0], 0)
        self.assertGreater(cropped.shape[1], 0)

        bbox_img = Image.open(BytesIO(base64.b64decode(bbox)))
        self.assertEqual(bbox_img.size, (640, 480))

    def test_orb_extractor_extracts_roi(self):
        root = Path(__file__).resolve().parents[2]
        img_path = root / "test" / "img" / "img.png"
        reference = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        self.assertIsNotNone(reference)

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
        cropped, cropped_ext, bbox = extractor.extract(reference)

        self.assertIsNotNone(cropped)
        self.assertIsNotNone(cropped_ext)
        self.assertEqual(cropped.shape[:2], (90, 220))
        self.assertEqual(cropped_ext.shape[:2], (108, 240))
        self.assertIsNotNone(bbox)

    def test_orb_serialization_roundtrip(self):
        root = Path(__file__).resolve().parents[2]
        img_path = root / "test" / "img" / "img.png"
        reference = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        self.assertIsNotNone(reference)

        config = {
            "display_corners": [
                [262, 282],
                [433, 152],
                [454, 183],
                [286, 312],
            ],
            "min_inliers": 4,
            "inlier_ratio_threshold": 0.1,
            "max_reprojection_error": 10.0,
            "matching_mask_padding": 4,
            "lowe_ratio": 0.9,
            "orb_nfeatures": 1500,
        }

        extractor = ORBExtractor(reference, config)
        ref_b64, config_json, precomputed_b64 = extractor.serialize_template()

        loaded = ORBExtractor.deserialize_template(ref_b64, config_json, precomputed_b64)

        self.assertIsNotNone(loaded.ref_descriptors)
        self.assertIsNotNone(loaded.ref_keypoints)
        self.assertTrue(len(loaded.ref_keypoints) > 0)


if __name__ == "__main__":
    unittest.main()
