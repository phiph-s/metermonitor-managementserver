import base64
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from lib.meter_processing.roi_extractors.base import ROIExtractor


class BypassExtractor(ROIExtractor):
    def __init__(self, rotated_180=False):
        self.rotated_180 = rotated_180

    def extract(self, input_image):
        print("[ROIExtractor (Bypass)] Bypassing region-of-interest detection...")
        self.last_error = None
        img_np = np.array(input_image)

        # Convert RGB (from PIL) to BGR for consistent OpenCV processing
        if img_np.ndim == 3 and img_np.shape[2] >= 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        if self.rotated_180:
            img_np = cv2.rotate(img_np, cv2.ROTATE_180)

        height, width = img_np.shape[:2]

        img_with_bbox = img_np.copy()
        cv2.rectangle(img_with_bbox, (0, 0), (width - 1, height - 1), (255, 0, 0), 2)

        # Convert back to RGB for PIL
        img_with_bbox_rgb = cv2.cvtColor(img_with_bbox, cv2.COLOR_BGR2RGB) if img_with_bbox.ndim == 3 else img_with_bbox
        pil_img = Image.fromarray(img_with_bbox_rgb)
        buffered = BytesIO()
        pil_img.save(buffered, format="PNG")
        boundingboxed_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return img_np, None, boundingboxed_image
