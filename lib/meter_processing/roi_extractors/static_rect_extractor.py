from lib.log import log
"""
Static Rectangle ROI Extractor

Simple extractor that crops a fixed rectangular region from the input image
without any alignment or transformation. The user defines the rectangle corners
and the extractor simply crops that region.
"""

import base64
import json
import numpy as np
import cv2
from lib.meter_processing.roi_extractors.base import ROIExtractor, ROIExtractorTemplated


class StaticRectExtractor(ROIExtractorTemplated):
    """
    Extract ROI by cropping a fixed rectangle from the input image.

    No alignment, feature matching, or perspective transformation is performed.
    The user simply defines the 4 corners of the rectangle and this extractor
    crops that exact region from any input image.
    """

    def __init__(self, reference_image, config_dict):
        """
        Initialize static rectangle extractor.

        Args:
            reference_image: Reference image (used only for visualization/template creation)
            config_dict: Must contain 'display_corners' with 4 points [x, y]
        """
        super().__init__(reference_image, config_dict)

        # Extract corners from config
        if 'display_corners' not in config_dict:
            raise ValueError("config_dict must contain 'display_corners'")

        self.corners = np.array(config_dict['display_corners'], dtype=np.float32)

        if self.corners.shape != (4, 2):
            raise ValueError("display_corners must be a 4x2 array")

        # Calculate target dimensions
        self.target_width = config_dict.get('target_width', 400)
        self.target_height = config_dict.get('target_height', 100)
        self.target_width_ext = config_dict.get('target_width_ext', int(self.target_width * 1.2))
        self.target_height_ext = config_dict.get('target_height_ext', int(self.target_height * 1.2))

        self.segment_mode = config_dict.get('segment_mode', 'display')
        digit_corners = config_dict.get('digit_corners')
        if digit_corners:
            self.digit_corners = np.array(digit_corners, dtype=np.float32)
        else:
            self.digit_corners = None

        # Calculate bounding box for the rectangle
        self._compute_bbox()

    def _compute_bbox(self):
        """Compute axis-aligned bounding box of the rectangle."""
        x_coords = self.corners[:, 0]
        y_coords = self.corners[:, 1]

        self.x_min = int(np.floor(np.min(x_coords)))
        self.y_min = int(np.floor(np.min(y_coords)))
        self.x_max = int(np.ceil(np.max(x_coords)))
        self.y_max = int(np.ceil(np.max(y_coords)))

        self.bbox_width = self.x_max - self.x_min
        self.bbox_height = self.y_max - self.y_min

    def compute_precomputed_data(self):
        """
        For static rect extractor, we don't need any precomputed data.
        Everything is defined by the corners in the config.
        """
        return {
            "type": "static_rect",
            "bbox": {
                "x_min": self.x_min,
                "y_min": self.y_min,
                "x_max": self.x_max,
                "y_max": self.y_max,
                "width": self.bbox_width,
                "height": self.bbox_height
            }
        }

    def load_precomputed_data(self, precomputed_dict):
        """
        Load precomputed data (bounding box).

        Args:
            precomputed_dict: Dictionary with precomputed bbox info
        """
        if precomputed_dict and 'bbox' in precomputed_dict:
            bbox = precomputed_dict['bbox']
            self.x_min = bbox['x_min']
            self.y_min = bbox['y_min']
            self.x_max = bbox['x_max']
            self.y_max = bbox['y_max']
            self.bbox_width = bbox['width']
            self.bbox_height = bbox['height']

    def extract(self, input_image):
        """
        Extract ROI by cropping the fixed rectangle from input image.

        Args:
            input_image: Input image (numpy array)

        Returns:
            tuple: (cropped, cropped_ext, boundingboxed_image)
                - cropped: Warped display region at target size
                - cropped_ext: Warped extended region at target_ext size
                - boundingboxed_image: Input image with rectangle drawn on it
        """
        img_height, img_width = input_image.shape[:2]

        # Validate corners are within image bounds
        if (np.any(self.corners[:, 0] < 0) or np.any(self.corners[:, 0] >= img_width) or
            np.any(self.corners[:, 1] < 0) or np.any(self.corners[:, 1] >= img_height)):
            log("[StaticRect] Corners are outside image bounds")
            return None, None, None

        # Use perspective transform to warp the quadrilateral to a rectangle
        # This allows handling of non-axis-aligned rectangles
        dst_points = np.array([
            [0, 0],
            [self.target_width - 1, 0],
            [self.target_width - 1, self.target_height - 1],
            [0, self.target_height - 1]
        ], dtype=np.float32)

        # Compute perspective transform matrix
        M = cv2.getPerspectiveTransform(self.corners, dst_points)

        # Warp the display region
        cropped = cv2.warpPerspective(input_image, M, (self.target_width, self.target_height))

        # For extended region, use same corners but scale output
        dst_points_ext = np.array([
            [0, 0],
            [self.target_width_ext - 1, 0],
            [self.target_width_ext - 1, self.target_height_ext - 1],
            [0, self.target_height_ext - 1]
        ], dtype=np.float32)

        M_ext = cv2.getPerspectiveTransform(self.corners, dst_points_ext)
        cropped_ext = cv2.warpPerspective(input_image, M_ext, (self.target_width_ext, self.target_height_ext))

        # Create bounding box visualization
        boundingboxed_image = input_image.copy()

        # Draw the rectangle
        pts = self.corners.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(boundingboxed_image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # Draw corner points
        for i, corner in enumerate(self.corners):
            cv2.circle(boundingboxed_image, tuple(corner.astype(int)), 5, (0, 0, 255), -1)
            cv2.putText(boundingboxed_image, str(i), tuple(corner.astype(int) + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Add label
        label_pos = (self.x_min, self.y_min - 10 if self.y_min > 20 else self.y_max + 20)
        cv2.putText(boundingboxed_image, "Static Rect", label_pos,
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        log(f"[StaticRect] Cropped region: {self.target_width}x{self.target_height}")

        success, buffer = cv2.imencode(".png", boundingboxed_image)
        boundingboxed_image_b64 = base64.b64encode(buffer).decode("utf-8") if success else None

        return cropped, cropped_ext, boundingboxed_image_b64

    def extract_segments(self, input_image, segments, extended_last_digit=False, shrink_last_3=False, segment_mode="display"):
        self.last_error = None
        if segment_mode != "each_digit":
            return super().extract_segments(
                input_image,
                segments=segments,
                extended_last_digit=extended_last_digit,
                shrink_last_3=shrink_last_3,
                segment_mode=segment_mode,
            )

        if self.digit_corners is None or len(self.digit_corners) == 0:
            self.last_error = "digit_corners missing for each_digit mode. Please update and save the template."
            return [], None

        if len(self.digit_corners) != segments:
            self.last_error = f"digit_corners count ({len(self.digit_corners)}) does not match segments ({segments}). Please update and save the template."
            return [], None

        img_height, img_width = input_image.shape[:2]
        digits = []

        for idx, quad in enumerate(self.digit_corners):
            quad = np.array(quad, dtype=np.float32)
            if quad.shape != (4, 2):
                self.last_error = f"Invalid digit quad at index {idx}"
                return [], None
            if (np.any(quad[:, 0] < 0) or np.any(quad[:, 0] >= img_width) or
                np.any(quad[:, 1] < 0) or np.any(quad[:, 1] >= img_height)):
                self.last_error = f"Digit quad {idx} is outside image bounds"
                return [], None
            warped = ROIExtractor.warp_quad(input_image, quad)
            if warped is None:
                self.last_error = f"Failed to warp digit quad {idx}"
                return [], None
            digits.append(warped)

        boundingboxed_image = self._draw_digit_quads(input_image, self.digit_corners)
        return digits, boundingboxed_image

    def _draw_digit_quads(self, input_image, quads):
        img = input_image.copy()
        for i, quad in enumerate(quads):
            pts = np.array(quad, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            center = np.mean(quad, axis=0).astype(int)
            cv2.putText(img, str(i), tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        success, buffer = cv2.imencode(".png", img)
        return base64.b64encode(buffer).decode("utf-8") if success else None

    @classmethod
    def from_database(cls, db_connection, template_id):
        """
        Load StaticRectExtractor from database.

        Args:
            db_connection: SQLite database connection
            template_id: Template UUID

        Returns:
            StaticRectExtractor instance
        """
        cursor = db_connection.cursor()
        cursor.execute(
            """
            SELECT reference_image_base64, config_json, precomputed_data_base64
            FROM templates
            WHERE id = ?
            """,
            (template_id,)
        )
        row = cursor.fetchone()

        if not row:
            raise ValueError(f"Template {template_id} not found")

        ref_b64, config_json, precomputed_b64 = row

        # Decode reference image
        img_bytes = base64.b64decode(ref_b64)
        img_np = np.frombuffer(img_bytes, np.uint8)
        reference_image = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        if reference_image is None:
            raise ValueError("Failed to decode reference image")

        # Parse config
        config = json.loads(config_json) if config_json else {}

        # Create instance
        extractor = cls(reference_image, config)

        # Load precomputed data
        if precomputed_b64:
            precomputed_bytes = base64.b64decode(precomputed_b64)
            precomputed_dict = json.loads(precomputed_bytes.decode('utf-8'))
            extractor.load_precomputed_data(precomputed_dict)

        return extractor
