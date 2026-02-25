import base64
import cv2
import numpy as np
import sqlite3
from PIL import Image
from lib.meter_processing.roi_extractors.base import ROIExtractor, ROIExtractorTemplated


class ORBExtractor(ROIExtractorTemplated):
    """
    ORB-based ROI extractor with masked feature matching.

    Uses ORB features and BFMatcher for robust ROI extraction
    even with lighting changes and slight camera shifts.
    """

    def __init__(self, reference_image, config_dict):
        """
        Initialize ORB extractor.

        Args:
            reference_image: Reference image (numpy array)
            config_dict: Configuration with:
                - display_corners: 4 points [(x,y), ...] clockwise
                - target_width/height: Output size
                - target_width_ext/height_ext: Extended output size
                - min_inliers: Minimum inliers (default: 10)
                - inlier_ratio_threshold: Minimum ratio (default: 0.3)
                - max_reprojection_error: Max RANSAC error (default: 3.0)
                - matching_mask_padding: Padding around display (default: 10)
                - orb_nfeatures: Number of features (default: 2000)
                - orb_scale_factor: Pyramid scaling (default: 1.2)
                - orb_nlevels: Pyramid levels (default: 8)
        """
        super().__init__(reference_image, config_dict)

        # Parse config with defaults
        self.display_corners = np.array(config_dict['display_corners'], dtype=np.float32)

        target_width = config_dict.get('target_width')
        target_height = config_dict.get('target_height')
        if target_width is None or target_height is None:
            target_width, target_height = self._estimate_target_size(self.display_corners)
        self.target_width = int(target_width)
        self.target_height = int(target_height)

        target_width_ext = config_dict.get('target_width_ext')
        target_height_ext = config_dict.get('target_height_ext')
        if target_width_ext is None or target_height_ext is None:
            target_width_ext = int(round(self.target_width * 1.2))
            target_height_ext = int(round(self.target_height * 1.2))
        self.target_width_ext = int(target_width_ext)
        self.target_height_ext = int(target_height_ext)

        self.min_inliers = config_dict.get('min_inliers', 6)
        self.inlier_ratio_threshold = config_dict.get('inlier_ratio_threshold', 0.2)
        self.max_reprojection_error = config_dict.get('max_reprojection_error', 5.0)
        self.matching_mask_padding = config_dict.get('matching_mask_padding', 4)
        self.lowe_ratio = config_dict.get('lowe_ratio', 0.8)

        self.segment_mode = config_dict.get('segment_mode', 'display')
        digit_corners = config_dict.get('digit_corners')
        if digit_corners:
            self.digit_corners = np.array(digit_corners, dtype=np.float32)
        else:
            self.digit_corners = None

        # ORB parameters
        orb_nfeatures = config_dict.get('orb_nfeatures', 3000)
        orb_scale_factor = config_dict.get('orb_scale_factor', 1.2)
        orb_nlevels = config_dict.get('orb_nlevels', 8)

        # Initialize detector and matcher
        self.detector = cv2.ORB_create(
            nfeatures=orb_nfeatures,
            scaleFactor=orb_scale_factor,
            nlevels=orb_nlevels
        )
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Precomputed data (will be loaded or computed)
        self.matching_mask = None
        self.ref_keypoints = None
        self.ref_descriptors = None

    @classmethod
    def from_database(cls, db_connection, template_name):
        """
        Load template from SQLite database.

        Args:
            db_connection: sqlite3.Connection
            template_name: Template name

        Returns:
            ORBExtractor instance
        """
        cursor = db_connection.cursor()
        cursor.execute("""
            SELECT reference_image_base64, config_json, precomputed_data_base64
            FROM templates
            WHERE id = ? OR name = ?
            LIMIT 1
        """, (template_name, template_name))

        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Template '{template_name}' not found!")

        ref_img_b64, config_json, precomputed_b64 = row
        return cls.deserialize_template(ref_img_b64, config_json, precomputed_b64)

    def compute_precomputed_data(self):
        """
        Compute features and matching mask for cache.

        Returns:
            dict with: matching_mask, ref_keypoints, ref_descriptors
        """
        # Create matching mask
        h, w = self.reference_gray.shape
        mask = np.ones((h, w), dtype=np.uint8) * 255

        poly = self.display_corners.astype(np.int32)
        cv2.fillPoly(mask, [poly], 0)

        # Add padding
        if self.matching_mask_padding > 0:
            center = poly.mean(axis=0)
            expanded_poly = []
            for pt in poly:
                direction = pt - center
                length = np.linalg.norm(direction)
                if length > 0:
                    direction = direction / length
                    expanded_pt = pt + direction * self.matching_mask_padding
                    expanded_poly.append(expanded_pt)
            expanded_poly = np.array(expanded_poly, dtype=np.int32)
            cv2.fillPoly(mask, [expanded_poly], 0)

        # Extract features (fallback to full image if masked area is too sparse)
        keypoints, descriptors = self.detector.detectAndCompute(self.reference_gray, mask)
        min_ref_features = max(10, int(self.min_inliers))
        if descriptors is None or len(keypoints) < min_ref_features:
            keypoints, descriptors = self.detector.detectAndCompute(self.reference_gray, None)
            if descriptors is None or len(keypoints) < min_ref_features:
                found = 0 if keypoints is None else len(keypoints)
                raise ValueError(f"Too few features found in reference image! ({found})")
            mask = None

        return {
            'matching_mask': mask,
            'ref_keypoints': keypoints,
            'ref_descriptors': descriptors
        }

    def load_precomputed_data(self, precomputed_dict):
        """
        Load precomputed features from cache.

        Args:
            precomputed_dict: Dictionary with matching_mask, ref_keypoints, ref_descriptors
        """
        self.matching_mask = precomputed_dict['matching_mask']
        self.ref_keypoints = precomputed_dict['ref_keypoints']
        self.ref_descriptors = precomputed_dict['ref_descriptors']

    def extract(self, input_image):
        """
        Extract ROI from input image.

        Args:
            input_image: Input image (numpy array, BGR or grayscale)

        Returns:
            (cropped, cropped_ext, boundingboxed_image) or (None, None, None)
        """
        self.last_error = None
        if isinstance(input_image, Image.Image):
            input_image = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)

        H, num_inliers, inlier_ratio = self._compute_homography(input_image)
        if H is None:
            return None, None, None

        # 4. Transform ROI corners
        corners_homog = np.hstack([self.display_corners, np.ones((4, 1))])
        transformed = (H @ corners_homog.T).T
        transformed_corners = (transformed[:, :2] / transformed[:, 2:3]).astype(np.float32)

        # 5. Extract display
        cropped = self._warp_roi(input_image, transformed_corners,
                                 self.target_width, self.target_height)
        cropped_ext = self._warp_roi(input_image, transformed_corners,
                                     self.target_width_ext, self.target_height_ext)
        boundingboxed = self._draw_bbox(input_image, transformed_corners)

        print("[ROIExtractor (ORB)]" + f"Success: {num_inliers} inliers, ratio: {inlier_ratio:.2f}")
        return cropped, cropped_ext, boundingboxed

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

        self.last_error = None
        if isinstance(input_image, Image.Image):
            input_image = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)

        H, num_inliers, inlier_ratio = self._compute_homography(input_image)
        if H is None:
            return [], None

        transformed_quads = []
        digits = []
        for idx, quad in enumerate(self.digit_corners):
            quad = np.array(quad, dtype=np.float32)
            if quad.shape != (4, 2):
                self.last_error = f"Invalid digit quad at index {idx}"
                return [], None
            quad_homog = np.hstack([quad, np.ones((4, 1))])
            transformed = (H @ quad_homog.T).T
            quad_transformed = (transformed[:, :2] / transformed[:, 2:3]).astype(np.float32)
            warped = ROIExtractor.warp_quad(input_image, quad_transformed)
            if warped is None:
                self.last_error = f"Failed to warp digit quad {idx}"
                return [], None
            transformed_quads.append(quad_transformed)
            digits.append(warped)

        boundingboxed = self._draw_digit_quads(input_image, transformed_quads)
        print("[ROIExtractor (ORB)]" + f"Success (each_digit): {num_inliers} inliers, ratio: {inlier_ratio:.2f}")
        return digits, boundingboxed

    def _compute_homography(self, input_image):
        # Ensure precomputed data is available
        if self.ref_descriptors is None:
            precomputed = self.compute_precomputed_data()
            self.load_precomputed_data(precomputed)

        # Convert to grayscale
        if len(input_image.shape) == 3:
            gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = input_image

        # 1. Feature extraction in new image
        kp_new, desc_new = self.detector.detectAndCompute(gray, None)

        if desc_new is None or len(kp_new) < 10:
            self.last_error = "Too few features in input image"
            print("[ROIExtractor (ORB)]" + self.last_error)
            return None, None, None

        # 2. Feature matching with Lowe's ratio test
        matches = self.matcher.knnMatch(self.ref_descriptors, desc_new, k=2)

        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.lowe_ratio * n.distance:
                    good_matches.append(m)

        if len(good_matches) < self.min_inliers:
            self.last_error = f"Too few matches: {len(good_matches)}"
            print("[ROIExtractor (ORB)]" + self.last_error)
            return None, None, None

        # 3. Homography estimation with RANSAC
        src_pts = np.float32([self.ref_keypoints[m.queryIdx].pt for m in good_matches])
        dst_pts = np.float32([kp_new[m.trainIdx].pt for m in good_matches])

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.max_reprojection_error)

        if H is None:
            self.last_error = "Homography estimation failed"
            print("[ROIExtractor (ORB)]" + self.last_error)
            return None, None, None

        num_inliers = int(np.sum(mask))
        inlier_ratio = num_inliers / len(good_matches)

        # Quality check
        if num_inliers < self.min_inliers:
            self.last_error = f"Too few inliers: {num_inliers}"
            print("[ROIExtractor (ORB)]" + self.last_error)
            return None, None, None

        if inlier_ratio < self.inlier_ratio_threshold:
            self.last_error = f"Inlier ratio too low: {inlier_ratio:.2f}"
            print("[ROIExtractor (ORB)]" + self.last_error)
            return None, None, None

        return H, num_inliers, inlier_ratio

    def _draw_digit_quads(self, image, quads):
        result = image.copy()
        for i, quad in enumerate(quads):
            pts = np.array(quad, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(result, [pts], True, (0, 255, 0), 2)
            center = np.mean(quad, axis=0).astype(int)
            cv2.putText(result, str(i), tuple(center),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        max_dim = max(result.shape[0], result.shape[1])
        if max_dim > 1600:
            scale = 1600 / max_dim
            new_w = int(result.shape[1] * scale)
            new_h = int(result.shape[0] * scale)
            result = cv2.resize(result, (new_w, new_h), interpolation=cv2.INTER_AREA)
        success, buffer = cv2.imencode('.png', result)
        if not success:
            return None
        return base64.b64encode(buffer).decode('utf-8')

    def _estimate_target_size(self, corners):
        """Estimate target size from ROI corners."""
        width_a = np.linalg.norm(corners[0] - corners[1])
        width_b = np.linalg.norm(corners[2] - corners[3])
        height_a = np.linalg.norm(corners[0] - corners[3])
        height_b = np.linalg.norm(corners[1] - corners[2])
        target_width = int(round(max(width_a, width_b)))
        target_height = int(round(max(height_a, height_b)))
        if target_width <= 0 or target_height <= 0:
            target_width = 400
            target_height = 200
        return target_width, target_height

    def _warp_roi(self, image, corners, width, height):
        """Warp ROI to target size."""
        dst_corners = np.array([
            [0, 0], [width - 1, 0],
            [width - 1, height - 1], [0, height - 1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(corners, dst_corners)
        return cv2.warpPerspective(image, M, (width, height))

    def _draw_bbox(self, image, corners):
        """Draw bounding box on image."""
        result = image.copy()
        pts = corners.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(result, [pts], True, (0, 255, 0), 2)
        for i, pt in enumerate(corners):
            cv2.putText(result, str(i), tuple(pt.astype(int)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        max_dim = max(result.shape[0], result.shape[1])
        if max_dim > 1600:
            scale = 1600 / max_dim
            new_w = int(result.shape[1] * scale)
            new_h = int(result.shape[0] * scale)
            result = cv2.resize(result, (new_w, new_h), interpolation=cv2.INTER_AREA)
        success, buffer = cv2.imencode('.png', result)
        if not success:
            return None
        return base64.b64encode(buffer).decode('utf-8')
