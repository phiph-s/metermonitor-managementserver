from lib.log import log
import base64
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from lib.meter_processing.roi_extractors.base import ROIExtractor


class YOLOExtractor(ROIExtractor):
    def __init__(self, yolo_session, yolo_input_name, extended_last_digit=False):
        self.yolo_session = yolo_session
        self.yolo_input_name = yolo_input_name
        self.extended_last_digit = extended_last_digit

    def extract(self, input_image):
        self.last_error = None
        log("[ROIExtractor (YOLO)] Running YOLO region-of-interest detection...")

        img_np = np.array(input_image)
        # Convert RGB (from PIL) to BGR for consistent OpenCV processing
        if img_np.ndim == 3 and img_np.shape[2] >= 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        # Ensure 3-channel RGB input (drop alpha or expand grayscale)
        if img_np.ndim == 2:
            img_np = np.repeat(img_np[:, :, None], 3, axis=2)
        elif img_np.shape[2] == 4:
            img_np = img_np[:, :, :3]
        original_height, original_width = img_np.shape[:2]

        target_size = 640
        scale = min(target_size / original_width, target_size / original_height)
        new_w = int(original_width * scale)
        new_h = int(original_height * scale)

        img_resized = cv2.resize(img_np, (new_w, new_h))
        canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        top = (target_size - new_h) // 2
        left = (target_size - new_w) // 2
        canvas[top:top + new_h, left:left + new_w] = img_resized

        img_normalized = canvas.astype(np.float32) / 255.0
        img_transposed = img_normalized.transpose(2, 0, 1)
        img_batch = np.expand_dims(img_transposed, axis=0)

        try:
            outputs = self.yolo_session.run(None, {self.yolo_input_name: img_batch})
        except Exception as e:
            self.last_error = f"YOLO inference failed: {e}"
            log(f"[ROIExtractor (YOLO)] {self.last_error}")
            return None, None, None

        output = outputs[0]
        if output.shape[1] < output.shape[2]:
            output = output.transpose(0, 2, 1)

        predictions = output[0]
        if predictions.shape[1] < 6:
            self.last_error = "Invalid YOLO output shape."
            log(f"[ROIExtractor (YOLO)] {self.last_error}")
            return None, None, None

        if predictions.shape[1] == 6:
            col4 = predictions[:, 4]
            col5 = predictions[:, 5]
            max4 = np.max(col4)
            max5 = np.max(col5)
            if max5 > 1.05:
                confidence_idx = 4
                rotation_idx = 5
            elif max4 > 1.05:
                confidence_idx = 5
                rotation_idx = 4
            else:
                confidence_idx = 4
                rotation_idx = 5
        else:
            rotation_idx = -1
            for i in range(4, predictions.shape[1]):
                if np.max(predictions[:, i]) > 1.05:
                    rotation_idx = i
                    break
            if rotation_idx != -1:
                class_cols = [i for i in range(4, predictions.shape[1]) if i != rotation_idx]
                confidence_idx = class_cols[0] if class_cols else 4
            else:
                rotation_idx = predictions.shape[1] - 1
                confidence_idx = 4

        if predictions.shape[1] > 6:
            class_cols = [i for i in range(4, predictions.shape[1]) if i != rotation_idx]
            confidences = np.max(predictions[:, class_cols], axis=1) if class_cols else predictions[:, confidence_idx]
        else:
            confidences = predictions[:, confidence_idx]

        valid_mask = confidences > 0.15
        if not np.any(valid_mask):
            self.last_error = "No instances detected with confidence > 0.15"
            log(f"[ROIExtractor (YOLO)] {self.last_error}")
            return None, None, None

        valid_predictions = predictions[valid_mask]
        valid_confidences = confidences[valid_mask]
        best_idx = np.argmax(valid_confidences)
        detection = valid_predictions[best_idx]

        x_center_norm = detection[0]
        y_center_norm = detection[1]
        width_norm = detection[2]
        height_norm = detection[3]
        rotation = detection[rotation_idx]

        if x_center_norm < 2.0 and y_center_norm < 2.0 and width_norm < 2.0 and height_norm < 2.0:
            x_center_pixel = x_center_norm * 640.0
            y_center_pixel = y_center_norm * 640.0
            width_pixel = width_norm * 640.0
            height_pixel = height_norm * 640.0
        else:
            x_center_pixel = x_center_norm
            y_center_pixel = y_center_norm
            width_pixel = width_norm
            height_pixel = height_norm

        x_center_pixel -= left
        y_center_pixel -= top

        x_center = x_center_pixel / scale
        y_center = y_center_pixel / scale
        width = width_pixel / scale
        height = height_pixel / scale

        if height > width:
            width, height = height, width
            rotation = rotation + (np.pi / 2.0)

        hw = width / 2.0
        hh = height / 2.0
        corners_unrotated = np.array([
            [-hw, -hh],
            [hw, -hh],
            [hw, hh],
            [-hw, hh]
        ], dtype=np.float32)

        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)
        rotation_matrix = np.array([
            [cos_r, -sin_r],
            [sin_r, cos_r]
        ], dtype=np.float32)

        corners_rotated = corners_unrotated @ rotation_matrix.T
        obb_coords = corners_rotated + np.array([x_center, y_center], dtype=np.float32)

        pts = obb_coords.reshape(4, 2).astype(np.float32)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).ravel()
        top_left = pts[np.argmin(s)]
        bottom_right = pts[np.argmax(s)]
        top_right = pts[np.argmin(diff)]
        bottom_left = pts[np.argmax(diff)]
        points = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")

        width_a = np.linalg.norm(points[0] - points[1])
        width_b = np.linalg.norm(points[2] - points[3])
        max_width = max(int(width_a), int(width_b))

        height_a = np.linalg.norm(points[1] - points[2])
        height_b = np.linalg.norm(points[3] - points[0])
        max_height = max(int(height_a), int(height_b))

        if max_width <= 0 or max_height <= 0:
            self.last_error = "Invalid ROI size."
            log(f"[ROIExtractor (YOLO)] {self.last_error}")
            return None, None, None

        dst_points = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(points, dst_points)
        rotated_cropped_img = cv2.warpPerspective(img_np, M, (max_width, max_height))
        rotated_cropped_img_ext = None

        if rotated_cropped_img.shape[0] > rotated_cropped_img.shape[1]:
            rotated_cropped_img = cv2.rotate(rotated_cropped_img, cv2.ROTATE_90_CLOCKWISE)

        if self.extended_last_digit:
            rotated_cropped_img_ext = cv2.warpPerspective(img_np, M, (max_width, int(max_height * 1.2)))
            if rotated_cropped_img_ext.shape[0] > rotated_cropped_img_ext.shape[1]:
                rotated_cropped_img_ext = cv2.rotate(rotated_cropped_img_ext, cv2.ROTATE_90_CLOCKWISE)

        # Create bounding box visualization (needs to be in RGB for PIL)
        img_with_bbox = img_np.copy()
        boundingboxed_image = None
        if obb_coords is not None:
            obb_points = obb_coords.reshape(4, 2).astype(np.int32)
            cv2.polylines(img_with_bbox, [obb_points], isClosed=True, color=(0, 0, 255), thickness=2)  # BGR red
            # Convert back to RGB for PIL
            img_with_bbox_rgb = cv2.cvtColor(img_with_bbox, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_with_bbox_rgb)
            buffered = BytesIO()
            pil_img.save(buffered, format="PNG")
            boundingboxed_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return rotated_cropped_img, rotated_cropped_img_ext, boundingboxed_image
