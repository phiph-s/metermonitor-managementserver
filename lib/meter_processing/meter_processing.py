import base64
import os
import random
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
from tensorflow.keras.models import load_model

class MeterPredictor:
    """
    A class to perform water meter digit detection (using YOLO with OBB)
    and digit classification (using a TFLite model), for one image at a time.
    """

    def __init__(self, yolo_model_path, digit_classifier_model_path):
        """
        Initializes the YOLO model and TFLite Interpreter, so they can
        be reused for multiple images without re-initializing.

        Args:
            yolo_model_path (str): Path to the YOLO .pt file
            tflite_model_path (str): Path to the digit classification TFLite model
        """
        # Load YOLO model (oriented bounding box capable)
        self.model = YOLO(yolo_model_path)
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')

        # Define the model architecture (same as during training)
        self.digitmodel = load_model('models/th_digit_classifier_large.h5')
        self.class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'r']

    def predict_single_image(self, input_image, segments=7, contrast=1.0, padding=0.0, enhance_sharpness=False, extended_last_digit=False, shrink_last_3=False):
        """
        Predicts the water meter reading on a single image:
          - Runs YOLO detection for oriented bounding box (OBB)
          - Applies perspective transform to 'straighten' the meter
          - Splits the meter into 7 vertical segments
          - Classifies each segment using the TFLite model
          - Saves two images:
               1) The original with bounding box & predicted number (prefixed "pred_")
               2) The cropped/straightened region (prefixed "pred_cut_")

        Args:
            input_image_path (str): Path to the input image
            output_folder (str, None): Folder where to save results
        """
        results = self.model.predict(input_image, save=False)
        obb_data = results[0].obb

        # If no OBB found, bail out
        if (
                obb_data is None or
                obb_data.xyxyxyxy is None or
                len(obb_data.xyxyxyxy) == 0
        ):
            print(f"[INFO] No instances detected in the image.")
            [],[]

        # We'll use the first detected bounding box
        obb_coords = obb_data.xyxyxyxy[0].cpu().numpy()
        img = np.array(input_image)

        if enhance_sharpness:
            img = cv2.addWeighted(img, 4, cv2.blur(img, (30, 30)), -4, 128)

        # Apply contrast adjustment
        img = cv2.convertScaleAbs(img, alpha=contrast, beta=0)

        # 3. Add padding at the bottom for annotation
        img = cv2.copyMakeBorder(img, 0, 50, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])

        # 4. Reshape OBB coordinates into four (x,y) points
        points = obb_coords.reshape(4, 2).astype(np.float32)

        # Sort the points by y-coordinate (top to bottom)
        points = sorted(points, key=lambda x: x[1])

        # Separate top-left, top-right vs bottom-left, bottom-right
        if points[0][0] < points[1][0]:
            top_left, top_right = points[0], points[1]
        else:
            top_left, top_right = points[1], points[0]

        if points[2][0] < points[3][0]:
            bottom_left, bottom_right = points[2], points[3]
        else:
            bottom_left, bottom_right = points[3], points[2]

        # Reassemble into final order: [top-left, top-right, bottom-right, bottom-left]
        points = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")

        # 5. Compute bounding box width/height
        width_a = np.linalg.norm(points[0] - points[1])
        width_b = np.linalg.norm(points[2] - points[3])
        max_width = max(int(width_a), int(width_b))

        height_a = np.linalg.norm(points[1] - points[2])
        height_b = np.linalg.norm(points[3] - points[0])
        max_height = max(int(height_a), int(height_b))

        # 6. Perspective transform to get the "front-facing" rectangle
        dst_points = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(points, dst_points)
        rotated_cropped_img = cv2.warpPerspective(img, M, (max_width, max_height))
        rotated_cropped_img_ext = None
        if extended_last_digit:
            rotated_cropped_img_ext = cv2.warpPerspective(img, M, (max_width, int(max_height * 1.2)))

        # 7. Split the cropped meter into segments vertical parts for classification
        part_width = rotated_cropped_img.shape[1] // segments

        base64s = []
        digits = []

        last_x = 0

        for i in range(segments):
            if shrink_last_3 and i >= segments - 3:
                t_part_width = int(part_width * 0.8)
            elif shrink_last_3:
                t_part_width = int(((part_width * segments) - (3 * part_width * 0.8)) / (segments - 3))
            else:
                t_part_width = part_width

            # Extract segment from last_x to last_x + t_part_width
            part = rotated_cropped_img[:, last_x: last_x + t_part_width]

            if extended_last_digit and i == segments - 1:
                part = rotated_cropped_img_ext[:, i * t_part_width: (i + 1) * t_part_width]

            last_x = last_x + t_part_width

            # Convert segment to base64 string for storage
            digits.append(part)

        mean_brightnesses = [np.mean(img) for img in digits]
        # Adjust brightness of each image
        adjusted_images = []
        target_brightness = np.mean(mean_brightnesses)
        for img, mean_brightness in zip(digits, mean_brightnesses):
            adjustment_factor = target_brightness / mean_brightness
            adjusted_img = np.clip(img * adjustment_factor, 0, 255).astype(np.uint8)
            adjusted_images.append(adjusted_img)

        digits = adjusted_images

        for part in digits:
            pil_img = Image.fromarray(part)

            buffered = BytesIO()
            pil_img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue())
            # to string
            img_str = img_str.decode('utf-8')

            base64s.append(img_str)

        return base64s, digits

    def apply_threshold(self, digit, threshold_low, threshold_high, invert=False):
        if invert:
            digit = cv2.bitwise_not(digit)

        # Convert the digit image to grayscale.
        digit = cv2.cvtColor(digit, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to get a binary image.
        digit = cv2.inRange(digit, threshold_low, threshold_high)

        # --- Crop to content with extra vertical padding (10% on top and bottom) ---
        # Assuming background is white (255) and the content is darker.
        coords = np.column_stack(np.where(digit != 255))
        if coords.size > 0:
            # Get the bounding box of non-background pixels.
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0)
            # Compute the height of the content.
            content_height = y1 - y0 + 1
            # Calculate 10% of the content height as padding.
            pad = int(0.1 * content_height)
            # Expand the bounding box vertically (making sure we don't go out of bounds).
            new_y0 = max(0, y0 - pad)
            new_y1 = min(digit.shape[0] - 1, y1 + pad)
            digit_cropped = digit[new_y0:new_y1 + 1, x0:x1 + 1]
        else:
            # If no content is found, use the whole image.
            digit_cropped = digit

        # --- Resize while preserving aspect ratio ---
        target_width, target_height = 40, 64
        h, w = digit_cropped.shape[:2]
        # Determine the scaling factor so that the image fits within the target dimensions.
        scale = min(target_width / w, target_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        digit_resized = cv2.resize(digit_cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # --- Place the resized image on a white canvas of the target size ---
        digit_padded = np.full((target_height, target_width), 255, dtype=np.uint8)
        x_offset = (target_width - new_w) // 2
        y_offset = (target_height - new_h) // 2
        digit_padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = digit_resized

        # --- Normalize & add extra dimensions ---
        img_norm = digit_padded.astype('float32') / 255.0
        img_norm = np.expand_dims(img_norm, axis=-1)  # add channel dimension
        img_norm = np.expand_dims(img_norm, axis=0)  # add batch dimension

        # --- Encode image to base64 ---
        pil_img = Image.fromarray(digit_padded)
        buffered = BytesIO()
        pil_img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return img_str, img_norm

    def predict_digit(self, digit):
        predictions = self.digitmodel.predict(digit)
        top3 = np.argsort(predictions[0])[-3:][::-1]
        pairs = [(self.class_names[i], float(predictions[0][i])) for i in top3]
        return pairs


    def predict_digits(self, digits):
        """
        Digits are np arrays
        predict the digits
        """

        # Predict each digit
        predicted_digits = []
        for i,digit in enumerate(digits):
            digit = self.predict_digit(digit)
            predicted_digits.append(digit)

        return predicted_digits

    def apply_thresholds(self, digits, thresholds, invert=False):
        """
        Digits are np arrays
        apply black/white thresholding to each digit
        """

        # Apply thresholding
        thresholded_digits = []
        base64s = []

        threshold_low = thresholds[0]
        threshold_high = thresholds[1]
        for i, digit in enumerate(digits):
            img_str, digit = self.apply_threshold(digit, threshold_low, threshold_high, invert)

            thresholded_digits.append(digit)
            base64s.append(img_str)

        return base64s, thresholded_digits