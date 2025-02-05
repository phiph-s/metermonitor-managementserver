import base64
import os
import random
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
import torch
from pytesseract import pytesseract
from sklearn.cluster import KMeans
from torchvision.models import resnet18
from torch import nn
from torchvision.transforms import transforms
from ultralytics import YOLO
from transformers import pipeline

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
        self.digitmodel = resnet18(pretrained=False)  # Ensure pretrained=False if loading your weights
        self.digitmodel.fc = nn.Linear(self.digitmodel.fc.in_features, 11)  # Adjust for your classes

        # Load the saved weights
        state_dict = torch.load(digit_classifier_model_path, map_location=torch.device("cpu"))
        self.digitmodel.load_state_dict(state_dict)

        # Set the model to evaluation mode
        self.digitmodel.eval()

        # Move the model to the appropriate device (CPU or GPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.digitmodel.to(self.device)

        self.classification_pipe = pipeline("image-classification", model="farleyknight/mnist-digit-classification-2022-09-04")

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

        # Convert the digit image to grayscale
        digit = cv2.cvtColor(digit, cv2.COLOR_BGR2GRAY)

        # Apply the thresholding to get black areas
        digit = cv2.inRange(digit, threshold_low, threshold_high)

        inverted = cv2.bitwise_not(digit)

        # Find connected components (8-connectivity by default)
        num_labels, labels = cv2.connectedComponents(inverted)

        # Create a BGR color image with white background
        color_image = np.full((*digit.shape, 3), (255, 255, 255), dtype=np.uint8)

        # Get the dimensions of the image
        height, width = digit.shape

        # Calculate the middle 60% region (with 20% padding on all sides)
        start_x = int(0.4 * width)
        end_x = int(0.6 * width)
        start_y = int(0.4 * height)
        end_y = int(0.6 * height)

        # Assign color based on component's presence in the middle region
        for label in range(1, num_labels):
            # Slice the labels to the middle region and check for any occurrence of the current label
            component_region = labels[start_y:end_y, start_x:end_x]
            in_middle = np.any(component_region == label)

            if in_middle:
                color = (0, 0, 0)  # Red in BGR
            else:
                color = (255, 255, 255)  # Black

            color_image[labels == label] = color

        pil_img = Image.fromarray(color_image)

        buffered = BytesIO()
        pil_img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())
        # to string
        img_str = img_str.decode('utf-8')

        return img_str, color_image

    def resize_and_pad(self, im, desired_size, fill_color=(0, 0, 0)):
        # Get the original size
        old_width, old_height = im.size

        # Calculate the scaling factor to fit the image within the desired_size square
        scale = desired_size / max(old_width, old_height)
        new_width = int(old_width * scale)
        new_height = int(old_height * scale)

        # Resize the image with the calculated scale factor
        im_resized = im.resize((new_width, new_height))

        # Create a new square image with the desired size and fill color (black by default)
        new_im = Image.new("RGB", (desired_size, desired_size), fill_color)

        # Compute top-left coordinates to paste the resized image onto the center of the new image
        top_left_x = (desired_size - new_width) // 2
        top_left_y = (desired_size - new_height) // 2

        # Paste the resized image onto the square background
        new_im.paste(im_resized, (top_left_x, top_left_y))

        return new_im

    def predict_digit(self, digit):
        # invert digit
        digit = cv2.bitwise_not(digit)
        pil_img = Image.fromarray(digit)
        pil_img = self.resize_and_pad(pil_img, 224)

        r = self.classification_pipe(pil_img)
        class_name = r[0]['label']
        confidence = r[0]['score']

        return class_name, confidence

    def predict_digits(self, digits):
        """
        Digits are np arrays
        predict the digits
        """

        # Predict each digit
        predicted_digits = []
        for digit in digits:
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