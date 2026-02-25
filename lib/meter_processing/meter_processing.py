import base64
import gc
import os
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
import onnxruntime as ort

from lib.meter_processing.roi_extractors import YOLOExtractor, BypassExtractor

class MeterPredictor:
    """
    A class to perform water meter digit detection (using YOLO with OBB)
    and digit classification
    """

    def __init__(self):
        """
        Initializes the ONNX inference sessions for YOLO and digit classifier.
        Optimized for minimal memory usage - uses ~70% less RAM than TensorFlow+PyTorch.
        """
        print("[MeterPredictor] Loading ONNX models...")

        # Configure ONNX Runtime for minimal memory usage
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_mem_pattern = False  # Reduce memory fragmentation
        sess_options.enable_cpu_mem_arena = False  # Reduce memory overhead

        # Load YOLO ONNX model for oriented bounding box detection
        self.yolo_session = ort.InferenceSession(
            "models/yolo-best-obb-2.onnx",
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )

        # Load digit classifier ONNX model
        self.digit_session = ort.InferenceSession(
            'models/best_model.onnx',
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )

        self.class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'r']
        self.last_error = None

        # Get input/output names for both models
        self.yolo_input_name = self.yolo_session.get_inputs()[0].name
        self.yolo_output_names = [output.name for output in self.yolo_session.get_outputs()]

        self.digit_input_name = self.digit_session.get_inputs()[0].name
        self.digit_output_name = self.digit_session.get_outputs()[0].name

        # Force garbage collection after loading models
        gc.collect()
        print("[MeterPredictor] ONNX models loaded successfully with minimal memory footprint.")
        print(f"[MeterPredictor] YOLO input: {self.yolo_input_name}")
        print(f"[MeterPredictor] Digit classifier input: {self.digit_input_name}")

    def extract_display_and_segment(self, input_image, segments=7, rotated_180=False, extended_last_digit=False, shrink_last_3=False, target_brightness=None, roi_extractor="yolo", extractor_instance=None, segment_mode="display"):
        """
        Predicts the water meter reading on a single image:
          - Runs YOLO detection for oriented bounding box (OBB)
          - Applies perspective transform to 'straighten' the meter
          - Splits the meter into vertical segments

        Args:
            input_image (PIL.Image): The input image to process.
            segments (int): The number of segments to split the meter into.
            rotated_180 (bool): Whether to rotate the meter 180 degrees.
            extended_last_digit (bool): Whether to extend the last digit for better classification.
            shrink_last_3 (bool): Whether to shrink the last 3 digits for better classification.
            target_brightness (float): The target brightness to adjust the image to.
        """

        self.last_error = None
        use_templated_extractor = extractor_instance is not None

        if use_templated_extractor:
            if isinstance(input_image, Image.Image):
                input_image = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
        elif roi_extractor == "yolo" and rotated_180:
            input_image = input_image.rotate(180, expand=True)

        if extractor_instance is not None:
            extractor = extractor_instance
        elif roi_extractor == "bypass":
            extractor = BypassExtractor()
        else:
            extractor = YOLOExtractor(self.yolo_session, self.yolo_input_name, extended_last_digit=extended_last_digit)
        digits, boundingboxed_image = extractor.extract_segments(
            input_image,
            segments=segments,
            extended_last_digit=extended_last_digit,
            shrink_last_3=shrink_last_3,
            segment_mode=segment_mode,
        )
        if not digits or len(digits) == 0:
            self.last_error = getattr(extractor, "last_error", None) or "No result found"
            return [], [], None, None

        if use_templated_extractor and rotated_180:
            digits = [cv2.rotate(digit, cv2.ROTATE_180) for digit in digits]

        # Adjust brightness of each image
        mean_brightnesses = [np.mean(img) for img in digits]
        adjusted_images = []
        if target_brightness is None:
            target_brightness = np.mean(mean_brightnesses)
        for img, mean_brightness in zip(digits, mean_brightnesses):
            adjustment_factor = target_brightness / mean_brightness
            adjusted_img = np.clip(img * adjustment_factor, 0, 255).astype(np.uint8)
            adjusted_images.append(adjusted_img)

        digits = adjusted_images

        # Convert to base64 for temporary storage
        base64s = []
        for part in digits:
            # Store cutouts as RGB to avoid BGR/RGB channel confusion.
            if len(part.shape) == 3:
                part = cv2.cvtColor(part, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(part)

            buffered = BytesIO()
            pil_img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue())
            # to string
            img_str = img_str.decode('utf-8')

            base64s.append(img_str)

        return base64s, digits, target_brightness, boundingboxed_image

    def apply_threshold(self, digit, threshold_low, threshold_high, islanding_padding=40, invert=False):
        threshold_low, threshold_high = int(threshold_low), int(threshold_high)
        islanding_padding = int(islanding_padding)

        # Convert the digit image to grayscale if needed
        if len(digit.shape) == 3:
            digit = cv2.cvtColor(digit, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to get a binary image.
        digit = cv2.inRange(digit, threshold_low, threshold_high)

        # Find connected regions, by default the background needs to be black (0) and the digits white (255).
        # Invert the image to match this requirement.
        inverted = cv2.bitwise_not(digit)

        # Find connected components (8-connectivity by default)
        num_labels, labels = cv2.connectedComponents(inverted)

        # Create a BGR color image with white background
        color_image = np.full((*digit.shape, 3), (255, 255, 255), dtype=np.uint8)

        # Get the dimensions of the image
        height, width = digit.shape

        # Calculate the middle x% region (with islanding_padding% padding on all sides)
        start_x = int((islanding_padding / 100.0) * width)
        end_x = int(1.0 - (islanding_padding / 100.0)  * width)
        start_y = int((islanding_padding / 100.0) * height)
        end_y = int(1.0 - (islanding_padding / 100.0) * height)

        extracted = 0
        extracted_percentage = 0
        for label in range(1, num_labels):
            # Slice the labels to the middle region and check for any occurrence of the current label
            component_region = labels[start_y:end_y, start_x:end_x]
            in_middle = np.any(component_region == label)

            if in_middle:
                color = (0, 0, 0)
                extracted += 1

                # Calculate the percentage of the component in the middle region
                component_area = np.sum(labels == label)
                total_area = height * width
                extracted_percentage += component_area / total_area * 100
            else:
                # remove the component if it is not in the middle region
                color = (255, 255, 255)

            color_image[labels == label] = color

        # if no components are in the middle region or less than 10% of the image is extracted, use the whole image
        if extracted == 0 or extracted_percentage < 10:
            # use the whole image
            color_image = np.full((*digit.shape, 3), (255, 255, 255), dtype=np.uint8)
            color_image[labels != 0] = (0, 0, 0)

        # back to greyscale
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        digit = cv2.resize(color_image, (40, 64))

        # --- Normalize & add extra dimensions ---
        img_norm = digit.astype('float32') / 255.0
        img_norm = np.expand_dims(img_norm, axis=-1)  # add channel dimension
        img_norm = np.expand_dims(img_norm, axis=0)  # add batch dimension

        img_uint8 = (img_norm.squeeze() * 255).astype(np.uint8)  # Remove extra dims & convert to uint8
        pil_img = Image.fromarray(img_uint8)

        # Encode image to Base64
        buffered = BytesIO()
        if invert:
            pil_img = Image.fromarray(255 - img_uint8)  # Invert for
        pil_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return img_str, img_norm

    # use the classifier to predict the digit, returns the top 3 predictions with their confidence
    def predict_digit(self, digit):
        # Perform prediction using ONNX model
        predictions = self.digit_session.run(
            [self.digit_output_name],
            {self.digit_input_name: digit}
        )[0]

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

        # Clean up memory after batch prediction
        gc.collect()

        return predicted_digits

    def apply_thresholds(self, digits, thresholds, thresholds_last, islanding_padding):
        """
        Digits are np arrays
        apply black/white thresholding to each digit
        """

        # Apply thresholding
        thresholded_digits = []
        base64s = []
        base64s_inverted = []

        threshold_low = thresholds[0]
        threshold_high = thresholds[1]
        for i, digit in enumerate(digits):
            if i >= len(digits) - 3:
                threshold_low = thresholds_last[0]
                threshold_high = thresholds_last[1]
            img_str, digit = self.apply_threshold(digit, threshold_low, threshold_high, islanding_padding)

            thresholded_digits.append(digit)
            base64s.append(img_str)

            # also store inverted images as base64 for debugging
            img_uint8 = (digit.squeeze() * 255).astype(np.uint8)
            pil_img = Image.fromarray(255 - img_uint8)  # Invert for
            buffered = BytesIO()
            pil_img.save(buffered, format="PNG")
            img_str_inverted = base64.b64encode(buffered.getvalue()).decode('utf-8')
            base64s_inverted.append(img_str_inverted)

        return base64s, thresholded_digits, base64s_inverted
