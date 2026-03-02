"""
Threshold Optimizer Module

Optimizes threshold values for digit extraction by maximizing model confidence.
Uses a grid search approach with optional refinement.
"""

import base64
import json
import sqlite3
from io import BytesIO
from typing import List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image

from lib.meter_processing.meter_processing import MeterPredictor


class ThresholdOptimizer:
    """
    Optimizes threshold values for meter digit extraction.

    The thresholds work as follows:
    - Main threshold (threshold_low, threshold_high): Applied to all digits except the last 3
    - Last 3 threshold (threshold_last_low, threshold_last_high): Applied only to the last 3 digits

    The optimizer finds values that maximize the confidence of the digit recognition model.
    """

    def __init__(self, meter_predictor: MeterPredictor):
        self.meter_predictor = meter_predictor

    def search_optimal_thresholds(
        self,
        colored_digits: List[str],
        islanding_padding: int = 20,
        steps: int = 10,
        digit_models: Optional[List[str]] = None
    ) -> dict:
        """
        Search for optimal threshold values using grid search.

        Args:
            colored_digits: List of base64-encoded colored digit images
            islanding_padding: Padding value for island extraction (fixed during search)
            steps: Number of steps for grid search (higher = finer search, slower)

        Returns:
            Dictionary with optimal threshold values and confidence metrics
        """
        if not colored_digits:
            return {
                "error": "No digit images provided",
                "threshold": [0, 155],
                "threshold_last": [0, 155],
                "total_confidence": 0.0,
                "avg_confidence": 0.0
            }

        # Decode images once
        digit_images = self._decode_images(colored_digits)
        if not digit_images:
            return {
                "error": "Failed to decode digit images",
                "threshold": [0, 155],
                "threshold_last": [0, 155],
                "total_confidence": 0.0,
                "avg_confidence": 0.0
            }

        # Limit steps to reasonable range
        steps = max(3, min(steps, 25))

        # Phase 1: Coarse grid search
        step_size = max(1, 255 // steps)
        threshold_values = list(range(0, 256, step_size))

        best_result = {
            "threshold": [0, 155],
            "threshold_last": [0, 155],
            "total_confidence": 0.0,
            "avg_confidence": 0.0,
            "valid_digits": 0
        }

        # Search main threshold (for non-last-3 digits)
        main_models = digit_models[:-3] if digit_models and len(digit_images) > 3 else None
        best_main = self._search_threshold_range(
            digit_images[:-3] if len(digit_images) > 3 else [],
            threshold_values,
            islanding_padding,
            model_types=main_models
        )

        # Search last-3 threshold
        last_models = digit_models[-3:] if digit_models and len(digit_images) >= 3 else digit_models
        best_last = self._search_threshold_range(
            digit_images[-3:] if len(digit_images) >= 3 else digit_images,
            threshold_values,
            islanding_padding,
            model_types=last_models
        )

        # Phase 2: Refinement around best values
        if steps >= 5:
            refined_main = self._refine_threshold(
                digit_images[:-3] if len(digit_images) > 3 else [],
                best_main["threshold"],
                islanding_padding,
                refinement_range=step_size,
                model_types=main_models
            )
            if refined_main["confidence"] > best_main["confidence"]:
                best_main = refined_main

            refined_last = self._refine_threshold(
                digit_images[-3:] if len(digit_images) >= 3 else digit_images,
                best_last["threshold"],
                islanding_padding,
                refinement_range=step_size,
                model_types=last_models
            )
            if refined_last["confidence"] > best_last["confidence"]:
                best_last = refined_last

        # Combine results and calculate overall confidence
        combined_confidence = self._evaluate_combined_thresholds(
            digit_images,
            best_main["threshold"],
            best_last["threshold"],
            islanding_padding,
            model_types=digit_models
        )

        return {
            "threshold": best_main["threshold"],
            "threshold_last": best_last["threshold"],
            "total_confidence": combined_confidence["total_confidence"],
            "avg_confidence": combined_confidence["avg_confidence"],
            "valid_digits": combined_confidence["valid_digits"],
            "main_confidence": best_main["confidence"],
            "last_confidence": best_last["confidence"]
        }

    def _decode_images(self, base64_images: List[str]) -> List[np.ndarray]:
        """Decode base64 images to numpy arrays (converts RGB from storage to BGR for processing)."""
        images = []
        for b64 in base64_images:
            try:
                image_data = base64.b64decode(b64)
                image = Image.open(BytesIO(image_data))
                img_array = np.array(image)

                # Convert RGB (from PIL/storage) to BGR for consistent OpenCV processing
                if img_array.ndim == 3 and img_array.shape[2] >= 3:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                images.append(img_array)
            except Exception as e:
                print(f"[ThresholdOptimizer] Failed to decode image: {e}")
                continue
        return images

    def _search_threshold_range(
        self,
        digit_images: List[np.ndarray],
        threshold_values: List[int],
        islanding_padding: int,
        model_types: Optional[List[str]] = None
    ) -> dict:
        """
        Search for optimal threshold in given range.

        Returns dict with best threshold and confidence.
        """
        if not digit_images:
            return {"threshold": [0, 155], "confidence": 0.0}

        best_threshold = [0, 155]
        best_confidence = 0.0

        for low in threshold_values:
            for high in threshold_values:
                # High must be greater than low with some minimum gap
                if high <= low + 10:
                    continue

                confidence = self._evaluate_threshold_on_digits(
                    digit_images, [low, high], islanding_padding, model_types=model_types
                )

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_threshold = [low, high]

        return {"threshold": best_threshold, "confidence": best_confidence}

    def _refine_threshold(
        self,
        digit_images: List[np.ndarray],
        initial_threshold: List[int],
        islanding_padding: int,
        refinement_range: int = 20,
        model_types: Optional[List[str]] = None
    ) -> dict:
        """Refine threshold around initial values with finer steps."""
        if not digit_images:
            return {"threshold": initial_threshold, "confidence": 0.0}

        best_threshold = initial_threshold.copy()
        best_confidence = self._evaluate_threshold_on_digits(
            digit_images, initial_threshold, islanding_padding, model_types=model_types
        )

        low_start = max(0, initial_threshold[0] - refinement_range)
        low_end = min(255, initial_threshold[0] + refinement_range)
        high_start = max(0, initial_threshold[1] - refinement_range)
        high_end = min(255, initial_threshold[1] + refinement_range)

        step = max(1, refinement_range // 5)

        for low in range(low_start, low_end + 1, step):
            for high in range(high_start, high_end + 1, step):
                if high <= low + 10:
                    continue

                confidence = self._evaluate_threshold_on_digits(
                    digit_images, [low, high], islanding_padding, model_types=model_types
                )

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_threshold = [low, high]

        return {"threshold": best_threshold, "confidence": best_confidence}

    def _evaluate_threshold_on_digits(
        self,
        digit_images: List[np.ndarray],
        threshold: List[int],
        islanding_padding: int,
        model_types: Optional[List[str]] = None
    ) -> float:
        """
        Evaluate a threshold combination on a set of digit images.

        Returns average confidence for recognized digits (excluding 'r' predictions).
        """
        if not digit_images:
            return 0.0

        total_confidence = 0.0
        valid_count = 0

        for idx, digit_img in enumerate(digit_images):
            try:
                _, processed_digit = self.meter_predictor.apply_threshold(
                    digit_img,
                    threshold[0],
                    threshold[1],
                    islanding_padding,
                    invert=False
                )

                # Get prediction with confidence
                model_type = None
                if model_types and idx < len(model_types):
                    model_type = model_types[idx]
                predictions = self.meter_predictor.predict_digit(processed_digit, model_type=model_type)

                if predictions:
                    top_prediction, top_confidence = predictions[0]

                    # Only count valid digit predictions (not 'r' = rejected)
                    if top_prediction != 'r':
                        total_confidence += top_confidence
                        valid_count += 1
                    else:
                        # Penalize rejected predictions slightly
                        total_confidence += top_confidence * 0.3
                        valid_count += 1

            except Exception as e:
                # Skip on error
                continue

        return total_confidence / max(valid_count, 1)

    def _evaluate_combined_thresholds(
        self,
        digit_images: List[np.ndarray],
        main_threshold: List[int],
        last_threshold: List[int],
        islanding_padding: int,
        model_types: Optional[List[str]] = None
    ) -> dict:
        """
        Evaluate combined thresholds on all digits.

        Applies main_threshold to all digits except last 3,
        and last_threshold to the last 3 digits.
        """
        if not digit_images:
            return {"total_confidence": 0.0, "avg_confidence": 0.0, "valid_digits": 0}

        total_confidence = 0.0
        valid_count = 0
        num_digits = len(digit_images)

        for i, digit_img in enumerate(digit_images):
            # Use appropriate threshold based on position
            is_last_3 = i >= num_digits - 3
            current_threshold = last_threshold if is_last_3 else main_threshold

            try:
                _, processed_digit = self.meter_predictor.apply_threshold(
                    digit_img,
                    current_threshold[0],
                    current_threshold[1],
                    islanding_padding,
                    invert=False
                )

                model_type = None
                if model_types and i < len(model_types):
                    model_type = model_types[i]
                predictions = self.meter_predictor.predict_digit(processed_digit, model_type=model_type)

                if predictions:
                    top_prediction, top_confidence = predictions[0]
                    if top_prediction != 'r':
                        total_confidence += top_confidence
                        valid_count += 1

            except Exception:
                continue

        return {
            "total_confidence": total_confidence,
            "avg_confidence": total_confidence / max(valid_count, 1),
            "valid_digits": valid_count
        }


def search_thresholds_for_meter(
    db_file: str,
    name: str,
    meter_predictor: MeterPredictor,
    steps: int = 10
) -> dict:
    """
    Search for optimal thresholds for a specific water meter.

    Args:
        db_file: Path to the SQLite database
        name: Name of the water meter
        meter_predictor: MeterPredictor instance
        steps: Search depth (3-25, higher = finer search)

    Returns:
        Dictionary with optimal thresholds and confidence metrics
    """
    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()

        # Get the latest evaluation with colored digits
        cursor.execute(
            """
            SELECT colored_digits, id FROM evaluations 
            WHERE name = ? AND colored_digits IS NOT NULL
            ORDER BY id DESC 
            LIMIT 1
            """,
            (name,)
        )
        row = cursor.fetchone()

        if not row:
            return {
                "error": "No evaluation found for this meter",
                "threshold": [0, 155],
                "threshold_last": [0, 155]
            }

        colored_digits = json.loads(row[0])

        # Get current islanding_padding from settings
        cursor.execute(
            "SELECT islanding_padding, digit_models FROM settings WHERE name = ?",
            (name,)
        )
        settings_row = cursor.fetchone()
        islanding_padding = settings_row[0] if settings_row else 20
        digit_models = None
        if settings_row and len(settings_row) > 1 and settings_row[1]:
            try:
                digit_models = json.loads(settings_row[1])
            except Exception:
                digit_models = None

        # Run optimization
        optimizer = ThresholdOptimizer(meter_predictor)
        result = optimizer.search_optimal_thresholds(
            colored_digits,
            islanding_padding=islanding_padding,
            steps=steps,
            digit_models=digit_models
        )

        return result
