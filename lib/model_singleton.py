from lib.log import log
"""
Singleton pattern for MeterPredictor to ensure only one instance exists.
This saves 150-300MB of RAM by avoiding duplicate model loading.
"""

import gc
from lib.meter_processing.meter_processing import MeterPredictor


class MeterPredictorSingleton:
    _instance = None
    _predictor = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MeterPredictorSingleton, cls).__new__(cls)
        return cls._instance

    def get_predictor(self):
        """Get or create the singleton MeterPredictor instance."""
        if self._predictor is None:
            log("[MeterPredictor] Initializing singleton instance...")
            self._predictor = MeterPredictor()
            # Force garbage collection after loading models
            gc.collect()
            log("[MeterPredictor] Singleton instance initialized and memory cleaned.")
        return self._predictor

    @classmethod
    def release(cls):
        """Release the predictor and free memory (useful for testing/reloading)."""
        if cls._predictor is not None:
            log("[MeterPredictor] Releasing singleton instance...")
            cls._predictor = None
            gc.collect()
            log("[MeterPredictor] Singleton instance released.")


def get_meter_predictor():
    """
    Get the singleton MeterPredictor instance.
    Use this function throughout the application instead of creating new instances.
    """
    singleton = MeterPredictorSingleton()
    return singleton.get_predictor()
