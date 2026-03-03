from abc import ABC, abstractmethod
import base64
import json
import numpy as np
import cv2


class ROIExtractor(ABC):
    @abstractmethod
    def extract(self, input_image):
        """Return (cropped, cropped_ext, boundingboxed_image) or (None, None, None) on failure."""
        raise NotImplementedError

    def extract_segments(self, input_image, segments, extended_last_digit=False, shrink_last_3=False, segment_mode="display"):
        """
        Extract and segment digits from input image.

        Default implementation extracts the full display ROI and then splits
        it into segments (display mode). Subclasses can override to provide
        custom per-digit extraction.
        """
        if segment_mode not in {"display", "each_digit"}:
            segment_mode = "display"

        cropped, cropped_ext, boundingboxed_image = self.extract(input_image)
        if cropped is None:
            return [], boundingboxed_image

        digits = self.segment_display(
            cropped,
            cropped_ext,
            segments=segments,
            extended_last_digit=extended_last_digit,
            shrink_last_3=shrink_last_3,
        )
        return digits, boundingboxed_image

    @staticmethod
    def segment_display(cropped, cropped_ext, segments, extended_last_digit=False, shrink_last_3=False):
        """Split a cropped display image into digit segments."""
        if segments < 2:
            return []
        if cropped is None or cropped.shape[1] <= 0:
            return []

        part_width = cropped.shape[1] // segments
        if part_width <= 0:
            return []

        digits = []
        last_x = 0
        for i in range(segments):
            if shrink_last_3 and i >= segments - 3:
                t_part_width = int(part_width * 0.8)
            elif shrink_last_3:
                t_part_width = int(((part_width * segments) - (3 * part_width * 0.8)) / (segments - 3))
            else:
                t_part_width = part_width

            part = cropped[:, last_x:last_x + t_part_width]

            if extended_last_digit and i == segments - 1 and cropped_ext is not None:
                ext_end_x = cropped_ext.shape[1]
                ext_start_x = max(ext_end_x - t_part_width, 0)
                part = cropped_ext[:, ext_start_x:ext_end_x]

            last_x = last_x + t_part_width
            digits.append(part)

        return digits

    @staticmethod
    def estimate_quad_size(corners):
        """Estimate target size from 4 corner points."""
        width_a = np.linalg.norm(corners[0] - corners[1])
        width_b = np.linalg.norm(corners[2] - corners[3])
        height_a = np.linalg.norm(corners[0] - corners[3])
        height_b = np.linalg.norm(corners[1] - corners[2])
        target_width = int(round(max(width_a, width_b)))
        target_height = int(round(max(height_a, height_b)))
        if target_width <= 0 or target_height <= 0:
            return 0, 0
        return target_width, target_height

    @staticmethod
    def warp_quad(image, corners, target_width=None, target_height=None):
        """Warp a quadrilateral into a rectangular image."""
        if target_width is None or target_height is None:
            target_width, target_height = ROIExtractor.estimate_quad_size(corners)
        if target_width <= 0 or target_height <= 0:
            return None
        dst_corners = np.array([
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1]
        ], dtype=np.float32)
        M = cv2.getPerspectiveTransform(corners, dst_corners)
        return cv2.warpPerspective(image, M, (target_width, target_height))


class ROIExtractorTemplated(ROIExtractor):
    """
    Abstract base class for template-based ROI extractors.

    Handles serialization/deserialization of reference images and precomputed data.
    Subclasses must implement feature extraction and matching logic.
    """

    def __init__(self, reference_image, config_dict):
        """
        Initialize templated extractor.

        Args:
            reference_image: Reference image (numpy array)
            config_dict: Configuration dictionary (will be stored as JSON in DB)
        """
        self.reference_image = reference_image
        self.config = config_dict

        # Convert to grayscale if needed
        if len(reference_image.shape) == 3:
            self.reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
        else:
            self.reference_gray = reference_image

    @abstractmethod
    def compute_precomputed_data(self):
        """
        Compute and return precomputed data for caching (e.g., features, masks).

        Returns:
            dict: Dictionary with precomputed data (will be serialized to base64)
        """
        raise NotImplementedError

    @abstractmethod
    def load_precomputed_data(self, precomputed_dict):
        """
        Load precomputed data from cache.

        Args:
            precomputed_dict: Dictionary with precomputed data (deserialized from base64)
        """
        raise NotImplementedError

    def serialize_template(self):
        """
        Serialize template to database format.

        Returns:
            tuple: (reference_image_base64, config_json, precomputed_data_base64)
        """
        # Encode reference image
        _, buffer = cv2.imencode('.jpg', self.reference_image)
        ref_img_b64 = base64.b64encode(buffer).decode('utf-8')

        # Config to JSON
        config_json = json.dumps(self.config)

        # Compute and encode precomputed data
        precomputed = self.compute_precomputed_data()
        precomputed_json = json.dumps(precomputed, cls=NumpyEncoder)
        precomputed_b64 = base64.b64encode(precomputed_json.encode('utf-8')).decode('utf-8')

        return ref_img_b64, config_json, precomputed_b64

    @classmethod
    def deserialize_template(cls, reference_image_base64, config_json, precomputed_data_base64=None):
        """
        Deserialize template from database format.

        Args:
            reference_image_base64: Base64 encoded reference image
            config_json: JSON string with configuration
            precomputed_data_base64: Optional base64 encoded precomputed data

        Returns:
            Instance of the extractor class
        """
        # Decode reference image
        img_bytes = base64.b64decode(reference_image_base64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        reference_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Parse config
        config_dict = json.loads(config_json)

        # Create instance
        instance = cls(reference_image, config_dict)

        # Load precomputed data if available
        if precomputed_data_base64:
            precomputed_json = base64.b64decode(precomputed_data_base64).decode('utf-8')
            precomputed_dict = json.loads(precomputed_json, object_hook=numpy_decoder)
            instance.load_precomputed_data(precomputed_dict)

        return instance


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for numpy arrays and types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {
                '__numpy__': True,
                'dtype': str(obj.dtype),
                'shape': obj.shape,
                'data': base64.b64encode(obj.tobytes()).decode('utf-8')
            }
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, cv2.KeyPoint):
            return {
                '__keypoint__': True,
                'pt': obj.pt,
                'size': obj.size,
                'angle': obj.angle,
                'response': obj.response,
                'octave': obj.octave,
                'class_id': obj.class_id
            }
        return super().default(obj)


def numpy_decoder(obj):
    """JSON decoder for numpy arrays and types."""
    if isinstance(obj, dict):
        if obj.get('__numpy__'):
            data = base64.b64decode(obj['data'])
            arr = np.frombuffer(data, dtype=obj['dtype'])
            return arr.reshape(obj['shape'])
        if obj.get('__keypoint__'):
            return cv2.KeyPoint(
                x=obj['pt'][0],
                y=obj['pt'][1],
                size=obj['size'],
                angle=obj['angle'],
                response=obj['response'],
                octave=obj['octave'],
                class_id=obj['class_id']
            )
    return obj
