import numpy as np


class EyeFeatureExtractor:
    """
    Analyzes eye landmarks to calculate aspect ratios, curvature, and symmetry.
    """

    def __init__(self, normalized_np):
        """
        Initializes the extractor with normalized face landmarks.

        Args:
            normalized_np (np.array): Array of shape (N, 2) containing x, y coordinates.
        """
        self.np = normalized_np

        # Define landmark indices for left and right eyes
        self.left_indices = {'outer': 33, 'inner': 133, 'top': 159, 'bottom': 145}
        self.right_indices = {'outer': 263, 'inner': 362, 'top': 386, 'bottom': 374}

        # Calculate interocular distance for normalization
        p1 = self.np[self.left_indices['outer']]
        p2 = self.np[self.right_indices['outer']]
        self.interocular = np.linalg.norm(p1 - p2)

    def get_eye_dimensions(self, idx_dict):
        """Calculates width and height of a single eye."""
        width = np.linalg.norm(self.np[idx_dict['outer']] - self.np[idx_dict['inner']])
        height = np.linalg.norm(self.np[idx_dict['top']] - self.np[idx_dict['bottom']])
        return width, height

    def get_eye_curvature(self, idx_dict):
        """Calculates the curvature of the upper eyelid."""
        outer = idx_dict['outer']
        inner = idx_dict['inner']
        top = idx_dict['top']

        # Center point between corners
        eye_center = (self.np[outer] + self.np[inner]) / 2
        # Vector from center to top eyelid
        curvature_vector = self.np[top] - eye_center
        return np.linalg.norm(curvature_vector)

    def extract_metrics(self):
        """
        return:
            eye_aspect_ratio: capture eye openness; higher the rounder eye, the lower the slender eyes
            eye_curvature_ratio: reflects eyelid arch heigh
            symmetry_confidence: higher = more symmetry
        """

        # Get Dimensions
        left_w, left_h = self.get_eye_dimensions(self.left_indices)
        right_w, right_h = self.get_eye_dimensions(self.right_indices)

        # Get Curvature
        left_curv = self.get_eye_curvature(self.left_indices)
        right_curv = self.get_eye_curvature(self.right_indices)

        # Prevent division by zero
        if left_w == 0: left_w = 0.001
        if right_w == 0: right_w = 0.001

        # 1. Aspect Ratio (Openness)
        eye_aspect_ratio = np.mean([left_h / left_w, right_h / right_w])

        # 2. Curvature Ratio
        if self.interocular == 0: self.interocular = 0.001
        eye_curvature_ratio = np.mean([left_curv / self.interocular, right_curv / self.interocular])

        # 3. Symmetry Score
        # Avoid division by zero in symmetry calculation
        avg_w = (left_w + right_w) / 2
        avg_h = (left_h + right_h) / 2
        avg_curv = (left_curv + right_curv) / 2

        width_sym = abs(left_w - right_w) / avg_w if avg_w > 0 else 0
        height_sym = abs(left_h - right_h) / avg_h if avg_h > 0 else 0
        curv_sym = abs(left_curv - right_curv) / avg_curv if avg_curv > 0 else 0

        symmetry_score = 0.3 * width_sym + 0.3 * height_sym + 0.4 * curv_sym
        symmetry_confidence = 1 - symmetry_score

        return {
            "eye_aspect_ratio": eye_aspect_ratio,
            "eye_curvature_ratio": eye_curvature_ratio,
            "eye_symmetry": symmetry_confidence
        }