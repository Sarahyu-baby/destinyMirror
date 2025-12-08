import cv2
import numpy as np
import matplotlib.pyplot as plt


class FaceVisualizer:
    """
    Handles the visualization of landmarks and custom points on the image using OpenCV.
    """

    def __init__(self):
        """
        Initializes the visualizer with defined landmark groups and colors.
        """
        # Define landmark groups (indices based on MediaPipe FaceMesh)
        self.groups = {
            "top_forehead": [10],
            "nose": [1, 48, 168, 331],
            "chin": [152, 176, 400],
            "brows": [70, 300, 105, 334],
            "eyes": [33, 133, 145, 159, 263, 362, 386, 374],
            "mouth": [37, 61, 82, 84, 267, 291, 312, 314],
            "face_width": [172, 397, 234, 454]
        }

        # Define colors (BGR format for OpenCV)
        self.colors = {
            "top_forehead": (0, 255, 255),  # Yellow
            "nose": (0, 0, 255),  # Red
            "chin": (255, 0, 0),  # Blue
            "brows": (0, 255, 0),  # Green
            "eyes": (255, 255, 0),  # Cyan
            "mouth": (255, 0, 255),  # Magenta
            "face_width": (128, 128, 128)  # Gray
        }

        # Drawing settings
        self.RADIUS_SMALL = 2
        self.RADIUS_LARGE = 5
        self.COLOR_LANDMARK = (0, 255, 0)  # Green
        self.COLOR_VERTEX = (0, 0, 255)  # Red
        self.COLOR_BROW_MID = (255, 0, 0)  # Blue

    def draw_landmarks(self, image, landmarks_np):
        """
        Draws the defined landmark groups on the image.

        Args:
            image (np.array): The original OpenCV image (BGR).
            landmarks_np (np.array): The normalized (N,2) array of landmarks.

        Returns:
            np.array: Image with drawn landmarks.
        """
        if landmarks_np is None: return image

        img_copy = image.copy()
        h, w = img_copy.shape[:2]

        # Convert normalized coordinates -> pixel coordinates
        pixel_landmarks = (landmarks_np * [w, h]).astype(int)

        for group, indices in self.groups.items():
            color = self.colors.get(group, (255, 255, 255))

            for idx in indices:
                # Safety check to ensure index is within bounds
                if idx < len(pixel_landmarks):
                    x, y = pixel_landmarks[idx]
                    # Draw Dot
                    cv2.circle(img_copy, (x, y), 3, color, -1)
                    # Draw Text (Index number) - Scaled for mobile visibility
                    cv2.putText(img_copy, str(idx), (x + 2, y - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        return img_copy

    def draw_custom_points(self, image, points_dict):
        """
        Draws specific calculated points (like 'vertex') with text labels.

        Args:
            image (np.array): OpenCV image.
            points_dict (dict): Dictionary of {'name': [x, y]} normalized coordinates.

        Returns:
            np.array: Image with custom points drawn.
        """
        if not points_dict: return image

        img_copy = image.copy()
        h, w = img_copy.shape[:2]

        for label, point in points_dict.items():
            # Determine color based on label key
            color = self.COLOR_VERTEX if label == 'vertex' else self.COLOR_BROW_MID

            # Convert Normalized -> Pixel
            x = int(point[0] * w)
            y = int(point[1] * h)

            # Draw Point
            cv2.circle(img_copy, (x, y), self.RADIUS_LARGE, color, -1)

            # Draw Label Text
            cv2.putText(img_copy, label, (x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return img_copy
