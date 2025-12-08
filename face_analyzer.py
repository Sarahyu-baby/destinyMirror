import cv2
import numpy as np
import mediapipe as mp
from eye_feature_extractor import EyeFeatureExtractor


class FaceAnalyzer:
    """
    Main analysis class using MediaPipe Face Mesh to extract facial features.
    """

    def __init__(self):
        """
        Initializes MediaPipe FaceMesh.
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True
        )
        self.landmarks_np = None
        self.custom_points = None

    def process_image(self, img):
        """
        Main pipeline: Detect -> Measure -> Calculate Ratios -> Return Data
        """
        if img is None: return None

        # 1. Detection
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)

        if not results.multi_face_landmarks:
            self.landmarks_np = None
            self.custom_points = None
            return None

        # 2. Convert original MediaPipe landmark object to normalized np.array for easy math.
        face = results.multi_face_landmarks[0] #The raw LandmarkList object from MediaPipe, first faces detected
        self.landmarks_np = np.array([(lm.x, lm.y) for lm in face.landmark]) # normalized

        # 3. Calculate Measurements
        measurements, custom_points = self._measure_face(self.landmarks_np)

        # 4. Save Custom Points (Vertex, Brow Mid) to self so we can access them later
        self.custom_points = custom_points

        # 5. Calculate Ratios
        features = self._calculate_ratios(measurements)

        # 6. Add Eye Features
        eye_extractor = EyeFeatureExtractor(self.landmarks_np)
        eye_features = eye_extractor.extract_metrics()

        # Merge everything
        if eye_features:
            features.update(eye_features)

        # Round all values for clean output
        return {k: round(float(v), 3) for k, v in features.items()}

    def _measure_face(self, lms):
        """
        Internal/Private function only works inside class
        Calculates the raw physical distances/angles needed for the ratios.
        """
        # Helper for cleaner code
        pts = {
            'chin': lms[152], 'chin_l': lms[176], 'chin_r': lms[400],
            'nose_tip': lms[1], 'nose_bridge': lms[168],
            'nose_l': lms[48], 'nose_r': lms[331],
            'brow_mid': (lms[70] + lms[300]) / 2,
            # Captures the inner ends of the eyebrows, close to the glabella (between the eyes)
            'brow_center': (lms[105] + lms[334]) / 2,  # represents the center of each eyebrow, closer to the arch
            'face_l': lms[234], 'face_r': lms[454],
            'jaw_l_pt': lms[172], 'jaw_r_pt': lms[288],  # Points for angle calc
            'mouth_l': lms[61], 'mouth_r': lms[291],
            'mouth_l_top': lms[37], 'mouth_l_mid': lms[82], 'mouth_l_bottom': lms[84],
            'mouth_r_top': lms[267], 'mouth_r_mid': lms[312], 'mouth_r_bottom': lms[314],
            'top_forehead': lms[10],
            'eye_l_in': lms[133], 'eye_r_in': lms[362]
        }

        def dist(p1, p2): return np.linalg.norm(p1 - p2)  # Calculate the distance between two points

        def angle(p1, p2, p3):
            """
            Calculates angle at vertex p2.
            Vectors created: p2->p1 and p2->p3
            return: angle\theta
            """
            v1 = p1 - p2  # Vector from p2 to p1
            v2 = p3 - p2  # Vector from p2 to p3
            dot = np.dot(v1, v2)
            norm = np.linalg.norm(v1) * np.linalg.norm(v2)

            if norm == 0: return 0.0  # Safety check
            # Clamp value to avoid numerical errors outside [-1, 1]
            cos_theta = np.clip(dot / norm, -1.0, 1.0)
            theta_rad = np.arccos(cos_theta)
            theta_deg = np.degrees(theta_rad)
            return theta_deg

        # Forehead vertex projection
        forehead_scale = 1.7
        forehead_h = forehead_scale * dist(pts['top_forehead'], pts['brow_mid'])
        vertex_point = pts['top_forehead'] + (forehead_scale - 1) * (pts['top_forehead'] - pts['brow_mid'])

        # Pack the measurements for math
        measurements = {
            'face_width': dist(pts['face_l'], pts['face_r']),
            'face_length': dist(pts['brow_mid'], pts['chin']) + forehead_h,
            'forehead_height': forehead_h,
            'midface_height': dist(pts['nose_tip'], pts['brow_center']),
            'lower_face_height': dist(pts['nose_tip'], pts['chin']),
            'eye_distance': dist(pts['eye_l_in'], pts['eye_r_in']),
            'nose_width': dist(pts['nose_l'], pts['nose_r']),
            'nose_length': dist(pts['nose_bridge'], pts['nose_tip']),
            'mouth_width': dist(pts['mouth_l'], pts['mouth_r']),
            'chin_width': dist(pts['chin_l'], pts['chin_r']),
            'upper_lip_h': (dist(pts['mouth_l_top'], pts['mouth_l_mid']) + dist(pts['mouth_r_top'],
                                                                                pts['mouth_r_mid'])) / 2,
            'lower_lip_h': (dist(pts['mouth_l_bottom'], pts['mouth_l_mid']) + dist(pts['mouth_r_bottom'],
                                                                                   pts['mouth_r_mid'])) / 2,
            'jaw_angle_left': angle(pts['face_l'], pts['jaw_l_pt'], pts['chin_l']),
            'jaw_angle_right': angle(pts['face_r'], pts['jaw_r_pt'], pts['chin_r']),
        }
        # Pack the coordinates for visualization
        custom_points = {
            "vertex": vertex_point,
            "brow_mid": pts['brow_mid']
        }

        # RETURN BOTH
        return measurements, custom_points

    def _calculate_ratios(self, m):
        """
        Internal/Private function only works inside class
        Applies ratio logic using the measurements (m).
        """
        # Safety check for zero division
        if m['face_length'] == 0: return {}

        jaw_avg = (m['jaw_angle_left'] + m['jaw_angle_right']) / 2

        return {
            "face_lw_ratio": m['face_width'] / m['face_length'],
            "forehead_ratio": m['forehead_height'] / m['face_length'],
            "midface_ratio": m['midface_height'] / m['face_length'],
            "lowerface_ratio": m['lower_face_height'] / m['face_length'],
            "eye_distance_ratio": m['eye_distance'] / m['face_width'],
            # Added safety checks for denominators
            "nose_ratio": m['nose_width'] / m['nose_length'] if m['nose_length'] > 0 else 0,
            "mouth_chin_ratio": m['mouth_width'] / m['chin_width'] if m['chin_width'] > 0 else 0,
            "jaw_angle": jaw_avg,
            "upper_lip_ratio": m['upper_lip_h'] / m['mouth_width'] if m['mouth_width'] > 0 else 0,
            "lower_lip_ratio": m['lower_lip_h'] / m['mouth_width'] if m['mouth_width'] > 0 else 0
        }
