import math

import cv2
import mediapipe as mp
import numpy as np


class GestureState:
    def __init__(self):
        self.prev_grabbing = False
        self.current_grabbing = False
        self.grab_started = False
        self.grab_ended = False

    def update(self, grabbing):
        self.current_grabbing = grabbing
        self.grab_started = grabbing and not self.prev_grabbing
        self.grab_ended = self.prev_grabbing and not grabbing
        self.prev_grabbing = grabbing


class VisionEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        self.aruco_available = hasattr(cv2, "aruco")

        self.hands = self.mp_hands.Hands(
            max_num_hands=cfg.MAX_NUM_HANDS,
            min_detection_confidence=cfg.HAND_DET_CONF,
            min_tracking_confidence=cfg.HAND_TRACK_CONF,
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=cfg.MAX_NUM_FACES,
            min_detection_confidence=cfg.FACE_DET_CONF,
            min_tracking_confidence=cfg.FACE_TRACK_CONF,
        )

        self.aruco_detector = None
        self.aruco_params = None
        self.aruco_dict = None
        if self.aruco_available:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cfg.ARUCO_DICT)
            self.aruco_params = cv2.aruco.DetectorParameters()
            if hasattr(cv2.aruco, "ArucoDetector"):
                self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

    def close(self):
        self.hands.close()
        self.face_mesh.close()

    def process_face(self, rgb_frame, curr_ox, curr_oy):
        res = self.face_mesh.process(rgb_frame)
        if res.multi_face_landmarks:
            pt = res.multi_face_landmarks[0].landmark[168]
            tx = (pt.x * 2.0 - 1.0) * self.cfg.P_SENSITIVITY
            ty = (pt.y * 2.0 - 1.0) * self.cfg.P_SENSITIVITY
            return (
                curr_ox * (1.0 - self.cfg.FACE_SMOOTH) + tx * self.cfg.FACE_SMOOTH,
                curr_oy * (1.0 - self.cfg.FACE_SMOOTH) + ty * self.cfg.FACE_SMOOTH,
            )
        return curr_ox, curr_oy

    def process_hands(self, rgb_frame, w, h):
        res = self.hands.process(rgb_frame)
        data = []
        if res.multi_hand_landmarks:
            for lms in res.multi_hand_landmarks:
                idx, thumb = lms.landmark[8], lms.landmark[4]
                wrist, mid_base = lms.landmark[0], lms.landmark[9]

                ix, iy = int(idx.x * w), int(idx.y * h)
                tx, ty = int(thumb.x * w), int(thumb.y * h)

                grabbing = math.hypot(tx - ix, ty - iy) < self.cfg.PINCH_THRESHOLD
                span = math.hypot(wrist.x - mid_base.x, wrist.y - mid_base.y)
                z_val = max(
                    self.cfg.DEPTH_MIN,
                    min(self.cfg.DEPTH_MAX, span * self.cfg.DEPTH_SCALE + self.cfg.DEPTH_OFFSET),
                )

                data.append(
                    {
                        "px": (ix, iy),
                        "pos3d": [idx.x * 2.0 - 1.0, -(idx.y * 2.0 - 1.0), z_val],
                        "grabbing": grabbing,
                    }
                )
        return data

    def process_aruco(self, bgr_frame, camera_matrix, dist_coeffs):
        if not self.aruco_available or camera_matrix is None:
            return None

        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        if self.aruco_detector is not None:
            corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is None or len(ids) == 0:
            return None

        ids_flat = ids.flatten()
        target_id = self.cfg.ARUCO_TARGET_ID
        target_idx = 0
        if target_id >= 0:
            matches = np.where(ids_flat == target_id)[0]
            if len(matches) == 0:
                return None
            target_idx = int(matches[0])

        one_corner = [corners[target_idx]]
        marker_id = int(ids_flat[target_idx])
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            one_corner,
            self.cfg.ARUCO_MARKER_LENGTH_M,
            camera_matrix,
            dist_coeffs,
        )

        if rvecs is None or len(rvecs) == 0:
            return None

        return {
            "id": marker_id,
            "corners": one_corner[0],
            "rvec": rvecs[0][0].astype(np.float32),
            "tvec": tvecs[0][0].astype(np.float32),
        }
