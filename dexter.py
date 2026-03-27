import sys
import cv2
import time
import math
import numpy as np
from datetime import datetime

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QPushButton,
    QFrame,
    QSizePolicy,
    QTextEdit,
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from vision import VisionEngine, GestureState
from holograms import HologramEngine, rot_matrix_x, rot_matrix_y, smooth_vec3, clamp, project
from ui import UIOverlay, point_in_rect


class Config:
    WINDOW_NAME = "DEXTER v19.0 // ARUCO WORLD ANCHOR ONLINE"
    CAMERA_INDEX = 0

    MAX_NUM_HANDS, HAND_DET_CONF, HAND_TRACK_CONF = 2, 0.7, 0.7
    MAX_NUM_FACES, FACE_DET_CONF, FACE_TRACK_CONF = 1, 0.7, 0.7
    FACE_SMOOTH, HAND_SMOOTH, OBJ_DRAG_SMOOTH = 0.12, 0.25, 0.34
    PINCH_THRESHOLD, DEPTH_SCALE, DEPTH_OFFSET, DEPTH_MIN, DEPTH_MAX = 45, 2.5, -0.5, -1.0, 1.0
    BASE_ROT_X, BASE_ROT_Y, P_SENSITIVITY, FOV = 0.15, -0.3, 0.8, 0.8
    PICK_RADIUS, MIN_SCALE, MAX_SCALE = 95, 0.05, 2.8
    CREATE_CUBE_BTN, CREATE_CORE_BTN, DELETE_BTN = (40, 40, 78, 62), (130, 40, 78, 62), (220, 40, 78, 62)

    STARK_BLUE, STARK_EDGE, STARK_SOFT = (255, 160, 60), (255, 230, 150), (180, 120, 50)
    WHITE_SOFT, GREEN_SOFT, CYAN_SOFT = (220, 220, 220), (120, 255, 160), (255, 255, 140)
    PANEL_BG = (24, 28, 34)
    HUD_TITLE = "DEXTER // AR CORE ONLINE"

    GRAVITY = 0.015
    FRICTION = 0.92
    BOUNCE = 0.6
    FLOOR_Y = 0.82

    ARUCO_DICT = cv2.aruco.DICT_4X4_50 if hasattr(cv2, "aruco") else None
    ARUCO_TARGET_ID = 0
    ARUCO_MARKER_LENGTH_M = 0.06
    ANCHOR_SMOOTH = 0.35
    ANCHOR_MAX_MISSES = 18
    CAMERA_FOCAL_SCALE = 1.05
    MARKER_DRAG_PLANE_Z = 0.0
    DRAW_ANCHOR_AXES = True


class VideoWorker(QThread):
    frame_ready = pyqtSignal(QImage)
    stats_ready = pyqtSignal(int)
    log_msg = pyqtSignal(str)
    telemetry_ready = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.cfg = Config()
        self.vision = VisionEngine(self.cfg)
        self.holo = HologramEngine(self.cfg)
        self.ui = UIOverlay(self.cfg)

        self.cap = cv2.VideoCapture(self.cfg.CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.objects_3d = []
        self.selected_obj_index = -1
        self.drag_obj_index = -1

        self.face_offset_x = 0.0
        self.face_offset_y = 0.0
        self.last_hand_pos3d = None
        self.gesture = GestureState()
        self.base_dist = None
        self.base_scale = None
        self.base_angle = None
        self.base_rot_z = None

        self.black_bg_mode = False
        self.hud_flash = 0.0
        self.running = True

        self.cmd_spawn = None
        self.cmd_clear = False
        self.cmd_toggle_bg = False
        self.pending_spawn_drag = False
        self.last_time = time.time()

        self.camera_matrix = None
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)
        self.frame_size = None

        self.anchor_rvec = None
        self.anchor_tvec = None
        self.anchor_id = None
        self.anchor_visible = False
        self.anchor_miss_count = 0

    def stop(self):
        self.running = False
        self.wait()

    def _has_anchor_pose(self):
        return self.anchor_rvec is not None and self.anchor_tvec is not None and self.camera_matrix is not None

    def _ensure_camera_matrix(self, w, h):
        if self.frame_size == (w, h) and self.camera_matrix is not None:
            return
        focal = float(max(w, h) * self.cfg.CAMERA_FOCAL_SCALE)
        self.camera_matrix = np.array(
            [
                [focal, 0.0, w * 0.5],
                [0.0, focal, h * 0.5],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        self.frame_size = (w, h)

    def _update_anchor_pose(self, detection):
        if detection is None:
            self.anchor_visible = False
            if self._has_anchor_pose():
                self.anchor_miss_count += 1
                if self.anchor_miss_count > self.cfg.ANCHOR_MAX_MISSES:
                    self.anchor_rvec = None
                    self.anchor_tvec = None
                    self.anchor_id = None
            return

        self.anchor_visible = True
        self.anchor_miss_count = 0
        self.anchor_id = detection["id"]

        if self.anchor_rvec is None or self.anchor_tvec is None:
            self.anchor_rvec = detection["rvec"].copy()
            self.anchor_tvec = detection["tvec"].copy()
            return

        a = self.cfg.ANCHOR_SMOOTH
        self.anchor_rvec = self.anchor_rvec * (1.0 - a) + detection["rvec"] * a
        self.anchor_tvec = self.anchor_tvec * (1.0 - a) + detection["tvec"] * a

    def _project_marker_points(self, points_world):
        if not self._has_anchor_pose():
            return None
        pts = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
        img_pts, _ = cv2.projectPoints(
            pts,
            self.anchor_rvec,
            self.anchor_tvec,
            self.camera_matrix,
            self.dist_coeffs,
        )
        return [(int(p[0][0]), int(p[0][1])) for p in img_pts]

    def _project_object_center(self, obj, R, w, h):
        if obj.get("space", "screen") == "marker":
            pts = self._project_marker_points([obj["pos"]])
            return None if pts is None else pts[0]
        return self.holo.get_object_center_projected(obj, R, w, h)
    
    
   
    def _project_object_vertices(self, obj, verts, R, w, h):
        if obj.get("space", "screen") == "marker":
            return self._project_marker_points(verts)
        return [project(R @ v, w, h, self.cfg.FOV) for v in verts]

    def _screen_to_marker_plane(self, px, py, plane_z):
        if not self._has_anchor_pose():
            return None

        fx = float(self.camera_matrix[0, 0])
        fy = float(self.camera_matrix[1, 1])
        cx = float(self.camera_matrix[0, 2])
        cy = float(self.camera_matrix[1, 2])

        ray_cam = np.array([(px - cx) / fx, (py - cy) / fy, 1.0], dtype=np.float32)
        ray_cam /= max(1e-6, float(np.linalg.norm(ray_cam)))

        rot, _ = cv2.Rodrigues(self.anchor_rvec.astype(np.float32))
        rot_t = rot.T

        cam_origin_world = -rot_t @ self.anchor_tvec.reshape(3, 1)
        ray_dir_world = rot_t @ ray_cam.reshape(3, 1)

        dir_z = float(ray_dir_world[2, 0])
        if abs(dir_z) < 1e-6:
            return None

        t = (plane_z - float(cam_origin_world[2, 0])) / dir_z
        if t <= 0.0:
            return None

        hit = cam_origin_world + ray_dir_world * t
        return hit[:, 0]

    def run(self):
        self.log_msg.emit(
            f"FRIDAY: Kinematic physics online. Gravity={self.cfg.GRAVITY:.3f} u/frame^2."
        )
        if self.vision.aruco_available:
            self.log_msg.emit(
                f"FRIDAY: ArUco world anchor online. Target ID={self.cfg.ARUCO_TARGET_ID}."
            )
        else:
            self.log_msg.emit("FRIDAY: OpenCV ArUco module not available. Using screen-space mode.")

        while self.running:
            now = time.time()
            fps = int(1.0 / (now - self.last_time)) if (now - self.last_time) > 0 else 0
            self.last_time = now

            if self.cmd_clear:
                self.objects_3d.clear()
                self.selected_obj_index = -1
                self.drag_obj_index = -1
                self.hud_flash = 1.0
                self.cmd_clear = False
                self.log_msg.emit("FRIDAY: Scene purged.")

            if self.cmd_spawn:
                spawn_space = "marker" if self._has_anchor_pose() else "screen"
                spawn_pos = [0.0, 0.0, 0.04] if spawn_space == "marker" else [0.0, -0.35, 0.15]

                self.objects_3d.append(
                    {
                        "type": self.cmd_spawn,
                        "space": spawn_space,
                        "pos": spawn_pos,
                        "vel": [0.0, 0.0, 0.0],
                        "scale": 0.35,
                        "color": self.cfg.STARK_BLUE if self.cmd_spawn == "cube" else self.cfg.STARK_EDGE,
                        "spawn_t": time.time(),
                        "rot_x": 0.0,
                        "rot_y": 0.0,
                        "rot_z": 0.0,
                    }
                )
                spawned_idx = len(self.objects_3d) - 1
                self.selected_obj_index = spawned_idx
                if self.pending_spawn_drag:
                    self.drag_obj_index = spawned_idx
                    self.pending_spawn_drag = False
                self.hud_flash = 1.0
                self.log_msg.emit(
                    f"FRIDAY: {self.cmd_spawn.upper()} instantiated in {spawn_space.upper()} space."
                )
                self.cmd_spawn = None

            if self.cmd_toggle_bg:
                self.black_bg_mode = not self.black_bg_mode
                self.cmd_toggle_bg = False

            if not self.cap.isOpened():
                time.sleep(0.05)
                continue

            ok, frame = self.cap.read()
            if not ok:
                continue
      
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            self._ensure_camera_matrix(w, h)

            aruco_detection = self.vision.process_aruco(frame, self.camera_matrix, self.dist_coeffs)
            self._update_anchor_pose(aruco_detection)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t = time.time()

            self.face_offset_x, self.face_offset_y = self.vision.process_face(
                rgb, self.face_offset_x, self.face_offset_y
            )
            R = rot_matrix_x(self.cfg.BASE_ROT_X + self.face_offset_y) @ rot_matrix_y(
                self.cfg.BASE_ROT_Y - self.face_offset_x
            )
            
            hands = self.vision.process_hands(rgb, w, h)
            self.handle_interaction(hands[0] if hands else None, hands, R, w, h)

            layers = [np.zeros_like(frame) for _ in range(5)]
            if hands:
                self.ui.draw_finger_hud(layers[4], hands[0]["px"], t)

            if aruco_detection is not None and self.vision.aruco_available:
                marker_ids = np.array([[aruco_detection["id"]]], dtype=np.int32)
                cv2.aruco.drawDetectedMarkers(layers[4], [aruco_detection["corners"]], marker_ids)

            if self._has_anchor_pose() and self.cfg.DRAW_ANCHOR_AXES:
                cv2.drawFrameAxes(
                    layers[4],
                    self.camera_matrix,
                    self.dist_coeffs,
                    self.anchor_rvec,
                    self.anchor_tvec,
                    self.cfg.ARUCO_MARKER_LENGTH_M * 0.5,
                    2,
                )
            
            anchor_txt = f"ARUCO ID {self.anchor_id}" if self._has_anchor_pose() else "ARUCO SEARCH"
            cv2.putText(
                layers[4],
                anchor_txt,
                (20, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                self.cfg.STARK_EDGE,
                1,
                cv2.LINE_AA,
            )

            for i, obj in enumerate(self.objects_3d):
                is_dragging_this = isinstance(self.drag_obj_index, int) and i == self.drag_obj_index
                obj_space = obj.get("space", "screen")

                if not is_dragging_this and obj_space == "screen":
                    obj["vel"][1] += self.cfg.GRAVITY
                    obj["pos"][0] += obj["vel"][0]
                    obj["pos"][1] += obj["vel"][1]
                    obj["pos"][2] += obj["vel"][2]
                    obj["vel"] = [v * self.cfg.FRICTION for v in obj["vel"]]

                    if obj["pos"][1] > self.cfg.FLOOR_Y:
                        obj["pos"][1] = self.cfg.FLOOR_Y
                        obj["vel"][1] *= -self.cfg.BOUNCE
                        obj["vel"][0] *= 0.8
                        obj["vel"][2] *= 0.8
                        if abs(obj["vel"][1]) < 1.0:
                            obj["vel"][1] = 0.0

                if not is_dragging_this:
                    obj["rot_y"] = obj.get("rot_y", 0.0) + 0.05
                    obj["rot_x"] = obj.get("rot_x", 0.0) + 0.02

                verts, faces = self.holo.get_geometry(obj)
                pts = self._project_object_vertices(obj, verts, R, w, h)
                if pts is None:
                    continue

                if i == self.selected_obj_index:
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    bx, by = min(xs), min(ys)
                    bw, bh = max(xs) - bx, max(ys) - by
                    center = self._project_object_center(obj, R, w, h)
                    if center is not None:
                        self.ui.draw_selected_hud(layers[4], center, (bx, by, bw, bh), t)

                for face in faces:
                    poly = np.array([pts[idx] for idx in face], dtype=np.int32)
                    cv2.polylines(layers[2], [poly], True, self.cfg.STARK_SOFT, 1)

            final = self.holo.compose_final(frame, layers, self.hud_flash, self.black_bg_mode)
            self.hud_flash *= 0.88

            final_rgb = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
            bytes_per_line = final_rgb.shape[2] * final_rgb.shape[1]
            qt_img = QImage(
                final_rgb.data,
                final_rgb.shape[1],
                final_rgb.shape[0],
                bytes_per_line,
                QImage.Format.Format_RGB888,
            ).copy()

            self.frame_ready.emit(qt_img)
            self.stats_ready.emit(len(self.objects_3d))
            self.telemetry_ready.emit(
                {
                    "fps": fps,
                    "hands": len(hands),
                    "face_yaw": round(self.face_offset_x, 2),
                    "face_pitch": round(self.face_offset_y, 2),
                    "anchor": "LOCK" if self._has_anchor_pose() else "SEARCH",
                    "marker_id": self.anchor_id if self.anchor_id is not None else "--",
                }
            )

        if self.cap.isOpened():
            self.cap.release()
        self.vision.close()

    def handle_interaction(self, primary, hands, R, w, h):
        self.gesture.update(primary["grabbing"] if primary else False)
        if not primary:
            return

        self.last_hand_pos3d = smooth_vec3(self.last_hand_pos3d, primary["pos3d"], self.cfg.HAND_SMOOTH)
        px = primary["px"]

        if self.gesture.grab_started:
            if point_in_rect(px, self.cfg.CREATE_CUBE_BTN):
                self.cmd_spawn = "cube"
                self.pending_spawn_drag = True
                self.drag_obj_index = -1
            elif point_in_rect(px, self.cfg.CREATE_CORE_BTN):
                self.cmd_spawn = "core"
                self.pending_spawn_drag = True
                self.drag_obj_index = -1
            elif point_in_rect(px, self.cfg.DELETE_BTN) and self.selected_obj_index != -1:
                self.objects_3d.pop(self.selected_obj_index)
                self.selected_obj_index = -1
                self.drag_obj_index = -1
                self.log_msg.emit("FRIDAY: Object discarded.")
            else:
                best_idx, best_d = -1, self.cfg.PICK_RADIUS
                for i, obj in enumerate(self.objects_3d):
                    center = self._project_object_center(obj, R, w, h)
                    if center is None:
                        continue
                    d = math.hypot(px[0] - center[0], px[1] - center[1])
                    if d < best_d:
                        best_d, best_idx = d, i
                self.drag_obj_index = best_idx
                self.selected_obj_index = best_idx
                if best_idx != -1:
                    self.objects_3d[best_idx]["vel"] = [0.0, 0.0, 0.0]
                    self.log_msg.emit("FRIDAY: Grav-control disabled. Object under manipulation.")

        if self.gesture.current_grabbing and isinstance(self.drag_obj_index, int) and self.drag_obj_index != -1:
            try:
                obj = self.objects_3d[self.drag_obj_index]
                old_pos = list(obj["pos"])

                if obj.get("space", "screen") == "marker" and self._has_anchor_pose():
                    hit = self._screen_to_marker_plane(px[0], px[1], self.cfg.MARKER_DRAG_PLANE_Z)
                    if hit is not None:
                        obj["pos"][0] = float(hit[0])
                        obj["pos"][1] = float(hit[1])
                else:
                    obj["pos"] = smooth_vec3(obj["pos"], self.last_hand_pos3d, self.cfg.OBJ_DRAG_SMOOTH)

                obj["vel"] = [
                    (obj["pos"][0] - old_pos[0]) * 1.5,
                    (obj["pos"][1] - old_pos[1]) * 1.5,
                    (obj["pos"][2] - old_pos[2]) * 1.5,
                ]

                if len(hands) == 2 and hands[1]["grabbing"]:
                    x1, y1 = hands[0]["px"]
                    x2, y2 = hands[1]["px"]
                    dist = math.hypot(x1 - x2, y1 - y2)
                    angle = math.atan2(y2 - y1, x2 - x1)
                    if self.base_dist is None:
                        self.base_dist = dist
                        self.base_scale = obj["scale"]
                        self.base_angle = angle
                        self.base_rot_z = obj.get("rot_z", 0.0)
                    else:
                        obj["scale"] = clamp(
                            self.base_scale * (dist / max(self.base_dist, 1.0)),
                            self.cfg.MIN_SCALE,
                            self.cfg.MAX_SCALE,
                        )
                        obj["rot_z"] = self.base_rot_z + (angle - self.base_angle)
                else:
                    self.base_dist = None
            except (IndexError, TypeError):
                pass

        if self.gesture.grab_ended:
            self.drag_obj_index = -1
            self.base_dist = None
            self.log_msg.emit("FRIDAY: Physics restored. Releasing object.")


class DexterWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(Config.WINDOW_NAME)
        self.resize(1280, 800)
        self.init_ui()

        self.worker = VideoWorker()
        self.worker.frame_ready.connect(self.update_video)
        self.worker.stats_ready.connect(self.update_stats)
        self.worker.log_msg.connect(self.append_log)
        self.worker.telemetry_ready.connect(self.update_telemetry)
        self.worker.start()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        header_label = QLabel("D.E.X.T.E.R. SYSTEM CORE // PROTOCOL ALPHA")
        header_label.setStyleSheet(
            "color: #00d2ff; font-family: 'Courier New'; font-size: 18px; font-weight: bold; letter-spacing: 2px;"
        )
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(header_label)

        middle_layout = QHBoxLayout()

        telemetry_panel = QFrame()
        telemetry_panel.setStyleSheet(
            "QFrame { background-color: rgba(10, 15, 20, 0.8); border: 1px solid #005577; border-radius: 8px; }"
        )
        tel_layout = QVBoxLayout(telemetry_panel)

        lbl_tel_title = QLabel("TELEMETRIA")
        lbl_tel_title.setStyleSheet(
            "color: #ff9900; font-weight: bold; font-family: 'Courier New'; font-size: 14px; border: none;"
        )
        self.lbl_fps = QLabel("FPS: --")
        self.lbl_hands = QLabel("Hands: 0")
        self.lbl_yaw = QLabel("Yaw: 0.0")
        self.lbl_pitch = QLabel("Pitch: 0.0")
        self.lbl_anchor = QLabel("Anchor: SEARCH")
        self.lbl_marker = QLabel("Marker: --")

        for lbl in [
            lbl_tel_title,
            self.lbl_fps,
            self.lbl_hands,
            self.lbl_yaw,
            self.lbl_pitch,
            self.lbl_anchor,
            self.lbl_marker,
        ]:
            if lbl != lbl_tel_title:
                lbl.setStyleSheet(
                    "color: #00d2ff; font-family: 'Courier New'; font-size: 13px; border: none;"
                )
            tel_layout.addWidget(lbl)
        tel_layout.addStretch()
        middle_layout.addWidget(telemetry_panel, stretch=1)

        self.video_label = QLabel("Iniciando Matriz Visual...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: #000; border: 2px solid #00d2ff; border-radius: 8px;")
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        middle_layout.addWidget(self.video_label, stretch=5)

        side_panel = QFrame()
        side_panel.setStyleSheet(
            """
            QFrame { background-color: rgba(10, 15, 20, 0.8); border-radius: 8px; border: 1px solid #005577; }
            QLabel { color: #00d2ff; font-family: 'Courier New'; font-size: 13px; font-weight: bold; border: none; }
            QPushButton {
                background-color: rgba(0, 50, 80, 0.5); color: #00d2ff; border: 1px solid #00d2ff;
                padding: 12px; border-radius: 4px; font-weight: bold; font-family: 'Courier New';
            }
            QPushButton:hover { background-color: #00d2ff; color: #000; box-shadow: 0px 0px 10px #00d2ff; }
            QPushButton:pressed { background-color: #ffffff; }
            """
        )
        side_layout = QVBoxLayout(side_panel)

        self.obj_count_label = QLabel("Objetos: 0")
        side_layout.addWidget(self.obj_count_label)
        side_layout.addSpacing(15)

        btn_cube = QPushButton("INSTANCIAR CUBO")
        btn_cube.clicked.connect(lambda: setattr(self.worker, "cmd_spawn", "cube"))

        btn_core = QPushButton("INSTANCIAR CORE")
        btn_core.clicked.connect(lambda: setattr(self.worker, "cmd_spawn", "core"))

        btn_bg = QPushButton("MODO CINEMATICO")
        btn_bg.clicked.connect(lambda: setattr(self.worker, "cmd_toggle_bg", True))

        btn_clear = QPushButton("PURGAR CENA")
        btn_clear.clicked.connect(lambda: setattr(self.worker, "cmd_clear", True))
        btn_clear.setStyleSheet(
            "QPushButton { color: #ff3333; border-color: #ff3333; } "
            "QPushButton:hover { background-color: #ff3333; color: black; }"
        )

        side_layout.addWidget(btn_cube)
        side_layout.addWidget(btn_core)
        side_layout.addWidget(btn_bg)
        side_layout.addSpacing(20)
        side_layout.addWidget(btn_clear)
        side_layout.addStretch()

        middle_layout.addWidget(side_panel, stretch=1)
        main_layout.addLayout(middle_layout, stretch=4)

        console_frame = QFrame()
        console_frame.setStyleSheet("background-color: #050505; border: 1px solid #333; border-radius: 5px;")
        console_layout = QVBoxLayout(console_frame)
        console_layout.setContentsMargins(5, 5, 5, 5)

        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setStyleSheet(
            "color: #00ff00; background-color: transparent; border: none; "
            "font-family: 'Consolas', 'Courier New'; font-size: 13px;"
        )
        self.console.setFixedHeight(120)
        console_layout.addWidget(self.console)

        main_layout.addWidget(console_frame, stretch=1)

    def update_video(self, qt_img):
        if self.video_label.width() > 0 and self.video_label.height() > 0:
            pixmap = QPixmap.fromImage(qt_img).scaled(
                self.video_label.width(),
                self.video_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.video_label.setPixmap(pixmap)

    def update_stats(self, count):
        self.obj_count_label.setText(f"Entidades Ativas: {count}")

    def update_telemetry(self, data):
        self.lbl_fps.setText(f"Taxa (FPS) : {data['fps']}")
        self.lbl_hands.setText(f"Maos Detec.: {data['hands']}")
        self.lbl_yaw.setText(f"Eixo Yaw   : {data['face_yaw']}")
        self.lbl_pitch.setText(f"Eixo Pitch : {data['face_pitch']}")
        self.lbl_anchor.setText(f"Anchor     : {data['anchor']}")
        self.lbl_marker.setText(f"Marker ID  : {data['marker_id']}")

    def append_log(self, msg):
        time_str = datetime.now().strftime("%H:%M:%S")
        self.console.append(f"[{time_str}] {msg}")
        self.console.verticalScrollBar().setValue(self.console.verticalScrollBar().maximum())

    def closeEvent(self, event):
        self.worker.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet("QMainWindow { background-color: #080a0d; }")
    window = DexterWindow()
    window.show()
    sys.exit(app.exec())
