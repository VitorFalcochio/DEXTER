import cv2
import mediapipe as mp
import numpy as np
import json
import math
import time


# =========================================================
# CONFIG
# =========================================================
class Config:
    WINDOW_NAME = "DEXTER v14.0"
    SCENE_FILE = "dexter_scene_v14.json"

    CAMERA_INDEX = 0

    MAX_NUM_HANDS = 2
    HAND_DET_CONF = 0.7
    HAND_TRACK_CONF = 0.7

    MAX_NUM_FACES = 1
    FACE_DET_CONF = 0.7
    FACE_TRACK_CONF = 0.7

    FACE_SMOOTH = 0.12
    HAND_SMOOTH = 0.25
    OBJ_DRAG_SMOOTH = 0.34

    PINCH_THRESHOLD = 45
    DEPTH_SCALE = 2.5
    DEPTH_OFFSET = -0.5
    DEPTH_MIN = -1.0
    DEPTH_MAX = 1.0

    BASE_ROT_X = 0.15
    BASE_ROT_Y = -0.3
    P_SENSITIVITY = 0.8
    FOV = 0.8

    PICK_RADIUS = 95
    MIN_SCALE = 0.05
    MAX_SCALE = 2.8

    CREATE_CUBE_BTN = (40, 40, 78, 62)
    CREATE_PYRAMID_BTN = (130, 40, 78, 62)
    DELETE_BTN = (220, 40, 78, 62)

    HUD_TITLE = "DEXTER v14.0 // CLEAN ARCHITECTURE EDITION"

    STARK_BLUE = (255, 160, 60)
    STARK_EDGE = (255, 230, 150)
    STARK_SOFT = (180, 120, 50)
    WHITE_SOFT = (220, 220, 220)
    RED_SOFT = (80, 80, 255)
    GREEN_SOFT = (120, 255, 120)
    CYAN_SOFT = (255, 220, 120)
    PANEL_BG = (35, 25, 10)
    SHADOW_COLOR = (30, 20, 10)
    GRID_COLOR = (80, 60, 25)


# =========================================================
# HELPERS
# =========================================================
def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def smooth_value(current, target, alpha):
    return current * (1.0 - alpha) + target * alpha


def smooth_vec3(current, target, alpha):
    if current is None:
        return list(target)
    return [
        current[0] * (1.0 - alpha) + target[0] * alpha,
        current[1] * (1.0 - alpha) + target[1] * alpha,
        current[2] * (1.0 - alpha) + target[2] * alpha,
    ]


def safe_normal(v):
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return v / norm


def point_in_rect(pt, rect):
    x, y = pt
    rx, ry, rw, rh = rect
    return rx <= x <= rx + rw and ry <= y <= ry + rh


def project(pt3d, w, h, fov=0.8):
    x, y, z = pt3d
    denom = 1.0 + z * fov
    if denom <= 0.01:
        denom = 0.01
    sx = int((x / denom) * w * 0.5 + w * 0.5)
    sy = int((y / denom) * h * 0.5 + h * 0.5)
    return sx, sy


def rot_matrix_x(a):
    return np.array([
        [1, 0, 0],
        [0, np.cos(a), -np.sin(a)],
        [0, np.sin(a),  np.cos(a)]
    ], dtype=np.float32)


def rot_matrix_y(a):
    return np.array([
        [ np.cos(a), 0, np.sin(a)],
        [0,          1, 0],
        [-np.sin(a), 0, np.cos(a)]
    ], dtype=np.float32)


# =========================================================
# GESTURE STATE
# =========================================================
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


# =========================================================
# APP
# =========================================================
class DexterApp:
    def __init__(self):
        self.cfg = Config()

        self.objects_3d = []
        self.selected_obj_index = -1
        self.drag_obj_index = -1

        self.face_offset_x = 0.0
        self.face_offset_y = 0.0

        self.last_hand_pos3d = None
        self.gesture = GestureState()

        self.base_dist_scaling = None
        self.base_scale_obj = None

        self.black_bg_mode = False
        self.hud_flash = 0.0

        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh

        self.hands = self.mp_hands.Hands(
            max_num_hands=self.cfg.MAX_NUM_HANDS,
            min_detection_confidence=self.cfg.HAND_DET_CONF,
            min_tracking_confidence=self.cfg.HAND_TRACK_CONF
        )

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=self.cfg.MAX_NUM_FACES,
            min_detection_confidence=self.cfg.FACE_DET_CONF,
            min_tracking_confidence=self.cfg.FACE_TRACK_CONF
        )

        self.cap = cv2.VideoCapture(self.cfg.CAMERA_INDEX)
        if not self.cap.isOpened():
            raise RuntimeError("Não foi possível acessar a câmera.")

        cv2.namedWindow(self.cfg.WINDOW_NAME)

    # -----------------------------------------------------
    # Scene / persistence
    # -----------------------------------------------------
    def create_object(self, obj_type, pos3d):
        color = self.cfg.STARK_BLUE if obj_type == "cube" else self.cfg.CYAN_SOFT
        obj = {
            "type": obj_type,
            "pos": list(pos3d),
            "scale": 0.35,
            "color": color,
            "bbox": (0, 0, 0, 0),
            "spawn_t": time.time()
        }
        self.objects_3d.append(obj)
        self.hud_flash = 1.0
        return len(self.objects_3d) - 1

    def save_scene(self):
        data = []
        for obj in self.objects_3d:
            data.append({
                "type": obj.get("type", "cube"),
                "pos": [float(v) for v in obj["pos"]],
                "scale": float(obj["scale"]),
                "color": list(obj["color"]),
                "spawn_t": float(obj.get("spawn_t", time.time()))
            })
        with open(self.cfg.SCENE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print("💾 Cena salva.")

    def load_scene(self):
        try:
            with open(self.cfg.SCENE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

            loaded = []
            for item in data:
                loaded.append({
                    "type": item.get("type", "cube"),
                    "pos": list(item.get("pos", [0.0, 0.0, 0.0])),
                    "scale": float(item.get("scale", 0.35)),
                    "color": tuple(item.get("color", self.cfg.STARK_BLUE)),
                    "bbox": (0, 0, 0, 0),
                    "spawn_t": float(item.get("spawn_t", time.time()))
                })

            self.objects_3d = loaded
            self.selected_obj_index = -1
            self.drag_obj_index = -1
            print("💾 Cena carregada.")
        except FileNotFoundError:
            print("⚠️ Arquivo de cena não encontrado.")
        except Exception as e:
            print(f"❌ Erro ao carregar cena: {e}")

    # -----------------------------------------------------
    # Geometry
    # -----------------------------------------------------
    def get_cube_geometry(self, obj):
        s = obj["scale"] / 2.0
        px, py, pz = obj["pos"]

        verts = np.array([
            [-s, -s, -s],
            [ s, -s, -s],
            [ s,  s, -s],
            [-s,  s, -s],
            [-s, -s,  s],
            [ s, -s,  s],
            [ s,  s,  s],
            [-s,  s,  s]
        ], dtype=np.float32)

        verts += np.array([px, py, pz], dtype=np.float32)

        faces = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [2, 3, 7, 6],
            [0, 3, 7, 4],
            [1, 2, 6, 5]
        ]
        return verts, faces

    def get_pyramid_geometry(self, obj):
        s = obj["scale"] / 2.0
        px, py, pz = obj["pos"]

        verts = np.array([
            [-s,  s, -s],
            [ s,  s, -s],
            [ s,  s,  s],
            [-s,  s,  s],
            [ 0, -s, 0],
        ], dtype=np.float32)

        verts += np.array([px, py, pz], dtype=np.float32)

        faces = [
            [0, 1, 2, 3],
            [0, 1, 4],
            [1, 2, 4],
            [2, 3, 4],
            [3, 0, 4]
        ]
        return verts, faces

    def get_geometry(self, obj):
        obj_type = obj.get("type", "cube")
        if obj_type == "pyramid":
            return self.get_pyramid_geometry(obj)
        return self.get_cube_geometry(obj)

    def get_object_center_projected(self, obj, R, w, h):
        center = np.array(obj["pos"], dtype=np.float32)
        rotated = R @ center
        return project(rotated, w, h, self.cfg.FOV)

    # -----------------------------------------------------
    # Tracking
    # -----------------------------------------------------
    def process_face(self, rgb_frame):
        res_face = self.face_mesh.process(rgb_frame)
        if res_face.multi_face_landmarks:
            face_landmarks = res_face.multi_face_landmarks[0]
            center_eye = face_landmarks.landmark[168]

            target_x = ((center_eye.x) * 2.0 - 1.0) * self.cfg.P_SENSITIVITY
            target_y = ((center_eye.y) * 2.0 - 1.0) * self.cfg.P_SENSITIVITY

            self.face_offset_x = smooth_value(self.face_offset_x, target_x, self.cfg.FACE_SMOOTH)
            self.face_offset_y = smooth_value(self.face_offset_y, target_y, self.cfg.FACE_SMOOTH)

    def process_hands(self, rgb_frame, w, h):
        res_hands = self.hands.process(rgb_frame)
        hand_data = []

        if res_hands.multi_hand_landmarks:
            for hand_lms in res_hands.multi_hand_landmarks:
                idx = hand_lms.landmark[8]
                thumb = hand_lms.landmark[4]
                wrist = hand_lms.landmark[0]
                mid_base = hand_lms.landmark[9]

                ix, iy = int(idx.x * w), int(idx.y * h)
                tx, ty = int(thumb.x * w), int(thumb.y * h)

                dist_pinch = math.hypot(tx - ix, ty - iy)
                grabbing = dist_pinch < self.cfg.PINCH_THRESHOLD

                hand_span = math.hypot(wrist.x - mid_base.x, wrist.y - mid_base.y)
                z_val = clamp(
                    hand_span * self.cfg.DEPTH_SCALE + self.cfg.DEPTH_OFFSET,
                    self.cfg.DEPTH_MIN,
                    self.cfg.DEPTH_MAX
                )

                x3d = idx.x * 2.0 - 1.0
                y3d = -(idx.y * 2.0 - 1.0)

                hand_data.append({
                    "px": (ix, iy),
                    "pos3d": [x3d, y3d, z_val],
                    "grabbing": grabbing
                })

        return hand_data

    # -----------------------------------------------------
    # Interaction
    # -----------------------------------------------------
    def get_scene_rotation(self):
        return rot_matrix_x(self.cfg.BASE_ROT_X + self.face_offset_y) @ rot_matrix_y(self.cfg.BASE_ROT_Y - self.face_offset_x)

    def pick_nearest_object(self, hand_px, R, w, h, radius=None):
        if radius is None:
            radius = self.cfg.PICK_RADIUS

        best_idx = -1
        best_dist = 999999.0

        for i, obj in enumerate(self.objects_3d):
            cx, cy = self.get_object_center_projected(obj, R, w, h)
            d = math.hypot(hand_px[0] - cx, hand_px[1] - cy)
            if d < radius and d < best_dist:
                best_dist = d
                best_idx = i

        return best_idx

    def reset_scaling_state(self):
        self.base_dist_scaling = None
        self.base_scale_obj = None

    def handle_interaction(self, primary_hand, hand_data, R, w, h):
        current_grabbing = primary_hand["grabbing"] if primary_hand else False
        self.gesture.update(current_grabbing)

        if primary_hand:
            self.last_hand_pos3d = smooth_vec3(
                self.last_hand_pos3d,
                primary_hand["pos3d"],
                self.cfg.HAND_SMOOTH
            )
            primary_hand["pos3d_smooth"] = self.last_hand_pos3d
        else:
            self.last_hand_pos3d = None

        if not primary_hand:
            self.drag_obj_index = -1
            self.reset_scaling_state()
            return

        hand_px = primary_hand["px"]
        hand_pos3d = primary_hand["pos3d_smooth"]

        if self.gesture.grab_started:
            if point_in_rect(hand_px, self.cfg.CREATE_CUBE_BTN):
                self.selected_obj_index = self.create_object("cube", hand_pos3d)
                self.drag_obj_index = self.selected_obj_index

            elif point_in_rect(hand_px, self.cfg.CREATE_PYRAMID_BTN):
                self.selected_obj_index = self.create_object("pyramid", hand_pos3d)
                self.drag_obj_index = self.selected_obj_index

            elif point_in_rect(hand_px, self.cfg.DELETE_BTN):
                if self.selected_obj_index != -1 and 0 <= self.selected_obj_index < len(self.objects_3d):
                    self.objects_3d.pop(self.selected_obj_index)
                    self.selected_obj_index = -1
                    self.drag_obj_index = -1

            else:
                picked = self.pick_nearest_object(hand_px, R, w, h)
                if picked != -1:
                    self.selected_obj_index = picked
                    self.drag_obj_index = picked
                else:
                    self.selected_obj_index = -1
                    self.drag_obj_index = -1

        if self.gesture.current_grabbing and self.drag_obj_index != -1 and 0 <= self.drag_obj_index < len(self.objects_3d):
            obj = self.objects_3d[self.drag_obj_index]
            obj["pos"] = smooth_vec3(obj["pos"], hand_pos3d, self.cfg.OBJ_DRAG_SMOOTH)

            if len(hand_data) == 2:
                h1, h2 = hand_data[0], hand_data[1]
                if h1["grabbing"] and h2["grabbing"]:
                    dist = math.hypot(h1["px"][0] - h2["px"][0], h1["px"][1] - h2["px"][1])
                    if self.base_dist_scaling is None:
                        self.base_dist_scaling = dist
                        self.base_scale_obj = obj["scale"]
                    else:
                        ratio = dist / max(self.base_dist_scaling, 1.0)
                        obj["scale"] = clamp(
                            self.base_scale_obj * ratio,
                            self.cfg.MIN_SCALE,
                            self.cfg.MAX_SCALE
                        )
                else:
                    self.reset_scaling_state()
            else:
                self.reset_scaling_state()

        if self.gesture.grab_ended:
            self.drag_obj_index = -1
            self.reset_scaling_state()

    # -----------------------------------------------------
    # Render helpers
    # -----------------------------------------------------
    def draw_button(self, img, rect, label, color):
        x, y, w, h = rect
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)
        cv2.putText(img, label, (x + 7, y + h + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.cfg.WHITE_SOFT, 1, cv2.LINE_AA)

    def draw_finger_hud(self, canvas, pt, t):
        x, y = pt
        pulse = 8 + int(5 * (0.5 + 0.5 * math.sin(t * 5.0)))
        ring2 = 18 + int(4 * (0.5 + 0.5 * math.cos(t * 3.7)))
        ring3 = 30 + int(6 * (0.5 + 0.5 * math.sin(t * 2.3)))

        cv2.circle(canvas, (x, y), pulse, self.cfg.STARK_EDGE, 1, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), ring2, self.cfg.STARK_BLUE, 1, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), ring3, self.cfg.STARK_SOFT, 1, cv2.LINE_AA)

        cv2.line(canvas, (x - 40, y), (x - 14, y), self.cfg.STARK_SOFT, 1, cv2.LINE_AA)
        cv2.line(canvas, (x + 14, y), (x + 40, y), self.cfg.STARK_SOFT, 1, cv2.LINE_AA)
        cv2.line(canvas, (x, y - 40), (x, y - 14), self.cfg.STARK_SOFT, 1, cv2.LINE_AA)
        cv2.line(canvas, (x, y + 14), (x, y + 40), self.cfg.STARK_SOFT, 1, cv2.LINE_AA)

    def draw_selected_hud(self, canvas, center_px, bbox, t):
        x, y = center_px
        bx, by, bw, bh = bbox

        cv2.rectangle(canvas, (bx - 10, by - 10), (bx + bw + 10, by + bh + 10),
                      self.cfg.GREEN_SOFT, 1, cv2.LINE_AA)

        orbit = 24 + int(5 * math.sin(t * 4.0))
        cv2.circle(canvas, (x, y), orbit, self.cfg.GREEN_SOFT, 1, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), orbit + 12, self.cfg.CYAN_SOFT, 1, cv2.LINE_AA)

        cv2.putText(canvas, "TRACK LOCK", (bx, by - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.cfg.GREEN_SOFT, 1, cv2.LINE_AA)

    def draw_spawn_effect(self, canvas, center_px, elapsed):
        if elapsed > 0.7:
            return
        x, y = center_px
        progress = elapsed / 0.7
        radius = int(12 + progress * 50)
        alpha_color = self.cfg.STARK_EDGE if progress < 0.5 else self.cfg.CYAN_SOFT

        cv2.circle(canvas, (x, y), radius, alpha_color, 1, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), max(8, radius // 2), self.cfg.STARK_BLUE, 1, cv2.LINE_AA)

    def draw_world_grid(self, canvas, R, w, h):
        for gx in np.linspace(-1.7, 1.7, 11):
            p1 = np.array([gx, 0.9, -1.0], dtype=np.float32)
            p2 = np.array([gx, 0.9,  1.0], dtype=np.float32)
            cv2.line(canvas, project(R @ p1, w, h, self.cfg.FOV), project(R @ p2, w, h, self.cfg.FOV),
                     self.cfg.GRID_COLOR, 1, cv2.LINE_AA)

        for gz in np.linspace(-1.0, 1.0, 11):
            p1 = np.array([-1.7, 0.9, gz], dtype=np.float32)
            p2 = np.array([ 1.7, 0.9, gz], dtype=np.float32)
            cv2.line(canvas, project(R @ p1, w, h, self.cfg.FOV), project(R @ p2, w, h, self.cfg.FOV),
                     self.cfg.GRID_COLOR, 1, cv2.LINE_AA)

    def draw_axes(self, canvas, R, w, h):
        origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        x_axis = np.array([0.55, 0.0, 0.0], dtype=np.float32)
        y_axis = np.array([0.0, -0.55, 0.0], dtype=np.float32)
        z_axis = np.array([0.0, 0.0, 0.55], dtype=np.float32)

        po = project(R @ origin, w, h, self.cfg.FOV)
        px = project(R @ x_axis, w, h, self.cfg.FOV)
        py = project(R @ y_axis, w, h, self.cfg.FOV)
        pz = project(R @ z_axis, w, h, self.cfg.FOV)

        cv2.line(canvas, po, px, (80, 80, 255), 2, cv2.LINE_AA)
        cv2.line(canvas, po, py, (80, 255, 80), 2, cv2.LINE_AA)
        cv2.line(canvas, po, pz, (255, 80, 80), 2, cv2.LINE_AA)

    def draw_radar_hud(self, canvas, t):
        cx, cy = 130, 170
        cv2.circle(canvas, (cx, cy), 65, self.cfg.STARK_SOFT, 1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 42, self.cfg.STARK_SOFT, 1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 20, self.cfg.STARK_SOFT, 1, cv2.LINE_AA)

        angle = t * 1.8
        x2 = int(cx + math.cos(angle) * 65)
        y2 = int(cy + math.sin(angle) * 65)
        cv2.line(canvas, (cx, cy), (x2, y2), self.cfg.STARK_EDGE, 1, cv2.LINE_AA)

        cv2.putText(canvas, "SCAN", (cx - 18, cy + 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.cfg.STARK_EDGE, 1, cv2.LINE_AA)

    def draw_side_panel(self, img):
        h, w, _ = img.shape
        panel_w = 270
        x0 = w - panel_w - 20
        y0 = 80
        y1 = h - 40

        overlay = img.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y1), self.cfg.PANEL_BG, -1)
        cv2.addWeighted(overlay, 0.28, img, 0.72, 0, img)

        cv2.rectangle(img, (x0, y0), (x0 + panel_w, y1), self.cfg.STARK_SOFT, 1, cv2.LINE_AA)

        cv2.putText(img, "SYSTEM PANEL", (x0 + 15, y0 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.cfg.STARK_EDGE, 1, cv2.LINE_AA)

        lines = [
            f"Objects: {len(self.objects_3d)}",
            f"Selected: {self.selected_obj_index if self.selected_obj_index != -1 else 'NONE'}",
            f"View mode: {'BLACK BG' if self.black_bg_mode else 'NORMAL'}",
            "Controls:",
            "- pinch = select / drag",
            "- 2 hands = scale",
            "- B = toggle background",
            "- S = save",
            "- L = load",
            "- X = delete selected",
            "- C = clear scene",
            "- Q = quit",
            "",
            "Buttons:",
            "[CUBE] [PYRAMID] [DELETE]"
        ]

        yy = y0 + 60
        for line in lines:
            cv2.putText(img, line, (x0 + 15, yy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.43, self.cfg.WHITE_SOFT, 1, cv2.LINE_AA)
            yy += 24

        if self.selected_obj_index != -1 and 0 <= self.selected_obj_index < len(self.objects_3d):
            obj = self.objects_3d[self.selected_obj_index]
            info = [
                "",
                "Object data:",
                f"type: {obj.get('type', 'cube')}",
                f"scale: {obj.get('scale', 0):.2f}",
                f"x: {obj['pos'][0]:.2f}",
                f"y: {obj['pos'][1]:.2f}",
                f"z: {obj['pos'][2]:.2f}",
            ]
            for line in info:
                cv2.putText(img, line, (x0 + 15, yy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.43, self.cfg.CYAN_SOFT, 1, cv2.LINE_AA)
                yy += 24

    def draw_bottom_hud(self, img):
        h, _w, _ = img.shape
        cv2.putText(img, self.cfg.HUD_TITLE, (20, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.56, self.cfg.WHITE_SOFT, 1, cv2.LINE_AA)

        cv2.putText(img, f"OBJETS: {len(self.objects_3d)}", (20, h - 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, self.cfg.STARK_EDGE, 1, cv2.LINE_AA)

        cv2.putText(img, f"SELECTED: {self.selected_obj_index if self.selected_obj_index != -1 else 'NONE'}", (220, h - 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, self.cfg.GREEN_SOFT, 1, cv2.LINE_AA)

        mode_text = "MODE: BLACK BG" if self.black_bg_mode else "MODE: NORMAL"
        cv2.putText(img, mode_text, (450, h - 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, self.cfg.CYAN_SOFT, 1, cv2.LINE_AA)

    # -----------------------------------------------------
    # Render
    # -----------------------------------------------------
    def render_scene(self, frame, primary_hand, R, t):
        h, w, _ = frame.shape

        canvas_grid = np.zeros_like(frame)
        canvas_faces = np.zeros_like(frame)
        canvas_wire = np.zeros_like(frame)
        canvas_fx = np.zeros_like(frame)
        canvas_hud = np.zeros_like(frame)

        self.draw_world_grid(canvas_grid, R, w, h)
        self.draw_axes(canvas_grid, R, w, h)
        self.draw_radar_hud(canvas_hud, t)

        if primary_hand:
            self.draw_finger_hud(canvas_hud, primary_hand["px"], t)

        all_faces = []

        for obj_idx, obj in enumerate(self.objects_3d):
            verts, faces = self.get_geometry(obj)
            rotated_verts = [R @ v for v in verts]
            projected_pts = [project(v, w, h, self.cfg.FOV) for v in rotated_verts]

            pts_np = np.array(projected_pts, dtype=np.int32)
            min_x = int(np.min(pts_np[:, 0]))
            min_y = int(np.min(pts_np[:, 1]))
            max_x = int(np.max(pts_np[:, 0]))
            max_y = int(np.max(pts_np[:, 1]))
            obj["bbox"] = (min_x, min_y, max_x - min_x, max_y - min_y)

            center_world = np.array(obj["pos"], dtype=np.float32)
            shadow_pos = np.array([center_world[0], 0.9, center_world[2]], dtype=np.float32)
            shadow_rot = R @ shadow_pos
            shadow_px = project(shadow_rot, w, h, self.cfg.FOV)
            shadow_size = int(22 * obj["scale"] * 2.3)
            cv2.ellipse(canvas_fx, shadow_px, (shadow_size, max(8, shadow_size // 3)),
                        0, 0, 360, self.cfg.SHADOW_COLOR, -1, cv2.LINE_AA)

            center_px = self.get_object_center_projected(obj, R, w, h)
            elapsed_spawn = t - obj.get("spawn_t", t)
            self.draw_spawn_effect(canvas_hud, center_px, elapsed_spawn)

            if obj_idx == self.selected_obj_index:
                self.draw_selected_hud(canvas_hud, center_px, obj["bbox"], t)

            for face in faces:
                avg_z = float(np.mean([rotated_verts[idx][2] for idx in face]))

                if len(face) >= 3:
                    edge1 = rotated_verts[face[1]] - rotated_verts[face[0]]
                    edge2 = rotated_verts[face[2]] - rotated_verts[face[0]]
                    normal = safe_normal(np.cross(edge1, edge2))
                else:
                    normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)

                all_faces.append({
                    "obj_idx": obj_idx,
                    "pts": np.array([projected_pts[idx] for idx in face], dtype=np.int32),
                    "z": avg_z,
                    "norm": normal
                })

        all_faces.sort(key=lambda item: item["z"], reverse=True)

        for face in all_faces:
            light_dir = safe_normal(np.array([0.22, -1.0, -0.6], dtype=np.float32))
            intensity = float(clamp(np.dot(face["norm"], light_dir), 0.25, 1.0))

            obj_idx = face["obj_idx"]
            selected = (obj_idx == self.selected_obj_index)
            obj = self.objects_3d[obj_idx]

            base_color = np.array(obj["color"], dtype=np.float32)
            if selected:
                base_color = np.array(self.cfg.STARK_EDGE, dtype=np.float32)

            shaded = tuple(int(c * intensity) for c in base_color)
            cv2.fillConvexPoly(canvas_faces, face["pts"], shaded)

            edge_color = self.cfg.GREEN_SOFT if selected else self.cfg.STARK_SOFT
            edge_thickness = 2 if selected else 1
            cv2.polylines(canvas_wire, [face["pts"]], True, edge_color, edge_thickness, cv2.LINE_AA)

        return canvas_grid, canvas_faces, canvas_wire, canvas_fx, canvas_hud

    def compose_final(self, frame, layers):
        canvas_grid, canvas_faces, canvas_wire, canvas_fx, canvas_hud = layers

        glow_grid = cv2.GaussianBlur(canvas_grid, (11, 11), 0)
        glow_faces = cv2.GaussianBlur(canvas_faces, (25, 25), 0)
        glow_wire = cv2.GaussianBlur(canvas_wire, (7, 7), 0)
        glow_fx = cv2.GaussianBlur(canvas_fx, (19, 19), 0)
        glow_hud = cv2.GaussianBlur(canvas_hud, (9, 9), 0)

        hologram = cv2.addWeighted(canvas_grid, 0.7, glow_grid, 1.0, 0)
        hologram = cv2.addWeighted(hologram, 1.0, canvas_fx, 0.5, 0)
        hologram = cv2.addWeighted(hologram, 1.0, glow_fx, 0.9, 0)
        hologram = cv2.addWeighted(hologram, 1.0, canvas_faces, 0.78, 0)
        hologram = cv2.addWeighted(hologram, 1.0, glow_faces, 1.25, 0)
        hologram = cv2.addWeighted(hologram, 1.0, canvas_wire, 1.0, 0)
        hologram = cv2.addWeighted(hologram, 1.0, glow_wire, 1.45, 0)
        hologram = cv2.addWeighted(hologram, 1.0, canvas_hud, 0.9, 0)
        hologram = cv2.addWeighted(hologram, 1.0, glow_hud, 1.1, 0)

        mask = cv2.cvtColor(canvas_faces + canvas_grid + canvas_wire + canvas_hud, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        alpha = cv2.merge([mask, mask, mask]).astype(np.float32) / 255.0

        base_frame = np.zeros_like(frame) if self.black_bg_mode else frame

        final = (base_frame.astype(np.float32) * (1.0 - alpha * 0.46) +
                 hologram.astype(np.float32) * (alpha * 0.96)).astype(np.uint8)

        self.draw_button(final, self.cfg.CREATE_CUBE_BTN, "CUBE", self.cfg.STARK_BLUE)
        self.draw_button(final, self.cfg.CREATE_PYRAMID_BTN, "PYRAMID", self.cfg.CYAN_SOFT)
        self.draw_button(final, self.cfg.DELETE_BTN, "DELETE", self.cfg.RED_SOFT)

        self.draw_side_panel(final)
        self.draw_bottom_hud(final)

        if self.hud_flash > 0.01:
            overlay = final.copy()
            intensity = clamp(self.hud_flash, 0.0, 1.0)
            cv2.rectangle(overlay, (0, 0), (final.shape[1], final.shape[0]), self.cfg.STARK_SOFT, -1)
            cv2.addWeighted(overlay, 0.04 * intensity, final, 1.0 - 0.04 * intensity, 0, final)
            self.hud_flash *= 0.88
        else:
            self.hud_flash = 0.0

        return final

    # -----------------------------------------------------
    # Keys
    # -----------------------------------------------------
    def handle_key(self, key):
        if key == ord('q'):
            return False
        elif key == ord('s'):
            self.save_scene()
        elif key == ord('l'):
            self.load_scene()
        elif key == ord('x'):
            if self.selected_obj_index != -1 and 0 <= self.selected_obj_index < len(self.objects_3d):
                self.objects_3d.pop(self.selected_obj_index)
                self.selected_obj_index = -1
                self.drag_obj_index = -1
        elif key == ord('c'):
            self.objects_3d.clear()
            self.selected_obj_index = -1
            self.drag_obj_index = -1
        elif key == ord('b'):
            self.black_bg_mode = not self.black_bg_mode
            print(f"🎛️ Modo visual: {'BLACK BG' if self.black_bg_mode else 'NORMAL'}")
        return True

    # -----------------------------------------------------
    # Main loop
    # -----------------------------------------------------
    def run(self):
        while self.cap.isOpened():
            ok, frame = self.cap.read()
            if not ok:
                break

            t = time.time()

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            self.process_face(rgb_frame)
            R = self.get_scene_rotation()

            hand_data = self.process_hands(rgb_frame, w, h)
            primary_hand = hand_data[0] if hand_data else None

            self.handle_interaction(primary_hand, hand_data, R, w, h)

            layers = self.render_scene(frame, primary_hand, R, t)
            final = self.compose_final(frame, layers)

            cv2.imshow(self.cfg.WINDOW_NAME, final)

            key = cv2.waitKey(1) & 0xFF
            if not self.handle_key(key):
                break

        self.cap.release()
        cv2.destroyAllWindows()


# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == "__main__":
    try:
        print("✅ DEXTER v14.0: CLEAN ARCHITECTURE EDITION ONLINE")
        app = DexterApp()
        app.run()
    except Exception as e:
        print(f"❌ ERRO: {e}")