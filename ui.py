import cv2
import math

def point_in_rect(pt, rect):
    x, y = pt
    rx, ry, rw, rh = rect
    return rx <= x <= rx + rw and ry <= y <= ry + rh

class UIOverlay:
    def __init__(self, cfg):
        self.cfg = cfg

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
        
        for d in [-40, 14]: # Linhas em cruz
            cv2.line(canvas, (x + d, y), (x + d + 26, y), self.cfg.STARK_SOFT, 1, cv2.LINE_AA)
            cv2.line(canvas, (x, y + d), (x, y + d + 26), self.cfg.STARK_SOFT, 1, cv2.LINE_AA)

    def draw_selected_hud(self, canvas, center_px, bbox, t):
        x, y = center_px
        bx, by, bw, bh = bbox
        cv2.rectangle(canvas, (bx - 10, by - 10), (bx + bw + 10, by + bh + 10), self.cfg.GREEN_SOFT, 1, cv2.LINE_AA)
        orbit = 24 + int(5 * math.sin(t * 4.0))
        cv2.circle(canvas, (x, y), orbit, self.cfg.GREEN_SOFT, 1, cv2.LINE_AA)
        cv2.putText(canvas, "TRACK LOCK", (bx, by - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.cfg.GREEN_SOFT, 1, cv2.LINE_AA)

    def draw_spawn_effect(self, canvas, center_px, elapsed):
        if elapsed > 0.7: return
        x, y = center_px
        progress = elapsed / 0.7
        radius = int(12 + progress * 50)
        alpha_color = self.cfg.STARK_EDGE if progress < 0.5 else self.cfg.CYAN_SOFT
        cv2.circle(canvas, (x, y), radius, alpha_color, 1, cv2.LINE_AA)

    def draw_radar_hud(self, canvas, t):
        cx, cy = 130, 170
        for r in [65, 42, 20]:
            cv2.circle(canvas, (cx, cy), r, self.cfg.STARK_SOFT, 1, cv2.LINE_AA)
        angle = t * 1.8
        cv2.line(canvas, (cx, cy), (int(cx + math.cos(angle) * 65), int(cy + math.sin(angle) * 65)), self.cfg.STARK_EDGE, 1, cv2.LINE_AA)
        cv2.putText(canvas, "SCAN", (cx - 18, cy + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.cfg.STARK_EDGE, 1, cv2.LINE_AA)

    def draw_panels(self, img, objects_3d, selected_index, black_bg_mode):
        h, w, _ = img.shape
        # Top Panel
        x0, y0, panel_w = w - 290, 80, 270
        cv2.rectangle(img, (x0, y0), (x0 + panel_w, h - 40), self.cfg.PANEL_BG, -1)
        cv2.rectangle(img, (x0, y0), (x0 + panel_w, h - 40), self.cfg.STARK_SOFT, 1, cv2.LINE_AA)
        cv2.putText(img, "SYSTEM PANEL", (x0 + 15, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.cfg.STARK_EDGE, 1, cv2.LINE_AA)
        
        lines = [
            f"Objects: {len(objects_3d)}", f"Selected: {selected_index if selected_index != -1 else 'NONE'}",
            f"View mode: {'BLACK BG' if black_bg_mode else 'NORMAL'}", "", "Controls:",
            "- pinch = select/drag", "- 2 hands = scale/rotate", "- B = toggle bg", "- C = clear scene", "- Q = quit"
        ]
        
        yy = y0 + 55
        for line in lines:
            cv2.putText(img, line, (x0 + 15, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.43, self.cfg.WHITE_SOFT, 1, cv2.LINE_AA)
            yy += 22

        # Bottom HUD
        cv2.putText(img, self.cfg.HUD_TITLE, (20, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.56, self.cfg.WHITE_SOFT, 1, cv2.LINE_AA)
