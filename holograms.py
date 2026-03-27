import numpy as np
import cv2

# --- Utils Matemáticos ---
def clamp(v, lo, hi): return max(lo, min(hi, v))
def smooth_vec3(c, t, a): return list(t) if c is None else [c[0]*(1-a)+t[0]*a, c[1]*(1-a)+t[1]*a, c[2]*(1-a)+t[2]*a]
def safe_normal(v): n = np.linalg.norm(v); return v/n if n > 1e-6 else np.array([0., 0., 1.], dtype=np.float32)

def project(pt3d, w, h, fov=0.8):
    x, y, z = pt3d
    denom = max(0.01, 1.0 + z * fov)
    return int((x / denom) * w * 0.5 + w * 0.5), int((y / denom) * h * 0.5 + h * 0.5)

def rot_matrix_x(a): return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]], dtype=np.float32)
def rot_matrix_y(a): return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]], dtype=np.float32)
def rot_matrix_z(a): return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]], dtype=np.float32)


# Hologram Engine 
class HologramEngine:
    def __init__(self, cfg):
        self.cfg = cfg

    def get_geometry(self, obj):
        s = obj["scale"] / 2.0
        px, py, pz = obj["pos"]
        R = rot_matrix_y(obj.get("rot_y", 0.0)) @ rot_matrix_x(obj.get("rot_x", 0.0)) @ rot_matrix_z(obj.get("rot_z", 0.0))
        
        if obj.get("type") == "core":
            verts = np.array([[0, s*1.5, 0], [0, -s*1.5, 0], [s*1.5, 0, 0], [-s*1.5, 0, 0], [0, 0, s*1.5], [0, 0, -s*1.5]], dtype=np.float32)
            faces = [[0,2,4], [0,4,3], [0,3,5], [0,5,2], [1,4,2], [1,3,4], [1,5,3], [1,2,5]]
        else: # Default (Cube)
            verts = np.array([[-s,-s,-s], [s,-s,-s], [s,s,-s], [-s,s,-s], [-s,-s,s], [s,-s,s], [s,s,s], [-s,s,s]], dtype=np.float32)
            faces = [[0,1,2,3], [4,5,6,7], [0,1,5,4], [2,3,7,6], [0,3,7,4], [1,2,6,5]]

        verts = np.array([R @ v for v in verts], dtype=np.float32) + np.array([px, py, pz], dtype=np.float32)
        return verts, faces

    def get_object_center_projected(self, obj, R, w, h):
        return project(R @ np.array(obj["pos"], dtype=np.float32), w, h, self.cfg.FOV)

    def compose_final(self, base_frame, layers, hud_flash, black_bg_mode):
        c_grid, c_faces, c_wire, c_fx, c_hud = layers
        holo = cv2.addWeighted(c_grid, 0.7, cv2.GaussianBlur(c_grid, (11, 11), 0), 1.0, 0)
        holo = cv2.addWeighted(holo, 1.0, c_faces, 0.78, 0)
        holo = cv2.addWeighted(holo, 1.0, cv2.GaussianBlur(c_faces, (25, 25), 0), 1.25, 0)
        holo = cv2.addWeighted(holo, 1.0, c_wire, 1.0, 0)
        holo = cv2.addWeighted(holo, 1.0, cv2.GaussianBlur(c_wire, (7, 7), 0), 1.45, 0)
        holo = cv2.addWeighted(holo, 1.0, c_hud, 0.9, 0)

        mask = cv2.threshold(cv2.cvtColor(c_faces + c_grid + c_wire + c_hud, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)[1]
        alpha = cv2.merge([mask, mask, mask]).astype(np.float32) / 255.0
        bg = np.zeros_like(base_frame) if black_bg_mode else base_frame
        
        final = (bg.astype(np.float32) * (1.0 - alpha * 0.46) + holo.astype(np.float32) * (alpha * 0.96)).astype(np.uint8)
        
        if hud_flash > 0.01:
            overlay = final.copy()
            cv2.rectangle(overlay, (0, 0), (final.shape[1], final.shape[0]), self.cfg.STARK_SOFT, -1)
            cv2.addWeighted(overlay, 0.04 * hud_flash, final, 1.0 - 0.04 * hud_flash, 0, final)
            
        return final
