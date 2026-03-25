import cv2
import mediapipe as mp
import numpy as np
import json
import math
import time

# =========================================================
# DEXTER v12.0 - BLUE GHOST EDITION (Aprimorado)
# =========================================================

# -----------------------------
# Inicialização
# -----------------------------
try:
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    print("✅ DEXTER v12.0: BLUE GHOST EDITION ONLINE (Aprimorado)")
except Exception as e:
    print(f"❌ ERRO NA INICIALIZAÇÃO: {e}")
    raise SystemExit

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Não foi possível acessar a câmera.")
    raise SystemExit

cv2.namedWindow("DEXTER v12.0 - Enhanced")

# -----------------------------
# Configurações visuais
# -----------------------------
STARK_BLUE = (255, 160, 60)      # BGR
STARK_EDGE = (255, 230, 150)
STARK_SOFT = (180, 120, 50)
WHITE_SOFT = (220, 220, 220)
RED_SOFT = (80, 80, 255)
GREEN_SOFT = (120, 255, 120)

# Cores aprimoradas para HUD
HUD_PRIMARY_COLOR = (255, 180, 80) # Azul mais vibrante
HUD_SECONDARY_COLOR = (200, 100, 40) # Azul mais escuro
HUD_ACCENT_COLOR = (0, 255, 255) # Ciano para destaque
HUD_TEXT_COLOR = (255, 255, 255) # Branco
HUD_WARNING_COLOR = (0, 0, 255) # Vermelho

# Botões (mantidos para compatibilidade, mas aprimorados visualmente)
CREATE_BTN = (40, 40, 70, 70)
DELETE_BTN = (130, 40, 70, 70)

HUD_TITLE = "DEXTER v12.0 // BLUE GHOST // ENHANCED"

# -----------------------------
# Estado global
# -----------------------------
objects_3d = []
selected_obj_index = -1
drag_obj_index = -1

face_offset_x = 0.0
face_offset_y = 0.0
P_SENSITIVITY = 0.8
FACE_SMOOTH = 0.15

last_hand_pos3d = None
HAND_SMOOTH = 0.28

prev_grabbing = False
grab_started = False
grab_ended = False

base_dist_scaling = None
base_scale_obj = None

# profundidade/rotação
BASE_ROT_X = 0.15
BASE_ROT_Y = -0.3

SCENE_FILE = "dexter_scene_v12.json"

# -----------------------------
# Funções matemáticas
# -----------------------------
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

def safe_normal(v):
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return v / norm

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
        current[2] * (1.0 - alpha) + target[2] * alpha
    ]

# -----------------------------
# Geometria
# -----------------------------
def get_cube_geometry(obj):
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

def get_object_center_projected(obj, R, w, h):
    center = np.array(obj["pos"], dtype=np.float32)
    rotated = R @ center
    return project(rotated, w, h)

# -----------------------------
# Persistência
# -----------------------------
def save_scene():
    serializable = []
    for obj in objects_3d:
        serializable.append({
            "type": obj.get("type", "cube"),
            "pos": [float(v) for v in obj["pos"]],
            "scale": float(obj["scale"]),
            "color": list(obj["color"])
        })
    with open(SCENE_FILE, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

def load_scene():
    global objects_3d, selected_obj_index, drag_obj_index
    try:
        with open(SCENE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        loaded = []
        for item in data:
            loaded.append({
                "type": item.get("type", "cube"),
                "pos": list(item.get("pos", [0.0, 0.0, 0.0])),
                "scale": float(item.get("scale", 0.4)),
                "color": tuple(item.get("color", STARK_BLUE)),
                "bbox": (0, 0, 0, 0)
            })
        objects_3d = loaded
        selected_obj_index = -1
        drag_obj_index = -1
        print("💾 Cena carregada.")
    except FileNotFoundError:
        print("⚠️ Arquivo de cena não encontrado.")
    except Exception as e:
        print(f"❌ Erro ao carregar cena: {e}")

# -----------------------------
# UI Aprimorada
# -----------------------------
def draw_circular_button(img, center, radius, label, color, thickness=2, fill=False, alpha=0.6):
    overlay = img.copy()
    if fill:
        cv2.circle(overlay, center, radius, color, -1)
    cv2.circle(overlay, center, radius, color, thickness, cv2.LINE_AA)
    
    # Adicionar texto com sombra para melhor legibilidade
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
    text_x = center[0] - text_size[0] // 2
    text_y = center[1] + text_size[1] // 2
    cv2.putText(overlay, label, (text_x + 1, text_y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA) # Sombra
    cv2.putText(overlay, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, HUD_TEXT_COLOR, 1, cv2.LINE_AA)
    
    img = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)
    return img

def draw_progress_bar(img, rect, progress, color, bg_color=(50,50,50), alpha=0.6):
    x, y, w, h = rect
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), bg_color, -1) # Fundo da barra
    cv2.rectangle(overlay, (x, y), (x + int(w * progress), y + h), color, -1) # Preenchimento
    cv2.rectangle(overlay, (x, y), (x + w, y + h), HUD_TEXT_COLOR, 1, cv2.LINE_AA) # Borda
    img = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)
    return img

def draw_dynamic_text(img, text, pos, font_scale=0.5, color=HUD_TEXT_COLOR, thickness=1, alpha=0.7):
    overlay = img.copy()
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    text_bg_rect = (pos[0] - 5, pos[1] - text_size[1] - 5, text_size[0] + 10, text_size[1] + 10)
    cv2.rectangle(overlay, (text_bg_rect[0], text_bg_rect[1]), 
                  (text_bg_rect[0] + text_bg_rect[2], text_bg_rect[1] + text_bg_rect[3]), 
                  (0,0,0), -1) # Fundo escuro para o texto
    cv2.putText(overlay, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
    img = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)
    return img

def draw_hud_enhanced(final, selected_idx, w, h):
    # Título principal
    draw_dynamic_text(final, HUD_TITLE, (20, h - 20), font_scale=0.6, color=HUD_PRIMARY_COLOR)

    # Botões circulares no canto superior esquerdo
    final = draw_circular_button(final, (75, 75), 30, "CRIAR", HUD_PRIMARY_COLOR)
    final = draw_circular_button(final, (165, 75), 30, "DEL", HUD_WARNING_COLOR)

    # Status de objetos
    status_obj = f"OBJETOS: {len(objects_3d)}"
    draw_dynamic_text(final, status_obj, (w - 170, 30), font_scale=0.5, color=HUD_TEXT_COLOR)

    # Objeto selecionado
    sel_text = f"SELECIONADO: {selected_idx if selected_idx != -1 else 'NENHUM'}"
    draw_dynamic_text(final, sel_text, (w - 220, 55), font_scale=0.45, color=HUD_ACCENT_COLOR)

    # Dicas de teclado
    tips = "Q: Sair | S: Salvar | L: Carregar | X: Deletar | C: Limpar"
    draw_dynamic_text(final, tips, (20, h - 45), font_scale=0.42, color=WHITE_SOFT)

    # Exemplo de barra de progresso (simulada)
    current_time = time.time()
    progress_val = (math.sin(current_time * 0.5) + 1) / 2 # Simula um progresso de 0 a 1
    final = draw_progress_bar(final, (w - 200, h - 30, 180, 15), progress_val, HUD_ACCENT_COLOR)
    draw_dynamic_text(final, "PROCESSANDO...", (w - 195, h - 35), font_scale=0.35, color=HUD_TEXT_COLOR)

    return final

def point_in_circular_button(pt, center, radius):
    px, py = pt
    cx, cy = center
    return math.hypot(px - cx, py - cy) <= radius

# -----------------------------
# Grid / Eixos
# -----------------------------
def draw_world_grid(canvas, R, w, h):
    grid_color = (80, 60, 25)
    for gx in np.linspace(-1.5, 1.5, 9):
        p1 = np.array([gx, 0.9, -0.8], dtype=np.float32)
        p2 = np.array([gx, 0.9,  0.8], dtype=np.float32)
        rp1 = R @ p1
        rp2 = R @ p2
        cv2.line(canvas, project(rp1, w, h), project(rp2, w, h), grid_color, 1, cv2.LINE_AA)

    for gz in np.linspace(-0.8, 0.8, 9):
        p1 = np.array([-1.5, 0.9, gz], dtype=np.float32)
        p2 = np.array([ 1.5, 0.9, gz], dtype=np.float32)
        rp1 = R @ p1
        rp2 = R @ p2
        cv2.line(canvas, project(rp1, w, h), project(rp2, w, h), grid_color, 1, cv2.LINE_AA)

def draw_axes(canvas, R, w, h):
    origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    x_axis = np.array([0.5, 0.0, 0.0], dtype=np.float32)
    y_axis = np.array([0.0, -0.5, 0.0], dtype=np.float32)
    z_axis = np.array([0.0, 0.0, 0.5], dtype=np.float32)

    ro = R @ origin
    rx = R @ x_axis
    ry = R @ y_axis
    rz = R @ z_axis

    po = project(ro, w, h)
    px = project(rx, w, h)
    py = project(ry, w, h)
    pz = project(rz, w, h)

    cv2.line(canvas, po, px, (80, 80, 255), 2, cv2.LINE_AA)   # X
    cv2.line(canvas, po, py, (80, 255, 80), 2, cv2.LINE_AA)   # Y
    cv2.line(canvas, po, pz, (255, 80, 80), 2, cv2.LINE_AA)   # Z

# -----------------------------
# Seleção
# -----------------------------
def pick_nearest_object(hand_px, R, w, h, radius=80):
    best_idx = -1
    best_dist = 999999.0

    for i, obj in enumerate(objects_3d):
        cx, cy = get_object_center_projected(obj, R, w, h)
        d = math.hypot(hand_px[0] - cx, hand_px[1] - cy)
        if d < radius and d < best_dist:
            best_dist = d
            best_idx = i

    return best_idx

# =========================================================
# LOOP PRINCIPAL
# =========================================================
while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    canvas_grid = np.zeros_like(frame)
    canvas_faces = np.zeros_like(frame)
    canvas_wire = np.zeros_like(frame)
    canvas_fx = np.zeros_like(frame)

    # -------------------------------------
    # 1) Face tracking com smoothing
    # -------------------------------------
    res_face = face_mesh.process(rgb_frame)
    if res_face.multi_face_landmarks:
        face_landmarks = res_face.multi_face_landmarks[0]
        center_eye = face_landmarks.landmark[168]

        target_x = ((center_eye.x) * 2.0 - 1.0) * P_SENSITIVITY
        target_y = ((center_eye.y) * 2.0 - 1.0) * P_SENSITIVITY

        face_offset_x = smooth_value(face_offset_x, target_x, FACE_SMOOTH)
        face_offset_y = smooth_value(face_offset_y, target_y, FACE_SMOOTH)

    # Matriz da cena já disponível para seleção e render
    R = rot_matrix_x(BASE_ROT_X + face_offset_y) @ rot_matrix_y(BASE_ROT_Y - face_offset_x)

    # -------------------------------------
    # 2) Hand tracking
    # -------------------------------------
    res_hands = hands.process(rgb_frame)
    hand_data = []

    if res_hands.multi_hand_landmarks:
        for hand_lms in res_hands.multi_hand_landmarks:
            idx = hand_lms.landmark[8] # Ponta do indicador
            thumb = hand_lms.landmark[4] # Ponta do polegar
            wrist = hand_lms.landmark[0] # Punho
            mid_base = hand_lms.landmark[9] # Base do dedo médio

            # Coordenadas 2D da ponta do indicador
            idx_x, idx_y = int(idx.x * w), int(idx.y * h)

            # Distância entre polegar e indicador para detectar 'pinça' (grab)
            dist_thumb_idx = math.hypot(idx.x - thumb.x, idx.y - thumb.y)
            grabbing = dist_thumb_idx < 0.06 # Limiar para considerar 'pinça'

            # Posição 3D da mão (usando a base do dedo médio como referência)
            hand_pos3d_target = np.array([mid_base.x - 0.5, -(mid_base.y - 0.5), mid_base.z], dtype=np.float32)
            hand_pos3d_target[2] = clamp(hand_pos3d_target[2] * 2.0 + 0.5, -0.9, 0.9)

            last_hand_pos3d = smooth_vec3(last_hand_pos3d, hand_pos3d_target, HAND_SMOOTH)
            current_hand_pos3d = np.array(last_hand_pos3d, dtype=np.float32)

            # -------------------------------------
            # Lógica de Interação Aprimorada
            # -------------------------------------
            if grabbing and not prev_grabbing: # Começou a agarrar
                grab_started = True
                # Tentar selecionar um objeto
                selected_obj_index = pick_nearest_object((idx_x, idx_y), R, w, h, radius=50) # Raio menor para seleção precisa
                if selected_obj_index != -1:
                    drag_obj_index = selected_obj_index
                    # Calcular offset inicial para arrastar
                    obj_center_proj_x, obj_center_proj_y = get_object_center_projected(objects_3d[drag_obj_index], R, w, h)
                    objects_3d[drag_obj_index]["drag_offset"] = (idx_x - obj_center_proj_x, idx_y - obj_center_proj_y)
                else:
                    # Verificar cliques em botões da HUD
                    if point_in_circular_button((idx_x, idx_y), (75, 75), 30): # Botão CRIAR
                        objects_3d.append({"type": "cube", "pos": [0.0, 0.0, 0.0], "scale": 0.4, "color": STARK_BLUE, "bbox": (0,0,0,0)})
                        print("➕ Objeto criado!")
                    elif point_in_circular_button((idx_x, idx_y), (165, 75), 30): # Botão DELETAR
                        if selected_obj_index != -1 and 0 <= selected_obj_index < len(objects_3d):
                            objects_3d.pop(selected_obj_index)
                            selected_obj_index = -1
                            drag_obj_index = -1
                            print("➖ Objeto deletado!")

            elif not grabbing and prev_grabbing: # Terminou de agarrar
                grab_ended = True
                drag_obj_index = -1

            prev_grabbing = grabbing

            if drag_obj_index != -1: # Arrastando objeto
                obj = objects_3d[drag_obj_index]
                drag_offset_x, drag_offset_y = obj["drag_offset"]
                
                # Inverter a projeção para mover o objeto em 3D
                # Esta é uma simplificação, uma inversão de projeção completa é complexa
                # Para um efeito mais realista, precisaríamos de uma matriz de projeção e visão
                # Por enquanto, vamos mover o objeto no plano XY da tela e ajustar Z levemente
                
                # Mapear a posição da mão para um espaço 3D relativo à câmera
                # Assumimos que a mão está em um plano a uma certa distância da câmera
                # Ajuste a sensibilidade conforme necessário
                new_obj_x = (idx_x - w * 0.5) / (w * 0.5) / P_SENSITIVITY
                new_obj_y = (idx_y - h * 0.5) / (h * 0.5) / P_SENSITIVITY
                
                # Manter a profundidade relativa do objeto selecionado
                current_obj_z = obj["pos"][2]
                
                # Ajustar a posição do objeto
                obj["pos"][0] = smooth_value(obj["pos"][0], new_obj_x, 0.3)
                obj["pos"][1] = smooth_value(obj["pos"][1], -new_obj_y, 0.3) # Inverter Y para corresponder à tela
                # obj["pos"][2] = smooth_value(obj["pos"][2], current_obj_z, 0.1) # Manter Z ou ajustar com outro gesto

            # Exibir pontos de referência da mão para depuração
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS, 
                                   mp_draw.DrawingSpec(color=HUD_ACCENT_COLOR, thickness=2, circle_radius=2),
                                   mp_draw.DrawingSpec(color=HUD_PRIMARY_COLOR, thickness=2, circle_radius=2))

    # -------------------------------------
    # 3) Renderização de objetos 3D
    # -------------------------------------
    all_faces_to_draw = []
    for obj_idx, obj in enumerate(objects_3d):
        verts, faces = get_cube_geometry(obj)
        
        # Aplicar rotação da cena aos vértices do objeto
        rotated_verts = (R @ (verts - np.array(obj["pos"]))).T + np.array(obj["pos"])
        rotated_verts = rotated_verts.T

        projected_verts = []
        for v in rotated_verts:
            projected_verts.append(project(v, w, h))

        # Calcular a normal da face para iluminação (simplificado)
        # Usar a normal do primeiro vértice para simplificar
        # Isso é uma simplificação e não é fisicamente preciso para iluminação
        light_dir = safe_normal(np.array([0.0, 0.0, 1.0])) # Luz vindo da frente

        obj_bbox_min_x, obj_bbox_min_y = w, h
        obj_bbox_max_x, obj_bbox_max_y = 0, 0

        for face_indices in faces:
            face_pts = np.array([projected_verts[i] for i in face_indices], dtype=np.int32)
            
            # Calcular centro da face para normal
            face_center_3d = np.mean([verts[i] for i in face_indices], axis=0)
            # Vetor normal simplificado (apenas para fins de demonstração)
            face_normal_3d = safe_normal(face_center_3d - np.array(obj["pos"])) # Vetor do centro do objeto para o centro da face
            
            # Ajustar a normal para a rotação da cena
            rotated_face_normal = R @ face_normal_3d
            
            intensity = max(0.2, np.dot(rotated_face_normal, light_dir)) # Iluminação básica

            all_faces_to_draw.append({"pts": face_pts, "intensity": intensity, "obj_idx": obj_idx})

            # Atualizar bounding box para seleção
            min_x, min_y = np.min(face_pts, axis=0)
            max_x, max_y = np.max(face_pts, axis=0)
            obj_bbox_min_x = min(obj_bbox_min_x, min_x)
            obj_bbox_min_y = min(obj_bbox_min_y, min_y)
            obj_bbox_max_x = max(obj_bbox_max_x, max_x)
            obj_bbox_max_y = max(obj_bbox_max_y, max_y)
        
        objects_3d[obj_idx]["bbox"] = (obj_bbox_min_x, obj_bbox_min_y, 
                                        obj_bbox_max_x - obj_bbox_min_x, obj_bbox_max_y - obj_bbox_min_y)

    # Desenhar faces preenchidas primeiro para oclusão
    for face in sorted(all_faces_to_draw, key=lambda f: f["intensity"]):
        obj_idx = face["obj_idx"]
        selected = (obj_idx == selected_obj_index)

        base_color = np.array(objects_3d[obj_idx]["color"], dtype=np.float32)
        if selected:
            base_color = np.array(HUD_ACCENT_COLOR, dtype=np.float32) # Cor de destaque para selecionado

        shaded = tuple(int(c * face["intensity"]) for c in base_color)

        cv2.fillConvexPoly(canvas_faces, face["pts"], shaded)

    # Desenhar wireframe
    for face in all_faces_to_draw:
        obj_idx = face["obj_idx"]
        selected = (obj_idx == selected_obj_index)

        edge_color = HUD_ACCENT_COLOR if selected else HUD_PRIMARY_COLOR
        edge_thickness = 2 if selected else 1
        cv2.polylines(canvas_wire, [face["pts"]], True, edge_color, edge_thickness, cv2.LINE_AA)

    # highlight bbox do selecionado
    if selected_obj_index != -1 and 0 <= selected_obj_index < len(objects_3d):
        bx, by, bw, bh = objects_3d[selected_obj_index]["bbox"]
        cv2.rectangle(canvas_wire, (bx - 8, by - 8), (bx + bw + 8, by + bh + 8),
                      HUD_ACCENT_COLOR, 1, cv2.LINE_AA)
        cv2.putText(canvas_wire, "SELECIONADO", (bx, by - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, HUD_ACCENT_COLOR, 1, cv2.LINE_AA)

    # -------------------------------------
    # 6) Composição final (efeito holográfico aprimorado)
    # -------------------------------------
    # Desenhar grid de fundo
    draw_world_grid(canvas_grid, R, w, h)

    glow_grid = cv2.GaussianBlur(canvas_grid, (11, 11), 0)
    glow_faces = cv2.GaussianBlur(canvas_faces, (25, 25), 0)
    glow_wire = cv2.GaussianBlur(canvas_wire, (7, 7), 0)
    glow_fx = cv2.GaussianBlur(canvas_fx, (21, 21), 0)

    hologram = cv2.addWeighted(canvas_grid, 0.7, glow_grid, 1.0, 0)
    hologram = cv2.addWeighted(hologram, 1.0, canvas_fx, 0.5, 0)
    hologram = cv2.addWeighted(hologram, 1.0, glow_fx, 0.8, 0)
    hologram = cv2.addWeighted(hologram, 1.0, canvas_faces, 0.75, 0)
    hologram = cv2.addWeighted(hologram, 1.0, glow_faces, 1.25, 0)
    hologram = cv2.addWeighted(hologram, 1.0, canvas_wire, 1.0, 0)
    hologram = cv2.addWeighted(hologram, 1.0, glow_wire, 1.6, 0)

    mask = cv2.cvtColor(canvas_faces + canvas_grid + canvas_wire, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    alpha = cv2.merge([mask, mask, mask]).astype(np.float32) / 255.0

    final = (frame.astype(np.float32) * (1.0 - alpha * 0.45) + \
             hologram.astype(np.float32) * (alpha * 0.95)).astype(np.uint8)

    final = draw_hud_enhanced(final, selected_obj_index, w, h)

    cv2.imshow("DEXTER v12.0 - Enhanced", final)

    # -------------------------------------
    # 7) Teclas
    # -------------------------------------
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == ord("s"):
        save_scene()
        print("💾 Cena salva.")
    elif key == ord("l"):
        load_scene()
    elif key == ord("x"):
        if selected_obj_index != -1 and 0 <= selected_obj_index < len(objects_3d):
            objects_3d.pop(selected_obj_index)
            selected_obj_index = -1
            drag_obj_index = -1
    elif key == ord("c"):
        objects_3d.clear()
        selected_obj_index = -1
        drag_obj_index = -1

cap.release()
cv2.destroyAllWindows() 
