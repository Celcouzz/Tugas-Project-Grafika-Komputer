from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np
import sys

try:
    from pywavefront import Wavefront
    from pywavefront.visualization import draw
    has_obj_loader = True
except ImportError:
    has_obj_loader = False
    print("Peringatan: PyWavefront tidak terinstal. Model .obj tidak akan dimuat. Install dengan 'pip install pywavefront'")

# --- Variabel Global dan State Aplikasi ---
window_width, window_height = 800, 600
mode = '2D'
objects_2d = []
clipping_window = []
selected_index_2d = -1
current_shape_2d = 'rect'
current_color_2d = [1.0, 0.0, 0.0]
current_thickness_2d = 2
drawing_start_pos = None
is_dragging_2d = False
current_transform_mode = 'none'
window_offset = [0, 0]
current_mouse_pos = [0, 0]
camera_zoom = -15
obj_3d_pos = [0, 0, 0]
obj_3d_rotation = [0, 0]
is_mouse_dragging_3d = False
last_mouse_pos = [0, 0]
scene_3d = None

# <<< BARU: Variabel state untuk mengubah ukuran window >>>
is_resizing_window = False
resizing_handle_index = -1  # 0:BL, 1:TR, 2:BR, 3:TL
HANDLE_SIZE = 5  # Ukuran handle dalam piksel

cube_vertices = [[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]
cube_faces = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[1,2,6,5],[0,3,7,4]]
cube_normals = [[0,0,-1],[0,0,1],[0,-1,0],[0,1,0],[1,0,0],[-1,0,0]]

# --- Implementasi Algoritma dan Fungsi Bantuan ---

def liang_barsky_clip(x1, y1, x2, y2, xmin, ymin, xmax, ymax):
    dx, dy = x2 - x1, y2 - y1
    p = [-dx, dx, -dy, dy]
    q = [x1 - xmin, xmax - x1, y1 - ymin, ymax - y1]
    t0, t1 = 0.0, 1.0
    for i in range(4):
        if p[i] == 0:
            if q[i] < 0: return None
        else:
            r = q[i] / p[i]
            if p[i] < 0: t0 = max(t0, r)
            else: t1 = min(t1, r)
    if t0 > t1: return None
    return (x1 + t0 * dx, y1 + t0 * dy, x1 + t1 * dx, y1 + t1 * dy)

def sutherland_hodgman_clip(subject_polygon, clip_boundary):
    xmin, ymin, xmax, ymax = clip_boundary
    clip_edges = [('left', xmin), ('right', xmax), ('bottom', ymin), ('top', ymax)]
    clipped_polygon = list(subject_polygon)
    for edge_type, edge_coord in clip_edges:
        input_list = list(clipped_polygon)
        clipped_polygon.clear()
        if not input_list: break
        s = input_list[-1]
        for e in input_list:
            s_inside, e_inside = False, False
            if edge_type == 'left': s_inside, e_inside = s[0] >= edge_coord, e[0] >= edge_coord
            elif edge_type == 'right': s_inside, e_inside = s[0] <= edge_coord, e[0] <= edge_coord
            elif edge_type == 'bottom': s_inside, e_inside = s[1] >= edge_coord, e[1] >= edge_coord
            elif edge_type == 'top': s_inside, e_inside = s[1] <= edge_coord, e[1] <= edge_coord
            if e_inside:
                if not s_inside:
                    x, y = 0, 0
                    if edge_type in ['left', 'right']:
                        x = edge_coord
                        y = s[1] + (e[1] - s[1]) * (x - s[0]) / (e[0] - s[0]) if (e[0] - s[0]) != 0 else e[1]
                    else:
                        y = edge_coord
                        x = s[0] + (e[0] - s[0]) * (y - s[1]) / (e[1] - s[1]) if (e[1] - s[1]) != 0 else e[0]
                    clipped_polygon.append((x, y))
                clipped_polygon.append(e)
            elif s_inside:
                x, y = 0, 0
                if edge_type in ['left', 'right']:
                    x = edge_coord
                    y = s[1] + (e[1] - s[1]) * (x - s[0]) / (e[0] - s[0]) if (e[0] - s[0]) != 0 else e[1]
                else:
                    y = edge_coord
                    x = s[0] + (e[0] - s[0]) * (y - s[1]) / (e[1] - s[1]) if (e[1] - s[1]) != 0 else e[0]
                clipped_polygon.append((x, y))
            s = e
    return clipped_polygon

def find_object_at_pos(x, y):
    if len(clipping_window) == 2 and find_handle_at_pos(x, y) != -1:
        return -1
    for i in range(len(objects_2d) - 1, -1, -1):
        obj = objects_2d[i]
        coords = get_transformed_vertices(obj)
        if not coords: continue
        min_x, max_x = min(c[0] for c in coords), max(c[0] for c in coords)
        min_y, max_y = min(c[1] for c in coords), max(c[1] for c in coords)
        if min_x - 5 <= x <= max_x + 5 and min_y - 5 <= y <= max_y + 5:
            return i
    return -1

def get_transformed_vertices(obj):
    if not obj['coords']: return []
    t = obj['transform']
    center_x = sum(c[0] for c in obj['coords']) / len(obj['coords'])
    center_y = sum(c[1] for c in obj['coords']) / len(obj['coords'])
    transformed_coords = []
    for x, y in obj['coords']:
        temp_x, temp_y = x - center_x, y - center_y
        scaled_x, scaled_y = temp_x * t['scale'], temp_y * t['scale']
        angle_rad = np.radians(t['angle'])
        rotated_x = scaled_x * np.cos(angle_rad) - scaled_y * np.sin(angle_rad)
        rotated_y = scaled_x * np.sin(angle_rad) + scaled_y * np.cos(angle_rad)
        final_x = rotated_x + center_x + t['tx']
        final_y = rotated_y + center_y + t['ty']
        transformed_coords.append((final_x, final_y))
    return transformed_coords

# <<< BARU: Fungsi-fungsi helper untuk resize window >>>
def get_clipping_window_handles():
    if len(clipping_window) != 2: return []
    p1, p2 = clipping_window[0], clipping_window[1]
    xmin, xmax = min(p1[0], p2[0]), max(p1[0], p2[0])
    ymin, ymax = min(p1[1], p2[1]), max(p1[1], p2[1])
    
    # [Bottom-Left, Top-Right, Bottom-Right, Top-Left]
    return [(xmin, ymin), (xmax, ymax), (xmax, ymin), (xmin, ymax)]

def find_handle_at_pos(x, y):
    handles = get_clipping_window_handles()
    for i, handle in enumerate(handles):
        handle_x = handle[0] + window_offset[0]
        handle_y = handle[1] + window_offset[1]
        if abs(x - handle_x) <= HANDLE_SIZE and abs(y - handle_y) <= HANDLE_SIZE:
            return i
    return -1

# --- Fungsi Menggambar ---
def draw_hud():
    glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity()
    gluOrtho2D(0, window_width, 0, window_height)
    glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()
    status_text = f"Alat: {current_shape_2d.upper()} | Tebal: {current_thickness_2d}"
    glColor3f(1.0, 1.0, 1.0)
    glRasterPos2f(10, window_height - 20)
    for char in status_text: glutBitmapCharacter(GLUT_BITMAP_9_BY_15, ord(char))
    glColor3fv(current_color_2d)
    glBegin(GL_QUADS); glVertex2f(10, window_height - 40); glVertex2f(30, window_height - 40); glVertex2f(30, window_height - 25); glVertex2f(10, window_height - 25); glEnd()
    glPopMatrix(); glMatrixMode(GL_PROJECTION); glPopMatrix(); glMatrixMode(GL_MODELVIEW)

def draw_live_preview():
    if drawing_start_pos is None: return
    glColor4f(0.8, 0.8, 0.8, 0.5); glLineWidth(2); glEnable(GL_LINE_STIPPLE); glLineStipple(1, 0x00FF)
    x1, y1 = drawing_start_pos; x2, y2 = current_mouse_pos
    if current_shape_2d == 'line': glBegin(GL_LINES); glVertex2f(x1, y1); glVertex2f(x2, y2); glEnd()
    elif current_shape_2d == 'rect': glBegin(GL_LINE_LOOP); glVertex2f(x1, y1); glVertex2f(x2, y1); glVertex2f(x2, y2); glVertex2f(x1, y2); glEnd()
    elif current_shape_2d == 'ellipse':
        cx, cy, rx, ry = (x1 + x2)/2, (y1+y2)/2, abs(x2 - x1)/2, abs(y2 - y1)/2
        if rx > 0 and ry > 0:
            glBegin(GL_LINE_LOOP)
            for i in range(360): rad = np.radians(i); glVertex2f(cx + rx * np.cos(rad), cy + ry * np.sin(rad))
            glEnd()
    glDisable(GL_LINE_STIPPLE)

def draw_2d_scene():
    for i, obj in enumerate(objects_2d):
        glLineWidth(obj['thickness']); glPointSize(obj['thickness'] * 2)
        v_transformed = get_transformed_vertices(obj)
        if not v_transformed: continue

        if len(clipping_window) == 2:
            xw1, yw1 = clipping_window[0][0] + window_offset[0], clipping_window[0][1] + window_offset[1]
            xw2, yw2 = clipping_window[1][0] + window_offset[0], clipping_window[1][1] + window_offset[1]
            xmin, xmax = min(xw1, xw2), max(xw1, xw2)
            ymin, ymax = min(yw1, yw2), max(yw1, yw2)
            
            if i == selected_index_2d: glColor3f(1.0, 1.0, 0.0)
            else: glColor3f(0.0, 1.0, 0.0)

            if obj['type'] == 'line':
                p1, p2 = v_transformed
                clipped_line = liang_barsky_clip(p1[0], p1[1], p2[0], p2[1], xmin, ymin, xmax, ymax)
                if clipped_line:
                    glBegin(GL_LINES); glVertex2f(clipped_line[0], clipped_line[1]); glVertex2f(clipped_line[2], clipped_line[3]); glEnd()
            else:
                clipped_polygon = sutherland_hodgman_clip(v_transformed, (xmin, ymin, xmax, ymax))
                if clipped_polygon:
                    gl_primitive = GL_POINTS if obj['type'] == 'point' else GL_LINE_LOOP
                    glBegin(gl_primitive)
                    for vertex in clipped_polygon: glVertex2fv(vertex)
                    glEnd()
        else:
            if i == selected_index_2d: glColor3f(1.0, 1.0, 0.0)
            else: glColor3fv(obj['color'])
            glPushMatrix()
            t = obj['transform']
            center_x = sum(c[0] for c in obj['coords']) / len(obj['coords'])
            center_y = sum(c[1] for c in obj['coords']) / len(obj['coords'])
            glTranslatef(t['tx'] + center_x, t['ty'] + center_y, 0)
            glRotatef(t['angle'], 0, 0, 1); glScalef(t['scale'], t['scale'], 1)
            glTranslatef(-center_x, -center_y, 0)
            glBegin_map = {'point': GL_POINTS, 'line': GL_LINES, 'rect': GL_LINE_LOOP, 'ellipse': GL_LINE_LOOP}
            if obj['type'] in glBegin_map:
                glBegin(glBegin_map[obj['type']])
                for x, y in obj['coords']: glVertex2f(x, y)
                glEnd()
            glPopMatrix()

    if len(clipping_window) == 2:
        glColor3f(0.0, 0.8, 0.8); glLineWidth(2)
        x1, y1 = clipping_window[0][0] + window_offset[0], clipping_window[0][1] + window_offset[1]
        x2, y2 = clipping_window[1][0] + window_offset[0], clipping_window[1][1] + window_offset[1]
        glBegin(GL_LINE_LOOP); glVertex2f(x1, y1); glVertex2f(x2, y1); glVertex2f(x2, y2); glVertex2f(x1, y2); glEnd()
        
        # <<< BARU: Menggambar handle untuk resize >>>
        glColor3f(1.0, 1.0, 0.0) # Warna kuning untuk handle
        glBegin(GL_QUADS)
        for hx, hy in get_clipping_window_handles():
            glVertex2f(hx + window_offset[0] - HANDLE_SIZE, hy + window_offset[1] - HANDLE_SIZE)
            glVertex2f(hx + window_offset[0] + HANDLE_SIZE, hy + window_offset[1] - HANDLE_SIZE)
            glVertex2f(hx + window_offset[0] + HANDLE_SIZE, hy + window_offset[1] + HANDLE_SIZE)
            glVertex2f(hx + window_offset[0] - HANDLE_SIZE, hy + window_offset[1] + HANDLE_SIZE)
        glEnd()
    
    draw_live_preview()
    draw_hud()

def draw_3d_scene():
    glEnable(GL_DEPTH_TEST); glEnable(GL_LIGHTING); glEnable(GL_LIGHT0); glShadeModel(GL_SMOOTH)
    glLightfv(GL_LIGHT0, GL_POSITION, [5.0, 5.0, 5.0, 1.0]); glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0]); glLightfv(GL_LIGHT0, GL_SPECULAR, [0.5, 0.5, 0.5, 1.0])
    glMaterialfv(GL_FRONT, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0]); glMaterialf(GL_FRONT, GL_SHININESS, 50.0)
    glPushMatrix()
    glTranslatef(obj_3d_pos[0], obj_3d_pos[1], camera_zoom + obj_3d_pos[2])
    glRotatef(obj_3d_rotation[0], 1, 0, 0); glRotatef(obj_3d_rotation[1], 0, 1, 0)
    glColor3f(0.6, 0.6, 0.9)
    if scene_3d: draw(scene_3d)
    else:
        for i, face in enumerate(cube_faces):
            glBegin(GL_QUADS); glNormal3fv(cube_normals[i])
            for vertex_index in face: glVertex3fv(cube_vertices[vertex_index])
            glEnd()
    glPopMatrix(); glDisable(GL_LIGHTING); glDisable(GL_DEPTH_TEST)

def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); glLoadIdentity()
    if mode == '2D':
        reshape(window_width, window_height); draw_2d_scene()
    else:
        reshape(window_width, window_height); gluLookAt(0, 0, 5, 0, 0, 0, 0, 1, 0); draw_3d_scene()
    glutSwapBuffers()

# --- Fungsi Input Handling ---

# <<< DIPERBARUI: Kontrol geser window dipindah ke special_keyboard >>>
def keyboard(key, x, y):
    global mode, selected_index_2d, camera_zoom, current_thickness_2d, current_shape_2d, current_color_2d
    key_str = key.decode('utf-8').lower()
    if key_str == 'm':
        mode = '3D' if mode == '2D' else '2D'; selected_index_2d = -1
        print(f"Mode diubah ke: {mode}"); glutPostRedisplay(); return
    if mode == '2D':
        if key_str == '1': current_shape_2d = 'line'; print("Alat: Garis")
        elif key_str == '2': current_shape_2d = 'rect'; print("Alat: Persegi")
        elif key_str == '3': current_shape_2d = 'ellipse'; print("Alat: Elips")
        elif key_str == '4': current_shape_2d = 'point'; print("Alat: Titik")
        elif key_str == 'r': current_color_2d = [1,0,0]; print("Warna: Merah")
        elif key_str == 'g': current_color_2d = [0,1,0]; print("Warna: Hijau")
        elif key_str == 'b': current_color_2d = [0,0,1]; print("Warna: Biru")
        elif key in [b'+', b'-']:
            is_increase = key == b'+'
            if selected_index_2d != -1:
                obj = objects_2d[selected_index_2d]
                obj['thickness'] = obj['thickness'] + 1 if is_increase else max(1, obj['thickness'] - 1)
            else:
                current_thickness_2d = current_thickness_2d + 1 if is_increase else max(1, current_thickness_2d - 1)
        elif key_str == 'c': clipping_window.clear(); print("Clipping window dihapus.")
        elif selected_index_2d != -1:
            obj = objects_2d[selected_index_2d]
            if key_str == 'q': obj['transform']['angle'] += 5
            elif key_str == 'e': obj['transform']['angle'] -= 5
            elif key_str == 'z': obj['transform']['scale'] *= 1.1
            elif key_str == 'x': obj['transform']['scale'] /= 1.1
    else: # mode == '3D'
        if key_str == 'w': camera_zoom += 0.5
        elif key_str == 's': camera_zoom -= 0.5
        elif key_str == 'a': obj_3d_pos[0] -= 0.5
        elif key_str == 'd': obj_3d_pos[0] += 0.5
        elif key_str == 'r': obj_3d_pos[1] += 0.5
        elif key_str == 'f': obj_3d_pos[1] -= 0.5
    glutPostRedisplay()

# <<< BARU: Fungsi untuk menangani tombol panah >>>
def special_keyboard(key, x, y):
    if mode == '2D' and len(clipping_window) == 2:
        if key == GLUT_KEY_UP: window_offset[1] += 10
        elif key == GLUT_KEY_DOWN: window_offset[1] -= 10
        elif key == GLUT_KEY_LEFT: window_offset[0] -= 10
        elif key == GLUT_KEY_RIGHT: window_offset[0] += 10
        glutPostRedisplay()

# <<< DIPERBARUI: Mouse function sekarang bisa mendeteksi handle >>>
def mouse(button, state, x, y):
    global drawing_start_pos, selected_index_2d, is_mouse_dragging_3d, last_mouse_pos
    global is_dragging_2d, current_transform_mode, is_resizing_window, resizing_handle_index
    y = window_height - y
    if mode == '3D':
        if button == GLUT_LEFT_BUTTON: is_mouse_dragging_3d = (state == GLUT_DOWN); last_mouse_pos = [x, y]
        return
    if button == GLUT_LEFT_BUTTON:
        if state == GLUT_DOWN:
            handle_idx = find_handle_at_pos(x, y)
            if handle_idx != -1:
                is_resizing_window = True
                resizing_handle_index = handle_idx
                return

            found_index = find_object_at_pos(x, y)
            if found_index != -1:
                is_dragging_2d, selected_index_2d, last_mouse_pos = True, found_index, [x, y]
                mods = glutGetModifiers()
                if mods == GLUT_ACTIVE_CTRL: current_transform_mode = 'rotate'
                elif mods == GLUT_ACTIVE_ALT: current_transform_mode = 'scale'
                else: current_transform_mode = 'translate'
            else:
                is_dragging_2d, drawing_start_pos, selected_index_2d = False, (x, y), -1
        elif state == GLUT_UP:
            if is_resizing_window:
                is_resizing_window = False
                resizing_handle_index = -1
            elif is_dragging_2d:
                 is_dragging_2d, current_transform_mode = False, 'none'
            elif drawing_start_pos:
                end_pos = (x, y); mods = glutGetModifiers()
                if mods == GLUT_ACTIVE_SHIFT:
                    if len(clipping_window) >= 2: clipping_window.clear()
                    clipping_window.append(drawing_start_pos); clipping_window.append(end_pos)
                    print("Clipping window diatur.")
                else:
                    coords = []
                    x1,y1 = drawing_start_pos; x2,y2 = end_pos
                    if current_shape_2d == 'rect': coords = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                    elif current_shape_2d == 'line': coords = [drawing_start_pos, end_pos]
                    elif current_shape_2d == 'point': coords = [drawing_start_pos]
                    elif current_shape_2d == 'ellipse':
                        cx, cy = (x1+x2)/2, (y1+y2)/2; rx, ry = abs(x2-x1)/2, abs(y2-y1)/2
                        if rx > 0 and ry > 0: coords = [(cx + rx * np.cos(np.radians(i)), cy + ry * np.sin(np.radians(i))) for i in range(360)]
                    if coords: objects_2d.append({'type': current_shape_2d, 'coords': coords, 'color': current_color_2d[:], 'thickness': current_thickness_2d, 'transform': {'tx': 0, 'ty': 0, 'angle': 0, 'scale': 1.0}})
                drawing_start_pos = None
    glutPostRedisplay()

# <<< DIPERBARUI: Motion function sekarang bisa handle resize window >>>
def motion(x, y):
    global obj_3d_rotation, last_mouse_pos, clipping_window
    y_inverted = window_height - y
    dx, dy = x - last_mouse_pos[0], y_inverted - last_mouse_pos[1]
    
    if is_resizing_window:
        mx, my = x - window_offset[0], y_inverted - window_offset[1]
        p1, p2 = list(clipping_window[0]), list(clipping_window[1])
        
        # [0:BL, 1:TR, 2:BR, 3:TL]
        if resizing_handle_index == 0: # Bottom-Left
             p1[0], p1[1] = mx, my
        elif resizing_handle_index == 1: # Top-Right
             p2[0], p2[1] = mx, my
        elif resizing_handle_index == 2: # Bottom-Right
             p2[0], p1[1] = mx, my
        elif resizing_handle_index == 3: # Top-Left
             p1[0], p2[1] = mx, my
        
        clipping_window = [(p1[0], p1[1]), (p2[0], p2[1])]

    elif mode == '2D' and is_dragging_2d and selected_index_2d != -1:
        obj = objects_2d[selected_index_2d]
        if current_transform_mode == 'translate': obj['transform']['tx'] += dx; obj['transform']['ty'] += dy
        elif current_transform_mode == 'rotate': obj['transform']['angle'] += dx
        elif current_transform_mode == 'scale': new_scale = obj['transform']['scale'] + dy * 0.01; obj['transform']['scale'] = max(0.1, new_scale)
    
    elif mode == '3D' and is_mouse_dragging_3d:
        obj_3d_rotation[0] += dy * 0.5; obj_3d_rotation[1] += dx * 0.5

    last_mouse_pos = [x, y_inverted]; glutPostRedisplay()

def passive_motion(x, y):
    global current_mouse_pos
    current_mouse_pos = [x, window_height - y]
    if drawing_start_pos: glutPostRedisplay()

def reshape(width, height):
    global window_width, window_height
    window_width, window_height = width, height; glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION); glLoadIdentity()
    if mode == '2D': gluOrtho2D(0, width, 0, height)
    else: gluPerspective(45, (width / height) if height > 0 else 1, 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)

# <<< DIPERBARUI: Petunjuk Penggunaan >>>
def print_instructions():
    print("===== PETUNJUK PENGGUNAAN APLIKASI (FINAL) =====")
    print("\n--- KONTROL GLOBAL ---")
    print("Tombol 'm'                   : Ganti mode antara 2D dan 3D")
    print("\n--- MODE 2D ---")
    print("ALAT GAMBAR (Keyboard)       : [1] Garis, [2] Persegi, [3] Elips, [4] Titik")
    print("WARNA (Keyboard)             : [r] Merah, [g] Hijau, [b] Biru")
    print("MENGGAMBAR                   : Klik Kiri & Drag di area kosong")
    print("PILIH & GESER OBJEK          : Klik Kiri & Drag pada objek")
    print("ROTASI OBJEK                 : Tahan [Ctrl] + Klik Kiri & Drag pada objek")
    print("SKALA OBJEK                  : Tahan [Alt] + Klik Kiri & Drag pada objek")
    print("TEBAL GARIS                  : Tekan [+] / [-]")
    print("\n--- FUNGSI CLIPPING WINDOW ---")
    print("BUAT WINDOW                  : Tahan [Shift] + Klik Kiri & Drag")
    print("GESER WINDOW                 : Tombol Panah (Arrow Keys)")
    print("UBAH UKURAN WINDOW           : Klik Kiri & Drag pada handle kuning di sudut window")
    print("HAPUS WINDOW                 : Tombol 'c'")
    print("\n--- MODE 3D ---")
    print("ROTASI OBJEK                 : Klik Kiri & Drag")
    print("TRANSLASI/ZOOM               : Tombol [w,s,a,d,r,f]")
    print("=====================================================================")
    if not has_obj_loader: print("\nINFO: 'model.obj' tidak ditemukan/dimuat. Menampilkan kubus sebagai gantinya.")

def main():
    global scene_3d
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(window_width, window_height)
    glutCreateWindow(b"UAS Grafika Komputer - Final (Resize & Pan)")
    glClearColor(0.1, 0.1, 0.2, 1.0)
    
    # Daftarkan semua callback
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)
    glutMouseFunc(mouse)
    glutMotionFunc(motion)
    glutPassiveMotionFunc(passive_motion)
    glutSpecialFunc(special_keyboard) # <<< BARU: Mendaftarkan fungsi untuk tombol panah
    
    if has_obj_loader:
        try:
            scene_3d = Wavefront('model.obj', collect_faces=True, parse=True)
            print("INFO: File 'model.obj' berhasil dimuat.")
        except Exception as e:
            print(f"Peringatan: Gagal memuat 'model.obj' - {e}. Menampilkan kubus."); scene_3d = None
    print_instructions()
    glutMainLoop()

if __name__ == '__main__':
    main()