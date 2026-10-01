import time
import os
import json
import threading
import sys
from pathlib import Path
import cv2
import numpy as np

_BUNDLED_SDK_SRC = Path(__file__).resolve().parent / "RoboMaster-SDK" / "src"
if _BUNDLED_SDK_SRC.is_dir() and str(_BUNDLED_SDK_SRC) not in sys.path:
    sys.path.insert(0, str(_BUNDLED_SDK_SRC))

from robomaster import blaster, led, robot

# ==============================================================================
# คอนฟิกและการประมวลผลเป้าหมาย
# ==============================================================================

GRID_ROWS = 3
GRID_COLS = 3
CAMERA_HFOV_DEG = 96.0
# Exploration cells are 60 cm wide; two grids is a 1.2 m firing limit.
MAX_TARGET_RANGE_M = 1.2


DEFAULT_HSV_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "hsv_config.json"
)


def resolve_config_path(config_file=DEFAULT_HSV_CONFIG_PATH):
    if os.path.isabs(config_file):
        return config_file

    # Prefer the config beside this controller so launching from another
    # working directory cannot silently select a different hsv_config.json.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    module_config = os.path.join(base_dir, config_file)
    if os.path.exists(module_config):
        return module_config
    if os.path.exists(config_file):
        return os.path.abspath(config_file)
    return module_config


def load_hsv_configs(config_file=DEFAULT_HSV_CONFIG_PATH):
    resolved_path = resolve_config_path(config_file)
    if os.path.exists(resolved_path):
        try:
            with open(resolved_path, "r", encoding="utf-8") as f:
                raw_cfg = json.load(f)
            print(f"[HSV] ใช้ไฟล์ค่าสี: {resolved_path}")
            configs = {}
            for color_name, data in raw_cfg.items():
                ranges = []
                for r in data.get("ranges", []):
                    ranges.append((
                        np.array(r["lower"], dtype=np.uint8),
                        np.array(r["upper"], dtype=np.uint8)
                    ))
                configs[color_name] = {
                    "ranges": ranges,
                    "draw_color": tuple(data.get("draw_color", [0, 255, 0])),
                    "led_rgb": tuple(data.get("led_rgb", [255, 255, 255])),
                    "name_th": data.get("name_th", color_name),
                    "physical_size_m": data.get("physical_size_m")
                }
            return configs
        except Exception as e:
            print(f"[!] โหลด {resolved_path} ไม่สำเร็จ: {e}")

    return {
        "Red": {"ranges": [(np.array([0, 100, 70]), np.array([10, 255, 255])), (np.array([165, 100, 70]), np.array([180, 255, 255]))], "draw_color": (0, 0, 255), "led_rgb": (255, 0, 0), "name_th": "แดง"},
        "Green": {"ranges": [(np.array([45, 80, 60]), np.array([85, 255, 255]))], "draw_color": (0, 255, 0), "led_rgb": (0, 255, 0), "name_th": "เขียว"},
        "Blue": {"ranges": [(np.array([95, 80, 50]), np.array([130, 255, 255]))], "draw_color": (255, 130, 0), "led_rgb": (0, 100, 255), "name_th": "น้ำเงิน"},
        "Yellow": {"ranges": [(np.array([20, 110, 90]), np.array([38, 255, 255]))], "draw_color": (0, 220, 255), "led_rgb": (255, 255, 0), "name_th": "เหลือง"}
    }


def get_color_mappings(config_file=DEFAULT_HSV_CONFIG_PATH):
    cfg = load_hsv_configs(config_file)
    hex_fallback = {
        "Red": "#FF4444",
        "Green": "#2ECC71",
        "Blue": "#3498DB",
        "Yellow": "#F1C40F"
    }
    available_colors = []
    color_th_map = {"ALL": "ทุกสี"}
    for cname, cdata in cfg.items():
        th = cdata.get("name_th", cname)
        cth = f"สี{th}" if not th.startswith("สี") else th
        led_rgb = cdata.get("led_rgb", (255, 255, 255))
        chex = hex_fallback.get(cname, f"#{led_rgb[0]:02x}{led_rgb[1]:02x}{led_rgb[2]:02x}")
        draw_color = cdata.get("draw_color", (0, 255, 0))
        available_colors.append((cname, cth, chex, draw_color))
        color_th_map[cname] = cth
    return available_colors, color_th_map


AVAILABLE_COLORS, COLOR_TH_MAP = get_color_mappings()

AVAILABLE_SHAPES = [
    ("Circle", "ทรงกลม", "🔘"),
    ("Square", "สี่เหลี่ยมจัตุรัส", "⬛"),
    ("Rect_H", "ผืนผ้านอน", "▰"),
    ("Rect_V", "ผืนผ้าตั้ง", "▮"),
    ("Rectangle", "ผืนผ้าทั้งหมด", "▭")
]

SHAPE_TH_MAP = {
    "Circle": "ทรงกลม",
    "Square": "สี่เหลี่ยมจัตุรัส",
    "Rectangle": "สี่เหลี่ยมผืนผ้า",
    "Rect_H": "ผืนผ้านอน",
    "Rect_V": "ผืนผ้าตั้ง",
    "ALL": "ทุกรูปทรง"
}

FOUR_DIRECTIONS = [
    {"id": "front", "name_th": "ด้านหน้า", "name_en": "FRONT", "yaw": 0.0,   "icon": "⬆️"},
    {"id": "right", "name_th": "ด้านขวา",  "name_en": "RIGHT", "yaw": 90.0,  "icon": "➡️"},
    {"id": "left",  "name_th": "ด้านซ้าย", "name_en": "LEFT",  "yaw": -90.0, "icon": "⬅️"},
]

DEFAULT_ROI = (100,150, 1080, 620)


# ==============================================================================
# ROIManager
# ==============================================================================

class ROIManager:
    def __init__(self, roi=DEFAULT_ROI, enabled=True):
        self.enabled = enabled
        self.roi = roi

    def get_bounds(self, img_shape):
        h, w = img_shape[:2]
        if not self.enabled or self.roi is None:
            return 0, 0, w, h
        x1, y1, x2, y2 = self.roi
        return max(0, min(w - 1, int(x1))), max(0, min(h - 1, int(y1))), max(1, min(w, int(x2))), max(1, min(h, int(y2)))

    def crop_roi(self, img):
        if not self.enabled or self.roi is None:
            return img.copy(), (0, 0)
        x1, y1, x2, y2 = self.get_bounds(img.shape)
        return img[y1:y2, x1:x2].copy(), (x1, y1)

    def draw_roi_hud(self, img):
        if not self.enabled or self.roi is None:
            return img
        h, w = img.shape[:2]
        x1, y1, x2, y2 = self.get_bounds(img.shape)

        # ทำมืดพื้นที่นอกกรอบ ROI
        if y1 > 0:
            img[:y1, :] = cv2.addWeighted(img[:y1, :], 0.25, np.zeros_like(img[:y1, :]), 0.75, 0)
        if y2 < h:
            img[y2:, :] = cv2.addWeighted(img[y2:, :], 0.25, np.zeros_like(img[y2:, :]), 0.75, 0)
        if x1 > 0 and y2 > y1:
            img[y1:y2, :x1] = cv2.addWeighted(img[y1:y2, :x1], 0.25, np.zeros_like(img[y1:y2, :x1]), 0.75, 0)
        if x2 < w and y2 > y1:
            img[y1:y2, x2:] = cv2.addWeighted(img[y1:y2, x2:], 0.25, np.zeros_like(img[y1:y2, x2:]), 0.75, 0)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 200), 2)
        c_len = 20
        cv2.line(img, (x1, y1), (x1 + c_len, y1), (0, 255, 255), 3)
        cv2.line(img, (x1, y1), (x1, y1 + c_len), (0, 255, 255), 3)
        cv2.line(img, (x2, y1), (x2 - c_len, y1), (0, 255, 255), 3)
        cv2.line(img, (x2, y1), (x2, y1 + c_len), (0, 255, 255), 3)
        cv2.line(img, (x1, y2), (x1 + c_len, y2), (0, 255, 255), 3)
        cv2.line(img, (x1, y2), (x1, y2 - c_len), (0, 255, 255), 3)
        cv2.line(img, (x2, y2), (x2 - c_len, y2), (0, 255, 255), 3)
        cv2.line(img, (x2, y2), (x2, y2 - c_len), (0, 255, 255), 3)

        cv2.putText(img, f"[ROI: ({x1}, {y1}) -> ({x2}, {y2})]", (x1 + 10, max(22, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 200), 1)
        return img


# ==============================================================================
# PID Controller & Target Memory
# ==============================================================================

class PIDController:
    def __init__(self, kp, ki, kd, limits=(-120, 120)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_limit, self.max_limit = limits
        self.last_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()

    def compute(self, error):
        now = time.time()
        dt = now - self.last_time
        if dt <= 0:
            dt = 0.01

        if abs(error) < 0.012:
            self.last_error = error
            self.last_time = now
            return 0.0

        p_term = self.kp * error
        self.integral += error * dt
        self.integral = max(-40.0, min(40.0, self.integral))
        i_term = self.ki * self.integral
        derivative = (error - self.last_error) / dt
        d_term = self.kd * derivative

        output = p_term + i_term + d_term
        output = max(self.min_limit, min(self.max_limit, output))

        self.last_error = error
        self.last_time = now
        return output

    def reset(self):
        self.last_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()


pid_yaw = PIDController(kp=115.0, ki=0.01, kd=4.5, limits=(-120, 120))
pid_pitch = PIDController(kp=95.0, ki=0.0, kd=4.0, limits=(-80, 80))

LOCK_TOLERANCE_X = 0.040
LOCK_TOLERANCE_Y = 0.040
LOCK_HOLD_TIME = 0.15
MIN_CONTOUR_AREA = 300

current_gimbal_yaw = 0.0
current_gimbal_pitch = 0.0


def on_gimbal_angle_cb(angle_info):
    global current_gimbal_yaw, current_gimbal_pitch
    pitch, yaw, pitch_ground, yaw_ground = angle_info
    current_gimbal_pitch = pitch
    current_gimbal_yaw = yaw


class TargetMemory:
    def __init__(self, hfov=96.0, yaw_match_tolerance=4.0):
        self.hfov = hfov
        self.yaw_match_tolerance = yaw_match_tolerance
        self.shot_records = []

    def record_shot(self, target_num, gimbal_yaw, target_info=None):
        self.shot_records.append({
            "target_num": target_num,
            "yaw": gimbal_yaw,
            "color": target_info.get("color") if target_info else "",
            "shape": target_info.get("shape") if target_info else "",
            "time": time.time()
        })

    def is_already_shot(self, norm_x, current_yaw, target_color=None, target_shape=None):
        if not self.shot_records:
            return False, None
        delta_yaw = (norm_x - 0.5) * self.hfov
        estimated_yaw = current_yaw + delta_yaw
        for record in self.shot_records:
            diff = (estimated_yaw - record["yaw"] + 180) % 360 - 180
            if abs(diff) < self.yaw_match_tolerance:
                rec_c = record.get("color")
                rec_s = record.get("shape")
                if target_color and rec_c and target_color != rec_c:
                    continue
                if target_shape and rec_s and target_shape != rec_s:
                    continue
                return True, record["target_num"]
        return False, None

    def annotate_targets(self, targets, current_yaw):
        for t in targets:
            is_shot, shot_id = self.is_already_shot(
                t["norm_x"], current_yaw,
                target_color=t.get("color"),
                target_shape=t.get("shape")
            )
            t["is_already_shot"] = is_shot
            t["shot_id"] = shot_id
        return targets

    def get_unshot_targets(self, targets):
        return [t for t in targets if not t.get("is_already_shot", False)]

    def clear(self):
        self.shot_records.clear()


# ==============================================================================
# Shape Classifier & Target Detection
# ==============================================================================

def rectify_quadrilateral_contour(mask, contour):
    """Warp a perspective-distorted four-corner target to a front-facing patch."""
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    if perimeter <= 0:
        return None

    quad = None
    for epsilon in (0.015, 0.02, 0.03, 0.04, 0.05):
        approx = cv2.approxPolyDP(hull, epsilon * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quad = approx.reshape(4, 2).astype(np.float32)
            break
    if quad is None:
        return None

    quad_area = cv2.contourArea(quad)
    contour_area = cv2.contourArea(contour)
    if quad_area <= 0 or contour_area / quad_area < 0.88:
        return None

    # Order the corners as top-left, top-right, bottom-right, bottom-left.
    sums = quad.sum(axis=1)
    diffs = quad[:, 1] - quad[:, 0]
    ordered = np.array([
        quad[np.argmin(sums)], quad[np.argmin(diffs)],
        quad[np.argmax(sums)], quad[np.argmax(diffs)]
    ], dtype=np.float32)

    tl, tr, br, bl = ordered
    out_w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    out_h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    if out_w < 12 or out_h < 12:
        return None
    out_w, out_h = min(out_w, 2048), min(out_h, 2048)

    x, y, w, h = cv2.boundingRect(contour)
    local_mask = np.zeros((h, w), dtype=np.uint8)
    local_contour = contour.copy()
    local_contour[:, 0, 0] -= x
    local_contour[:, 0, 1] -= y
    cv2.drawContours(local_mask, [local_contour], -1, 255, thickness=cv2.FILLED)

    source = ordered - np.array([x, y], dtype=np.float32)
    destination = np.array([
        [0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]
    ], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(source, destination)
    warped = cv2.warpPerspective(
        local_mask, transform, (out_w, out_h), flags=cv2.INTER_NEAREST
    )
    contours, _ = cv2.findContours(warped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea) if contours else None


def classify_shape(cnt, gimbal_pitch=0.0):
    area = cv2.contourArea(cnt)
    if area < MIN_CONTOUR_AREA:
        return None

    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        return None

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    if solidity < 0.60:
        return None

    # Check true quadrilaterals before roundness metrics (a square can look very circular).
    quad = None
    hull_perimeter = cv2.arcLength(hull, True)
    if hull_perimeter > 0:
        for eps_factor in (0.02, 0.03, 0.04, 0.05):
            candidate = cv2.approxPolyDP(hull, eps_factor * hull_perimeter, True)
            if len(candidate) == 4 and cv2.isContourConvex(candidate):
                if cv2.contourArea(candidate) > 0 and area / cv2.contourArea(candidate) >= 0.88:
                    quad = candidate
                    break
    if quad is not None:
        _, _, q_w, q_h = cv2.boundingRect(quad)
        if q_w > 0 and q_h > 0:
            aspect = float(q_w) / float(q_h)
            if 0.80 <= aspect <= 1.25:
                return "Square"
            return "Rect_H" if aspect > 1.25 else "Rect_V"

    circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
    _, radius = cv2.minEnclosingCircle(cnt)
    circle_area = np.pi * (radius ** 2)
    circle_ratio = area / circle_area if circle_area > 0 else 0

    if circularity >= 0.44 or circle_ratio >= 0.46:
        return "Circle"

    if len(cnt) >= 5:
        try:
            ellipse = cv2.fitEllipse(cnt)
            (e_cx, e_cy), (axis1, axis2), _ = ellipse
            major_axis = max(axis1, axis2)
            minor_axis = min(axis1, axis2)
            if major_axis > 0 and minor_axis > 0:
                ellipse_area = np.pi * (major_axis / 2.0) * (minor_axis / 2.0)
                ellipse_ratio = area / ellipse_area if ellipse_area > 0 else 0
                if 0.68 <= ellipse_ratio <= 1.32:
                    return "Circle"
        except Exception:
            pass

    approx_quad = None
    for eps_factor in (0.032, 0.040, 0.048, 0.055):
        app = cv2.approxPolyDP(hull, eps_factor * perimeter, True)
        if len(app) == 4 and cv2.isContourConvex(app):
            approx_quad = app
            break

    if approx_quad is not None:
        _, _, bw, bh = cv2.boundingRect(approx_quad)
        bbox_area = bw * bh
        extent = area / bbox_area if bbox_area > 0 else 0

        if extent >= 0.58:
            pitch_deg = abs(float(gimbal_pitch))
            comp_factor = float(np.cos(np.radians(pitch_deg))) if pitch_deg > 5.0 else 1.0
            comp_factor = max(0.60, min(1.0, comp_factor))
            aspect_raw = float(bw) / float(bh)
            aspect_comp = aspect_raw * comp_factor

            if (0.80 <= aspect_comp <= 1.25) or (0.82 <= aspect_raw <= 1.22):
                return "Square"
            elif aspect_comp > 1.25 or aspect_raw > 1.22:
                return "Rect_H"
            else:
                return "Rect_V"

    return None


def is_color_matched(color_name, color_filter):
    if color_filter == "ALL":
        return True
    if isinstance(color_filter, (list, tuple, set)):
        return ("ALL" in color_filter) or (color_name in color_filter)
    return color_name == color_filter


def is_shape_matched(shape, shape_filter):
    if shape_filter == "ALL":
        return True
    if isinstance(shape_filter, (list, tuple, set)):
        if "ALL" in shape_filter:
            return True
        for sf in shape_filter:
            if sf in ("Square", "สี่เหลี่ยมจัตุรัส") and shape == "Square":
                return True
            elif sf in ("Circle", "ทรงกลม") and shape == "Circle":
                return True
            elif sf in ("Rect_H", "Horizontal", "ผืนผ้านอน") and shape == "Rect_H":
                return True
            elif sf in ("Rect_V", "Vertical", "ผืนผ้าตั้ง") and shape == "Rect_V":
                return True
            elif sf in ("Rectangle", "สี่เหลี่ยมผืนผ้า") and shape in ("Rectangle", "Rect_H", "Rect_V"):
                return True
            elif sf == shape:
                return True
        return False
    else:
        if shape_filter in ("Square", "สี่เหลี่ยมจัตุรัส"):
            return shape == "Square"
        elif shape_filter in ("Circle", "ทรงกลม"):
            return shape == "Circle"
        elif shape_filter in ("Rect_H", "Horizontal", "ผืนผ้านอน"):
            return shape == "Rect_H"
        elif shape_filter in ("Rect_V", "Vertical", "ผืนผ้าตั้ง"):
            return shape == "Rect_V"
        elif shape_filter in ("Rectangle", "สี่เหลี่ยมผืนผ้า"):
            return shape in ("Rectangle", "Rect_H", "Rect_V")
        return shape == shape_filter


def format_filter_label(color_filter, shape_filter):
    c_str = "ALL" if color_filter == "ALL" else str(color_filter)
    s_str = "ALL" if shape_filter == "ALL" else str(shape_filter)
    return f"[{c_str} | {s_str}]"


def get_grid_cell(norm_x, norm_y):
    col = min(GRID_COLS - 1, max(0, int(norm_x * GRID_COLS)))
    row = min(GRID_ROWS - 1, max(0, int(norm_y * GRID_ROWS)))
    return f"{chr(ord('A') + row)}{col + 1}"


def estimate_world_grid(distance_m, robot_grid, robot_heading,
                        camera_direction, grid_size_m):
    """Estimate map cell along a cardinal scan direction from calibrated range."""
    if distance_m is None or not grid_size_m or grid_size_m <= 0:
        return None
    direction_offset = {"front": 0, "right": 1, "back": 2, "left": 3}.get(camera_direction)
    if direction_offset is None:
        return None
    direction = (int(robot_heading) + direction_offset) % 4
    dc, dr = ((0, -1), (1, 0), (0, 1), (-1, 0))[direction]
    steps = max(1, int(np.ceil(float(distance_m) / float(grid_size_m))))
    col, row = robot_grid
    return col + dc * steps, row + dr * steps


def estimate_distance_m(apparent_size_px, frame_width, physical_size_m):
    if not physical_size_m or apparent_size_px <= 0:
        return None
    focal_px = frame_width / (2.0 * np.tan(np.radians(CAMERA_HFOV_DEG / 2.0)))
    return float(physical_size_m) * focal_px / float(apparent_size_px)

def detect_targets(img, color_filter="Red", shape_filter="Circle", hsv_configs=None,
                   gimbal_pitch=None, roi_manager=None, wall_top_y=0):
    if hsv_configs is None:
        hsv_configs = load_hsv_configs()
    if gimbal_pitch is None:
        gimbal_pitch = current_gimbal_pitch

    h, w, _ = img.shape

    # ครอป ROI ผ่าน ROIManager โดยตรง ไม่เขียนคำนวณซ้ำ
    if roi_manager:
        proc_img, (offset_x, offset_y) = roi_manager.crop_roi(img)
    else:
        proc_img, (offset_x, offset_y) = img.copy(), (0, 0)

    # ตัดพื้นที่เหนือแนวกำแพงขาวใน ROI
    if wall_top_y > offset_y:
        rel_wall_top = min(proc_img.shape[0], wall_top_y - offset_y)
        proc_img[:rel_wall_top, :] = 0

    hsv = cv2.cvtColor(proc_img, cv2.COLOR_BGR2HSV)
    targets = []

    for color_name, color_cfg in hsv_configs.items():
        if not is_color_matched(color_name, color_filter):
            continue

        mask = None
        for lower, upper in color_cfg["ranges"]:
            part = cv2.inRange(hsv, lower, upper)
            mask = part if mask is None else cv2.bitwise_or(mask, part)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        for cnt in contours:
            rectified_cnt = None
            if abs(float(gimbal_pitch)) >= 8.0:
                rectified_cnt = rectify_quadrilateral_contour(mask, cnt)
            shape = classify_shape(rectified_cnt, gimbal_pitch=0.0) if rectified_cnt is not None else None
            if shape is None:
                shape = classify_shape(cnt, gimbal_pitch=gimbal_pitch)
            if shape is None or not is_shape_matched(shape, shape_filter):
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"]) + offset_x
            cy = int(M["m01"] / M["m00"]) + offset_y

            bx, by, bw, bh = cv2.boundingRect(cnt)
            bx += offset_x
            by += offset_y

            shape_th = SHAPE_TH_MAP.get(shape, shape)
            color_th = color_cfg.get("name_th", color_name)
            name_th = f"{shape_th} สี{color_th}" if not color_th.startswith("สี") else f"{shape_th} {color_th}"

            norm_x = cx / float(w)
            norm_y = cy / float(h)
            apparent_size_px = max(bw, bh)
            distance_m = estimate_distance_m(
                apparent_size_px, w, color_cfg.get("physical_size_m")
            )
            range_label = f"{distance_m:.2f} m" if distance_m is not None else f"{apparent_size_px}px (uncalibrated)"

            targets.append({
                "color": color_name,
                "shape": shape,
                "center": (cx, cy),
                "norm_x": norm_x,
                "norm_y": norm_y,
                "grid_cell": get_grid_cell(norm_x, norm_y),
                "apparent_size_px": apparent_size_px,
                "distance_m": distance_m,
                "range_label": range_label,
                "bbox": (bx, by, bw, bh),
                "area": cv2.contourArea(cnt),
                "draw_color": color_cfg["draw_color"],
                "led_rgb": color_cfg["led_rgb"],
                "name_th": name_th
            })

    targets.sort(key=lambda t: t["center"][0])
    return targets


# ==============================================================================
# Camera Pump & Control Loop
# ==============================================================================

last_valid_frame = None
last_valid_frame_time = None
frame_lock = threading.Lock()
live_dashboard_callback = None
chassis_hold_callback = None


def set_live_dashboard(callback):
    """Attach a renderer that combines the live camera with exploration status."""
    global live_dashboard_callback
    live_dashboard_callback = callback


def draw_camera_grid(img):
    height, width = img.shape[:2]
    line_color = (115, 115, 115)
    for col in range(1, GRID_COLS):
        x = width * col // GRID_COLS
        cv2.line(img, (x, 0), (x, height), line_color, 1)
    for row in range(1, GRID_ROWS):
        y = height * row // GRID_ROWS
        cv2.line(img, (0, y), (width, y), line_color, 1)
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            cell = f"{chr(ord('A') + row)}{col + 1}"
            cv2.putText(img, cell, (col * width // GRID_COLS + 8, row * height // GRID_ROWS + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)
    return img

def pump_camera_stream(ep_camera, window_name, status_text="", roi_manager=None, wall_top_y=0,
                       overlay_fn=None, read_timeout=0.10):
    global last_valid_frame, last_valid_frame_time
    if ep_camera is None:
        return None

    # During target scan/aim/fire, periodically reassert a zero chassis speed
    # from this same control loop while the gimbal is moving and frames pump.
    if chassis_hold_callback is not None:
        chassis_hold_callback()

    img = None
    try:
        img = ep_camera.read_cv2_image(strategy="newest", timeout=read_timeout)
    except Exception:
        img = None

    # กล้องอาจส่งเฟรมว่างชั่วคราวระหว่างขยับกิมบอล ห้ามส่งเฟรม 0x0 เข้า imshow
    frame_is_valid = (
        isinstance(img, np.ndarray)
        and img.ndim == 3
        and img.shape[0] > 0
        and img.shape[1] > 0
        and img.shape[2] >= 3
    )

    with frame_lock:
        if frame_is_valid:
            last_valid_frame = img.copy()
            last_valid_frame_time = time.monotonic()
        elif last_valid_frame is not None and last_valid_frame.size > 0:
            img = last_valid_frame.copy()
        else:
            img = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(img, "WAITING FOR CAMERA STREAM...", (380, 360),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2)

    display_img = draw_camera_grid(img.copy())
    h, w, _ = display_img.shape

    if roi_manager:
        display_img = roi_manager.draw_roi_hud(display_img)

    if status_text:
        cv2.rectangle(display_img, (20, 20), (min(w - 20, 920), 75), (0, 0, 0), -1)
        cv2.rectangle(display_img, (20, 20), (min(w - 20, 920), 75), (0, 255, 255), 2)
        cv2.putText(display_img, status_text, (35, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2)

    if not frame_is_valid:
        age = (time.monotonic() - last_valid_frame_time
               if last_valid_frame_time is not None else None)
        if age is None or age > 0.5:
            warning = "NO CAMERA FRAME" if age is None else f"CAMERA FRAME STALE: {age:.1f}s"
            cv2.putText(display_img, warning, (25, h - 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 255), 2)

    if overlay_fn:
        display_img = overlay_fn(display_img)

    # ใช้เฟรมที่ตรวจสอบแล้วหาก overlay ส่งภาพว่างกลับมา
    if not isinstance(display_img, np.ndarray) or display_img.ndim < 2 or display_img.shape[0] == 0 or display_img.shape[1] == 0:
        display_img = img.copy()

    if live_dashboard_callback is not None:
        live_dashboard_callback(display_img, status_text)
    else:
        cv2.imshow(window_name, display_img)
    cv2.waitKey(1)
    return img


def sleep_with_stream(ep_camera, window_name, duration, overlay_text="", roi_manager=None, wall_top_y=0):
    end_t = time.time() + duration
    while time.time() < end_t:
        pump_camera_stream(ep_camera, window_name, overlay_text, roi_manager, wall_top_y)
        time.sleep(0.01)


def wait_for_gimbal_idle(ep_gimbal, ep_camera, window_name, timeout=4.0,
                         roi_manager=None, wall_top_y=0):
    """Wait until the SDK dispatcher releases the previous gimbal action."""
    dispatcher = getattr(ep_gimbal, "_action_dispatcher", None)
    if dispatcher is None:
        return
    start_t = time.time()
    while getattr(dispatcher, "has_in_progress_actions", False):
        if time.time() - start_t >= timeout:
            raise TimeoutError("Gimbal ยังมี action ก่อนหน้าค้างอยู่; งดส่งคำสั่งซ้อน")
        pump_camera_stream(ep_camera, window_name, "WAITING FOR GIMBAL ACTION TO FINISH",
                           roi_manager, wall_top_y)
        time.sleep(0.02)


def move_gimbal_with_stream(ep_gimbal, ep_camera, pitch, yaw, pitch_speed=40, yaw_speed=45,
                            timeout=2.0, window_name="RoboMaster 4-Direction Gimbal Scan", status_text="",
                            roi_manager=None, wall_top_y=0):
    safe_pitch = max(-20.0, min(0.0, float(pitch)))
    # Moveto is absolute. Add enough time for long turns (e.g. 180°/270°)
    # plus a margin for SDK/network latency, so the next action will not overlap.
    pitch_delta = abs(safe_pitch - float(current_gimbal_pitch))
    yaw_delta = abs(float(yaw) - float(current_gimbal_yaw))
    pitch_time = pitch_delta / max(1.0, abs(float(pitch_speed)))
    yaw_time = yaw_delta / max(1.0, abs(float(yaw_speed)))
    # Firmware action completion notifications can lag behind the physical
    # movement, especially after a 4-direction wall scan. Keep the next gimbal
    # command serialized but allow enough time for that completion event.
    wait_limit = max(float(timeout), pitch_time + 4.0, yaw_time + 4.0)

    try:
        wait_for_gimbal_idle(ep_gimbal, ep_camera, window_name, wait_limit, roi_manager, wall_top_y)
        action = ep_gimbal.moveto(pitch=int(safe_pitch), yaw=int(yaw),
                                  pitch_speed=int(pitch_speed), yaw_speed=int(yaw_speed))
    except Exception as e:
        raise RuntimeError(f"ไม่สามารถสั่ง Gimbal เคลื่อนที่ได้: {e}") from e

    msg = status_text if status_text else f"MOVING GIMBAL -> Pitch: {safe_pitch:.1f}, Yaw: {yaw:.1f}..."
    start_t = time.monotonic()
    deadline = start_t + wait_limit
    # Do not call SDK wait_for_completed(timeout=...) in short polling slices:
    # the SDK marks the action as ACTION_EXCEPTION whenever one slice expires.
    # Poll its state without mutating it, while continuing to pump camera frames.
    while time.monotonic() < deadline:
        pump_camera_stream(ep_camera, window_name, msg, roi_manager, wall_top_y)
        if action and hasattr(action, "is_completed") and action.is_completed:
            break
        time.sleep(0.02)

    if action and hasattr(action, "is_completed") and not action.is_completed:
        state = getattr(action, "state", "unknown")
        percent = getattr(action, "_percent", "unknown")
        raise TimeoutError(
            f"Gimbal action ไม่ได้รับสถานะจบภายใน {wait_limit:.1f} วินาที "
            f"(state={state}, progress={percent}%); หยุดก่อนส่งคำสั่งถัดไป"
        )
    if action and hasattr(action, "has_succeeded") and not action.has_succeeded:
        failure_reason = getattr(action, "failure_reason", None)
        raise RuntimeError(
            f"Gimbal action จบไม่สำเร็จ: {getattr(action, 'state', 'unknown')}"
            f"{f' ({failure_reason})' if failure_reason else ''}"
        )

    wait_for_gimbal_idle(ep_gimbal, ep_camera, window_name, wait_limit, roi_manager, wall_top_y)
    sleep_with_stream(ep_camera, window_name, 0.1, roi_manager=roi_manager, wall_top_y=wall_top_y)
    pid_yaw.reset()
    pid_pitch.reset()


def turn_and_tilt_direction(ep_gimbal, ep_camera, target_yaw, target_pitch,
                            direction_en="FRONT", window_name="RoboMaster 4-Direction Gimbal Scan",
                            roi_manager=None, wall_top_y=0):
    safe_pitch = max(-20.0, min(0.0, float(target_pitch)))
    print(f"⬆️ ยก Gimbal ก่อนเปลี่ยนทิศ -> {direction_en} ({target_yaw}°)")

    move_gimbal_with_stream(
        ep_gimbal=ep_gimbal,
        ep_camera=ep_camera,
        pitch=0,
        yaw=current_gimbal_yaw,
        pitch_speed=40,
        yaw_speed=45,
        timeout=2.2,
        window_name=window_name,
        status_text="LIFTING GIMBAL BEFORE TURNING...",
        roi_manager=roi_manager,
        wall_top_y=wall_top_y
    )

    move_gimbal_with_stream(
        ep_gimbal=ep_gimbal,
        ep_camera=ep_camera,
        pitch=0,
        yaw=target_yaw,
        pitch_speed=40,
        yaw_speed=45,
        timeout=2.2,
        window_name=window_name,
        status_text=f"TURNING TO {direction_en.upper()} WHILE RAISED...",
        roi_manager=roi_manager,
        wall_top_y=wall_top_y
    )

    print(f"⬇️ ก้ม Gimbal เพื่อเช็ก {direction_en} -> Pitch: {safe_pitch}°")
    move_gimbal_with_stream(
        ep_gimbal=ep_gimbal,
        ep_camera=ep_camera,
        pitch=safe_pitch,
        yaw=target_yaw,
        pitch_speed=40,
        yaw_speed=45,
        timeout=2.2,
        window_name=window_name,
        status_text=f"LOWERING TO SCAN {direction_en.upper()}...",
        roi_manager=roi_manager,
        wall_top_y=wall_top_y
    )

def fire_double_shot(ep_blaster, ep_led, ep_camera=None, window_name="RoboMaster 4-Direction Gimbal Scan",
                     count=2, enable_fire=True, led_rgb=None, fire_mode="ir",
                     roi_manager=None, wall_top_y=0):
    if not enable_fire:
        sleep_with_stream(ep_camera, window_name, 0.3, "SAFE MODE: SIMULATED FIRE", roi_manager, wall_top_y)
        return

    print(f"\n💥 >> [FIRING] ลั่นไก {count} นัด ({fire_mode.upper()}) << 💥")
    fire_rgb = led_rgb if led_rgb is not None else (255, 50, 0)
    try:
        ep_led.set_led(comp=led.COMP_TOP_ALL, r=fire_rgb[0], g=fire_rgb[1], b=fire_rgb[2], effect=led.EFFECT_ON)
    except Exception:
        pass

    def _worker():
        if fire_mode == "dual":
            try:
                ret = ep_blaster.fire(fire_type=blaster.WATER_FIRE, times=count)
                if not ret:
                    print("[!] Water-gel shot returned no confirmation; continuing with IR.")
            except Exception as e:
                print(f"[!] Water-gel firing error: {e}")
            try:
                ep_blaster.fire(fire_type=blaster.INFRARED_FIRE, times=count)
            except Exception as e:
                print(f"[!] IR firing error: {e}")
        else:
            try:
                if fire_mode == "water":
                    ret = ep_blaster.fire(fire_type=blaster.WATER_FIRE, times=count)
                    if not ret:
                        ep_blaster.fire(fire_type=blaster.INFRARED_FIRE, times=count)
                else:
                    ep_blaster.fire(fire_type=blaster.INFRARED_FIRE, times=count)
            except Exception as e:
                print(f"[!] Firing error: {e}")

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    while t.is_alive():
        pump_camera_stream(ep_camera, window_name, f"FIRING [{count} SHOTS] ({fire_mode.upper()})...", roi_manager, wall_top_y)
        time.sleep(0.02)

    t.join(timeout=1.0)

    try:
        ep_led.set_led(comp=led.COMP_TOP_ALL, r=0, g=0, b=0, effect=led.EFFECT_OFF)
    except Exception:
        pass


def shoot_until_target_falls(ep_blaster, ep_led, ep_camera, ep_gimbal,
                             locked_target, color_name, shape_name, hsv_configs,
                             wall_segmenter, target_index, total_targets,
                             shots_per_target=2, max_retries=3, fire_mode="ir",
                             roi_manager=None, window_name="RoboMaster 4-Direction Gimbal Scan"):
    led_rgb = locked_target.get("led_rgb")

    if fire_mode == "ir":
        fire_double_shot(
            ep_blaster=ep_blaster, ep_led=ep_led, ep_camera=ep_camera,
            window_name=window_name, count=shots_per_target, enable_fire=True,
            led_rgb=led_rgb, fire_mode="ir", roi_manager=roi_manager
        )
        print(f"📡 [IR FIRE] ลั่นลำแสงอินฟราเรดใส่เป้าหมาย #{target_index} สำเร็จ")
        sleep_with_stream(ep_camera, window_name, 0.25, f"TARGET #{target_index} HIT CONFIRMED!", roi_manager)
        return True, shots_per_target

    pre_norm_x = locked_target["norm_x"]
    pre_norm_y = locked_target["norm_y"]
    pre_bbox = locked_target["bbox"]
    pre_area = pre_bbox[2] * pre_bbox[3]

    attempt = 0
    target_has_fallen = False
    total_shots_fired = 0

    while not target_has_fallen and attempt < max_retries:
        attempt += 1
        print(f"\n💥 [ENGAGE #{attempt}/{max_retries}] ลั่นไกใส่เป้าหมาย #{target_index} (โหมด: {fire_mode.upper()})...")

        fire_double_shot(
            ep_blaster=ep_blaster, ep_led=ep_led, ep_camera=ep_camera,
            window_name=window_name, count=shots_per_target, enable_fire=True,
            led_rgb=led_rgb, fire_mode=fire_mode, roi_manager=roi_manager
        )
        total_shots_fired += shots_per_target

        sleep_with_stream(ep_camera, window_name, 0.35, f"CHECKING IF TARGET FELL #{attempt}...", roi_manager)

        try:
            img = ep_camera.read_cv2_image(strategy="newest", timeout=0.15)
        except Exception:
            img = None

        if img is None:
            time.sleep(0.02)
            continue

        wall_top_y = 0
        post_targets = detect_targets(img, color_name, shape_name, hsv_configs,
                                      gimbal_pitch=current_gimbal_pitch, roi_manager=roi_manager,
                                      wall_top_y=wall_top_y)

        standing_target = None
        for pt in post_targets:
            dist = np.hypot(pt["norm_x"] - pre_norm_x, pt["norm_y"] - pre_norm_y)
            if dist < 0.085:
                standing_target = pt
                break

        if standing_target is None:
            target_has_fallen = True
            print(f"🎉 [TARGET FELL DOWN!] เป้าหมาย #{target_index} ล้มลงแล้ว! (หายไปในรอบที่ {attempt})")
            sleep_with_stream(ep_camera, window_name, 0.25, f"TARGET #{target_index} FELL DOWN!", roi_manager, wall_top_y)
            break
        else:
            cur_norm_y = standing_target["norm_y"]
            cur_area = standing_target["bbox"][2] * standing_target["bbox"][3]
            drop_y = cur_norm_y - pre_norm_y
            area_ratio = cur_area / float(pre_area) if pre_area > 0 else 1.0

            if drop_y > 0.040 or area_ratio < 0.45:
                target_has_fallen = True
                print(f"🎯 [TARGET FALLING] เป้าหมาย #{target_index} พับล้มลง (Y drop={drop_y:.3f}) -> ยืนยันเป้าล้ม!")
                sleep_with_stream(ep_camera, window_name, 0.25, f"TARGET #{target_index} FELL DOWN!", roi_manager, wall_top_y)
                break
            else:
                target_has_fallen = False
                print(f"⚠️ [STILL STANDING] เป้าหมาย #{target_index} ยังไม่ล้ม! กำลังยิงซ้ำรอบที่ {attempt + 1}...")
                sleep_with_stream(ep_camera, window_name, 0.15, f"RE-SHOOTING TARGET #{target_index}...", roi_manager, wall_top_y)

                if attempt < max_retries:
                    err_x = standing_target["norm_x"] - 0.5
                    err_y = 0.5 - standing_target["norm_y"]
                    ep_gimbal.drive_speed(pitch_speed=pid_pitch.compute(err_y), yaw_speed=pid_yaw.compute(err_x))
                    sleep_with_stream(ep_camera, window_name, 0.08, roi_manager=roi_manager, wall_top_y=wall_top_y)
                    ep_gimbal.drive_speed(0, 0)

    return target_has_fallen, total_shots_fired


def track_and_lock_target(ep_camera, ep_gimbal, ep_led, color_name, shape_name, hsv_configs,
                          target_index, total_targets, target_memory=None,
                          timeout=12.0, window_name="RoboMaster 4-Direction Gimbal Scan", direction_en="",
                          roi_manager=None, wall_segmenter=None, preferred_target=None,
                          fire_on_detection=True):
    pid_yaw.reset()
    pid_pitch.reset()
    lock_start = None
    start_track_time = time.time()
    committed_pos = None
    if preferred_target is not None:
        committed_pos = (preferred_target["norm_x"], preferred_target["norm_y"])

    valid_frames = 0
    frames_with_target = 0
    best_center_error = None

    while time.time() - start_track_time < timeout:
        try:
            img = ep_camera.read_cv2_image(strategy="newest", timeout=0.10)
        except Exception:
            img = None

        if img is None:
            pump_camera_stream(
                ep_camera, window_name,
                status_text=f"AIMING #{target_index}/{total_targets} [{direction_en}] | WAITING FOR CAMERA...",
                roi_manager=roi_manager
            )
            time.sleep(0.01)
            continue

        valid_frames += 1
        wall_top_y = 0
        targets = detect_targets(
            img, color_name, shape_name, hsv_configs,
            gimbal_pitch=current_gimbal_pitch, roi_manager=roi_manager,
            wall_top_y=wall_top_y
        )
        if target_memory is not None:
            targets = target_memory.annotate_targets(targets, current_gimbal_yaw)
            candidates = target_memory.get_unshot_targets(targets)
        else:
            candidates = targets

        target = None
        if committed_pos is None and candidates:
            target = candidates[0]
            committed_pos = (target["norm_x"], target["norm_y"])
        elif committed_pos is not None and candidates:
            best_t = min(
                candidates,
                key=lambda t: np.hypot(t["norm_x"] - committed_pos[0], t["norm_y"] - committed_pos[1])
            )
            distance = np.hypot(best_t["norm_x"] - committed_pos[0], best_t["norm_y"] - committed_pos[1])
            if distance < 0.35:
                target = best_t
                committed_pos = (target["norm_x"], target["norm_y"])

        now_t = time.time()
        locked = False
        if target is not None:
            frames_with_target += 1
            err_x = target["norm_x"] - 0.5
            err_y = 0.5 - target["norm_y"]
            center_error = float(np.hypot(err_x, err_y))
            best_center_error = center_error if best_center_error is None else min(best_center_error, center_error)
            is_centered = abs(err_x) < LOCK_TOLERANCE_X and abs(err_y) < LOCK_TOLERANCE_Y

            if is_centered:
                if lock_start is None:
                    lock_start = now_t
                ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
                locked = now_t - lock_start >= LOCK_HOLD_TIME
            else:
                lock_start = None
                # Proportional speed control avoids derivative spikes from noisy detections.
                yaw_spd = float(np.clip(err_x * 140.0, -55.0, 55.0))
                pitch_spd = float(np.clip(err_y * 120.0, -45.0, 45.0))
                if abs(err_x) > LOCK_TOLERANCE_X and abs(yaw_spd) < 8.0:
                    yaw_spd = 8.0 * float(np.sign(err_x))
                if abs(err_y) > LOCK_TOLERANCE_Y and abs(pitch_spd) < 7.0:
                    pitch_spd = 7.0 * float(np.sign(err_y))
                ep_gimbal.drive_speed(pitch_speed=pitch_spd, yaw_speed=yaw_spd)
        else:
            lock_start = None
            ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)

        if target is None:
            status_msg = f"AIMING #{target_index}/{total_targets} [{direction_en}] | SEARCHING SAVED TARGET | NO SHOT"
            err_x = err_y = None
            locked = False
            target_label = "SAVED TARGET NOT VISIBLE"
        else:
            target_label = target.get("name_th", f"{target.get('color', '')} {target.get('shape', '')}")
            status_msg = f"AIMING #{target_index}/{total_targets} [{direction_en}] | {target_label} | {'LOCKED' if locked else 'TRACKING'}"

        def overlay_fn(display_img, target=target, locked=locked,
                       err_x=err_x, err_y=err_y, target_label=target_label):
            frame_h, frame_w = display_img.shape[:2]
            center = (frame_w // 2, frame_h // 2)
            # Keep the aim point visible even while the saved target is temporarily lost.
            cv2.drawMarker(display_img, center, (255, 255, 255), cv2.MARKER_CROSS, 34, 2)
            cv2.circle(display_img, center, 18, (255, 255, 255), 1)
            if target is None:
                cv2.putText(display_img, target_label, (25, frame_h - 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)
                return display_img

            bx, by, bw, bh = target["bbox"]
            color = (0, 255, 0) if locked else (0, 165, 255)
            cv2.rectangle(display_img, (bx, by), (bx + bw, by + bh), color, 3)
            cx, cy = target["center"]
            cv2.drawMarker(display_img, (cx, cy), color, cv2.MARKER_CROSS, 20, 2)
            cv2.line(display_img, (cx, cy), center, color, 2)
            label_y = min(frame_h - 58, max(100, by + bh + 22))
            cv2.putText(display_img, target_label,
                        (max(8, bx), max(98, by - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            cv2.putText(display_img, f"GRID {target['grid_cell']} | {target['range_label']}",
                        (max(8, bx), label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)
            cv2.putText(display_img, f"dx={err_x:+.3f}  dy={err_y:+.3f}  tol=({LOCK_TOLERANCE_X:.3f},{LOCK_TOLERANCE_Y:.3f})",
                        (max(8, bx), min(frame_h - 12, label_y + 22)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
            return display_img

        pump_camera_stream(
            ep_camera, window_name, status_text=status_msg,
            roi_manager=roi_manager, wall_top_y=wall_top_y, overlay_fn=overlay_fn
        )
        if target is not None and fire_on_detection:
            ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
            print(f"🎯 ตรวจพบเป้า {target_index}/{total_targets} ที่ {direction_en} -> หยุดเล็งและเริ่มยิงทันที")
            return False, target
        if locked:
            return True, target

    ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
    if valid_frames == 0:
        print(f"⚠️ เป้า {target_index}: กล้องไม่ส่งเฟรมระหว่างเล็ง")
    elif frames_with_target == 0:
        print(f"⚠️ เป้า {target_index}: กลับมาทิศ {direction_en} แล้ว แต่ตรวจสี/รูปทรงเป้าที่บันทึกไว้ไม่พบ")
    else:
        print(f"⚠️ เป้า {target_index}: พบเป้าระหว่างเล็งแต่ยังไม่เข้าจุดล็อก; คลาดใกล้สุด {best_center_error:.3f}")
    # Leave the final view on screen briefly so the user can inspect the aim point and failure state.
    failure_status = f"AIM FAILED #{target_index}/{total_targets} [{direction_en}] | NO SHOT | inspect center reticle"
    end_view = time.time() + 1.5
    while time.time() < end_view:
        pump_camera_stream(ep_camera, window_name, status_text=failure_status,
                           roi_manager=roi_manager, overlay_fn=overlay_fn)
        time.sleep(0.03)
    return False, None


def scan_direction_targets(ep_camera, color_name="Red", shape_name="Circle", hsv_configs=None,
                           scan_sec=2.5, window_name="RoboMaster 4-Direction Gimbal Scan",
                           direction_info=None, target_memory=None, roi_manager=None, wall_segmenter=None,
                           robot_grid=None, robot_heading=0, grid_size_m=0.6):
    dir_en = direction_info.get("name_en", "FRONT") if direction_info else "FRONT"
    dir_name = direction_info.get("name_th", "ด้านหน้า") if direction_info else "ด้านหน้า"
    dir_yaw = direction_info.get("yaw", 0.0) if direction_info else 0.0
    dir_icon = direction_info.get("icon", "🧭") if direction_info else "🧭"

    print(f"\n{dir_icon} ตรวจสอบ {dir_name} ({dir_yaw}°)...")
    start_t = time.time()
    best_targets_detected = []

    while time.time() - start_t < scan_sec:
        img = None
        try:
            img = ep_camera.read_cv2_image(strategy="newest", timeout=0.10)
        except Exception:
            img = None

        if img is None:
            pump_camera_stream(
                ep_camera, window_name,
                status_text=f"{dir_en} | WAITING FOR FRESH CAMERA FRAME",
                roi_manager=roi_manager,
            )
            time.sleep(0.01)
            continue

        wall_top_y = 0
        targets = detect_targets(img, color_name, shape_name, hsv_configs,
                                 gimbal_pitch=current_gimbal_pitch, roi_manager=roi_manager,
                                 wall_top_y=wall_top_y)

        if robot_grid is not None:
            for target in targets:
                world_grid = estimate_world_grid(
                    target.get("distance_m"), robot_grid, robot_heading,
                    (direction_info or {}).get("id"), grid_size_m
                )
                target["world_grid"] = world_grid
                if world_grid is not None:
                    target["world_grid_label"] = f"({world_grid[0]}, {world_grid[1]})?"
                else:
                    target["world_grid_label"] = "uncertain"

        if target_memory is not None:
            targets = target_memory.annotate_targets(targets, current_gimbal_yaw)

        if len(targets) > len(best_targets_detected):
            best_targets_detected = targets

        def _draw_targets_overlay(disp_frame):
            for idx, t in enumerate(targets):
                bx, by, bw, bh = t["bbox"]
                cv2.rectangle(disp_frame, (bx, by), (bx + bw, by + bh), t["draw_color"], 2)
                map_label = t.get("world_grid_label", "uncertain")
                cv2.putText(disp_frame, f"#{idx+1} GRID {t['grid_cell']} | MAP {map_label}", (bx, max(15, by - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                cv2.putText(disp_frame, t["range_label"], (bx, min(disp_frame.shape[0] - 8, by + bh + 16)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1)
            cv2.putText(disp_frame, f"SCAN: {dir_en} ({dir_yaw:.1f} deg) | Found: {len(targets)} Targets",
                        (30, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2)
            return disp_frame

        pump_camera_stream(ep_camera, window_name, roi_manager=roi_manager, wall_top_y=wall_top_y, overlay_fn=_draw_targets_overlay)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord(' '), 13):
            break
        elif key == ord('q'):
            return None

    print(f"   -> [{dir_name}] พบ {len(best_targets_detected)} เป้าหมาย")
    for idx, target in enumerate(best_targets_detected, start=1):
        map_grid = target.get("world_grid_label", "uncertain")
        range_label = target.get("range_label", "ระยะไม่ทราบ")
        print(
            f"      #{idx}: IMAGE GRID {target['grid_cell']} | MAP GRID ~{map_grid} "
            f"| RANGE {range_label}"
        )
    return best_targets_detected


def render_standby_dashboard(img, summary_results, selected_color, selected_shape, total_shot):
    h, w, _ = img.shape
    filter_desc = format_filter_label(selected_color, selected_shape)

    cv2.rectangle(img, (15, 15), (w - 15, 75), (20, 20, 20), -1)
    cv2.rectangle(img, (15, 15), (w - 15, 75), (0, 255, 150), 2)
    cv2.putText(img, f"SCAN COMPLETED | Targets: {filter_desc} | Shots: {total_shot}",
                (30, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 150), 2)

    cx, cy = w // 2, h // 2 + 20
    box_size = 120
    cv2.rectangle(img, (cx - box_size, cy - box_size), (cx + box_size, cy + box_size), (0, 200, 255), 2)

    f_cnt = summary_results.get("front", {}).get("count", 0)
    r_cnt = summary_results.get("right", {}).get("count", 0)
    l_cnt = summary_results.get("left", {}).get("count", 0)

    cv2.putText(img, f"FRONT: {f_cnt}", (cx - 45, cy - box_size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.putText(img, f"RIGHT: {r_cnt}", (cx + box_size + 10, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.putText(img, f"LEFT: {l_cnt}", (cx - box_size - 95, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    return img


def run_4direction_scan_mission(ep_robot, color_name="ALL", shape_name="ALL",
                                tilt_pitch=-15.0, scan_sec=2.5,
                                auto_fire=False, shots_per_target=2, fire_mode="ir",
                                enabled_directions=None, side_pitch_offset=-1.0,
                                max_retries_per_target=3, fire_on_detection=True,
                                target_physical_size_m=0.0):
    ep_gimbal, ep_blaster = ep_robot.gimbal, ep_robot.blaster
    ep_led, ep_camera = ep_robot.led, ep_robot.camera
    try:
        ep_robot.set_robot_mode(mode=robot.FREE)
    except Exception:
        pass

    window_name = "RoboMaster 4-Direction Gimbal Scan"
    if live_dashboard_callback is None:
        cv2.namedWindow(window_name)
    try:
        ep_blaster.set_led(brightness=0, effect=blaster.LED_OFF)
        ep_gimbal.sub_angle(freq=20, callback=on_gimbal_angle_cb)
    except Exception:
        pass

    hsv_configs = load_hsv_configs()
    if target_physical_size_m and float(target_physical_size_m) > 0:
        for color_cfg in hsv_configs.values():
            color_cfg["physical_size_m"] = float(target_physical_size_m)
    target_memory = TargetMemory()
    roi_manager = ROIManager(roi=DEFAULT_ROI, enabled=True)
    wall_segmenter = None
    directions = FOUR_DIRECTIONS if enabled_directions is None else [
        d for d in FOUR_DIRECTIONS if d["id"] in enabled_directions
    ]
    mission_running = True

    try:
        while mission_running:
            summary = {d["id"]: {"name": d["name_en"], "yaw": d["yaw"], "count": 0} for d in FOUR_DIRECTIONS}
            total_shots = 0
            found_by_direction = []

            # สแกนให้ครบทุกทิศก่อน ยังไม่เริ่มยิงในช่วงนี้
            for direction in directions:
                did, yaw = direction["id"], direction["yaw"]
                name = direction["name_en"]
                pitch = max(-20.0, float(tilt_pitch) + float(side_pitch_offset)) if did in ("right", "left") else float(tilt_pitch)
                turn_and_tilt_direction(ep_gimbal, ep_camera, yaw, pitch, name, window_name, roi_manager)
                targets = scan_direction_targets(
                    ep_camera=ep_camera, color_name=color_name, shape_name=shape_name,
                    hsv_configs=hsv_configs, scan_sec=scan_sec, window_name=window_name,
                    direction_info=direction, target_memory=target_memory,
                    roi_manager=roi_manager, wall_segmenter=wall_segmenter
                )
                if targets is None:
                    mission_running = False
                    break
                summary[did]["count"] = len(targets)
                engage_targets = [t for t in targets if not t.get("is_already_shot", False)]
                if engage_targets:
                    found_by_direction.append((direction, pitch, engage_targets))
                    if not auto_fire:
                        print(f"พบ {len(engage_targets)} เป้าหมายที่ {name} แต่โหมดรายงานอย่างเดียวไม่ยิง")

            if not mission_running:
                break

            # หลังสแกนจบ จึงย้อนกลับไปแต่ละทิศเพื่อเล็งและยิง
            if auto_fire and found_by_direction:
                total_found = sum(len(group[2]) for group in found_by_direction)
                print(f"\n[SCAN COMPLETE] พบ {total_found} เป้าหมาย สแกนครบทุกทิศแล้ว เริ่มเล็งและยิง")
                for direction, pitch, targets in found_by_direction:
                    yaw, name = direction["yaw"], direction["name_en"]
                    print(f"กลับไปที่ {name}: {len(targets)} เป้าหมาย")
                    for idx in range(1, len(targets) + 1):
                        move_gimbal_with_stream(
                            ep_gimbal, ep_camera, pitch, 0, timeout=2.0,
                            window_name=window_name, status_text="RECENTERING BEFORE AIMING...",
                            roi_manager=roi_manager
                        )
                        move_gimbal_with_stream(
                            ep_gimbal, ep_camera, pitch, yaw, timeout=2.0,
                            window_name=window_name,
                            status_text=f"RETURNING TO {name} FOR TARGET {idx}/{len(targets)}...",
                            roi_manager=roi_manager
                        )
                        locked, target = track_and_lock_target(
                            ep_camera=ep_camera, ep_gimbal=ep_gimbal, ep_led=ep_led,
                            color_name=color_name, shape_name=shape_name, hsv_configs=hsv_configs,
                            target_index=idx, total_targets=len(targets), target_memory=target_memory,
                            timeout=12.0, window_name=window_name, direction_en=name,
                            preferred_target=targets[idx - 1],
                            roi_manager=roi_manager, wall_segmenter=wall_segmenter,
                            fire_on_detection=fire_on_detection
                        )
                        if not target or (not fire_on_detection and not locked):
                            print(f"เป้า {idx}/{len(targets)} ที่ {name} มองไม่เห็นในช่วงเล็ง จึงยังยิงไม่ได้")
                            continue
                        if not locked:
                            print(f"เป้า {idx}/{len(targets)} ที่ {name} ตรวจพบแล้ว แต่ยังไม่อยู่กลางจอ -> ยิงตามที่ร้องขอ")
                        aimed_yaw = current_gimbal_yaw
                        _, shots = shoot_until_target_falls(
                            ep_blaster=ep_blaster, ep_led=ep_led, ep_camera=ep_camera,
                            ep_gimbal=ep_gimbal, locked_target=target,
                            color_name=color_name, shape_name=shape_name, hsv_configs=hsv_configs,
                            wall_segmenter=wall_segmenter, target_index=idx,
                            total_targets=len(targets), shots_per_target=shots_per_target,
                            max_retries=max_retries_per_target, fire_mode=fire_mode,
                            roi_manager=roi_manager, window_name=window_name
                        )
                        total_shots += shots
                        target_memory.record_shot(idx, aimed_yaw, target)

            print("\n[สรุปผลการสแกน]")
            for direction in FOUR_DIRECTIONS:
                print(f"{direction['name_en']}: {summary[direction['id']]['count']} เป้าหมาย")
            move_gimbal_with_stream(
                ep_gimbal, ep_camera, float(tilt_pitch), 0, timeout=2.0,
                window_name=window_name, status_text="ROUND COMPLETE - RECENTERING GIMBAL",
                roi_manager=roi_manager
            )

            waiting = True
            while waiting:
                def draw_dashboard(frame):
                    return render_standby_dashboard(frame, summary, color_name, shape_name, total_shots)
                pump_camera_stream(ep_camera, window_name, overlay_fn=draw_dashboard)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    mission_running = False
                    waiting = False
                elif key in (ord(" "), 13):
                    waiting = False
                elif key == ord("r"):
                    target_memory.clear()
    finally:
        try:
            ep_gimbal.unsub_angle()
        except Exception:
            pass
        cv2.destroyAllWindows()


def scan_and_fire_directions_once(ep_robot, color_name="Red", shape_name="Circle",
                                 tilt_pitch=-18, scan_sec=2.5, auto_fire=True,
                                 fire_on_detection=True, shots_per_target=2,
                                 fire_mode="ir", enabled_directions=None,
                                 target_memory=None, target_physical_size_m=0.0,
                                 robot_grid=None, robot_heading=0, grid_size_m=0.6,
                                 sensor_hub=None, on_target=None):
    """Run one finite camera scan/engagement round for grid exploration."""
    ep_gimbal, ep_blaster = ep_robot.gimbal, ep_robot.blaster
    ep_led, ep_camera = ep_robot.led, ep_robot.camera
    window_name = "RoboMaster 4-Direction Gimbal Scan"
    hsv_configs = load_hsv_configs()
    if target_physical_size_m and float(target_physical_size_m) > 0:
        for color_cfg in hsv_configs.values():
            color_cfg["physical_size_m"] = float(target_physical_size_m)
    target_memory = target_memory or TargetMemory()
    roi_manager = ROIManager(roi=DEFAULT_ROI, enabled=True)
    enabled = set(enabled_directions or ("front", "right", "left"))
    directions = [d for d in FOUR_DIRECTIONS if d["id"] in enabled]
    shots_fired = 0
    cancelled = False
    # Without target-size calibration, use the ToF distance measured while
    # the gimbal faces the target direction. Refuse firing when no fresh,
    # valid range is available or the target is beyond the configured limit.
    tof_sample_id = 0
    global chassis_hold_callback
    if live_dashboard_callback is None:
        cv2.namedWindow(window_name)

    previous_hold_callback = chassis_hold_callback
    chassis_hold_callback = None
    try:
        # Match the wall-scan behavior: stop the base once, let it settle, and
        # keep chassis commands out of the SDK channel while gimbal actions run.
        ep_robot.chassis.drive_speed(x=0, y=0, z=0)
        time.sleep(0.25)
        for direction in directions:
            before_state = sensor_hub.get_latest_state() if sensor_hub is not None else None
            tof_sample_id = getattr(before_state, "tof_sample_id", 0)
            did, yaw = direction["id"], direction["yaw"]
            name = direction["name_en"]
            pitch = max(-20.0, float(tilt_pitch) - 1.0) if did in ("right", "left") else float(tilt_pitch)
            turn_and_tilt_direction(
                ep_gimbal, ep_camera, yaw, pitch, name, window_name, roi_manager
            )
            targets = scan_direction_targets(
                ep_camera=ep_camera,
                color_name=color_name,
                shape_name=shape_name,
                hsv_configs=hsv_configs,
                scan_sec=scan_sec,
                window_name=window_name,
                direction_info=direction,
                target_memory=target_memory,
                roi_manager=roi_manager,
                robot_grid=robot_grid,
                robot_heading=robot_heading,
                grid_size_m=grid_size_m,
            )
            if targets is None:
                cancelled = True
                break
            measured_range_m = None
            deadline = time.monotonic() + 1.0
            while sensor_hub is not None and time.monotonic() < deadline:
                range_state = sensor_hub.get_latest_state()
                if (getattr(range_state, "tof_sample_id", 0) > tof_sample_id
                        and getattr(range_state, "tof_valid", False)
                        and getattr(range_state, "tof_filtered_mm", None) is not None):
                    measured_range_m = float(range_state.tof_filtered_mm) / 1000.0
                    break
                time.sleep(0.02)

            for target in targets:
                # ToF measures clearance to the nearest surface, which can be
                # a wall in front of a distant target. Use only calibrated
                # target size for map-cell distance estimates.
                range_m = target.get("distance_m")
                target_pos = target.get("world_grid")
                if on_target is not None:
                    on_target(did, target, range_m, target_pos)

            if not auto_fire:
                continue

            candidates = [
                t for t in targets
                if not t.get("is_already_shot", False)
                and (t.get("distance_m") is not None or measured_range_m is not None)
                # Use the more conservative limit when both optical and ToF
                # ranges exist, so either estimate can stop a long-range shot.
                and (t.get("distance_m") is None
                     or t["distance_m"] <= MAX_TARGET_RANGE_M)
                and (measured_range_m is None
                     or measured_range_m <= MAX_TARGET_RANGE_M)
            ]
            unknown_range = [
                t for t in targets
                if not t.get("is_already_shot", False)
                and t.get("distance_m") is None
                and measured_range_m is None
            ]
            too_far = [
                t for t in targets
                if not t.get("is_already_shot", False)
                and ((t.get("distance_m") is not None
                      and t["distance_m"] > MAX_TARGET_RANGE_M)
                     or (measured_range_m is not None
                         and measured_range_m > MAX_TARGET_RANGE_M))
            ]
            for target in unknown_range:
                print(
                    "  [NO SHOT] ไม่มีระยะจากเป้าหรือ ToF ที่เชื่อถือได้; "
                    "หยุดเพื่อความปลอดภัย."
                )
            for target in too_far:
                ranges = [r for r in (target.get("distance_m"), measured_range_m) if r is not None]
                actual_range = max(ranges)
                print(
                    f"  [NO SHOT] เป้าอยู่ไกลเกิน 2 grids "
                    f"({actual_range:.2f} m > {MAX_TARGET_RANGE_M:.2f} m)."
                )
            for index, candidate in enumerate(candidates, start=1):
                move_gimbal_with_stream(
                    ep_gimbal, ep_camera, pitch, 0, timeout=2.0,
                    window_name=window_name, status_text="RAISED BEFORE AIMING",
                    roi_manager=roi_manager,
                )
                move_gimbal_with_stream(
                    ep_gimbal, ep_camera, pitch, yaw, timeout=2.0,
                    window_name=window_name,
                    status_text=f"RETURNING TO {name} TO AIM",
                    roi_manager=roi_manager,
                )
                locked, target = track_and_lock_target(
                    ep_camera=ep_camera,
                    ep_gimbal=ep_gimbal,
                    ep_led=ep_led,
                    color_name=color_name,
                    shape_name=shape_name,
                    hsv_configs=hsv_configs,
                    target_index=index,
                    total_targets=len(candidates),
                    target_memory=target_memory,
                    timeout=12.0,
                    window_name=window_name,
                    direction_en=name,
                    roi_manager=roi_manager,
                    preferred_target=candidate,
                    fire_on_detection=fire_on_detection,
                )
                if target is None or (not fire_on_detection and not locked):
                    continue
                aimed_yaw = current_gimbal_yaw
                _, count = shoot_until_target_falls(
                    ep_blaster=ep_blaster,
                    ep_led=ep_led,
                    ep_camera=ep_camera,
                    ep_gimbal=ep_gimbal,
                    locked_target=target,
                    color_name=color_name,
                    shape_name=shape_name,
                    hsv_configs=hsv_configs,
                    wall_segmenter=None,
                    target_index=index,
                    total_targets=len(candidates),
                    shots_per_target=shots_per_target,
                    fire_mode=fire_mode,
                    roi_manager=roi_manager,
                    window_name=window_name,
                )
                shots_fired += count
                target_memory.record_shot(index, aimed_yaw, target)
                ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
                sleep_with_stream(ep_camera, window_name, 0.2, roi_manager=roi_manager)
    finally:
        try:
            ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
        except Exception:
            pass
        try:
            ep_robot.chassis.drive_speed(x=0, y=0, z=0)
        except Exception:
            pass
        try:
            move_gimbal_with_stream(
                ep_gimbal, ep_camera, 0, 0, timeout=2.2,
                window_name=window_name, status_text="RETURNING GIMBAL TO FRONT",
                roi_manager=roi_manager,
            )
        finally:
            chassis_hold_callback = previous_hold_callback
            if live_dashboard_callback is None:
                cv2.destroyAllWindows()
    return None if cancelled else shots_fired
