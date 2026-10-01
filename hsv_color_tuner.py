import json
import os
import sys
import time
import cv2
import numpy as np

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hsv_config.json")

DEFAULT_CONFIG = {
    "Red": {
        "ranges": [
            {"lower": [0, 100, 70], "upper": [10, 255, 255]},
            {"lower": [165, 100, 70], "upper": [180, 255, 255]}
        ],
        "draw_color": [0, 0, 255],
        "led_rgb": [255, 0, 0],
        "name_th": "แดง"
    },
    "Green": {
        "ranges": [
            {"lower": [35, 80, 70], "upper": [85, 255, 255]}
        ],
        "draw_color": [0, 255, 0],
        "led_rgb": [0, 255, 0],
        "name_th": "เขียว"
    },
    "Blue": {
        "ranges": [
            {"lower": [95, 100, 70], "upper": [130, 255, 255]}
        ],
        "draw_color": [255, 130, 0],
        "led_rgb": [0, 100, 255],
        "name_th": "น้ำเงิน"
    },
    "Yellow": {
        "ranges": [
            {"lower": [20, 100, 100], "upper": [35, 255, 255]}
        ],
        "draw_color": [0, 220, 255],
        "led_rgb": [255, 255, 0],
        "name_th": "เหลือง"
    }
}


def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[!] ไม่สามารถอ่าน {CONFIG_FILE} ได้ ({e}) ใช้ค่าเริ่มต้นแทน")
    return DEFAULT_CONFIG.copy()


def save_config(cfg):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"\n>> [SAVED] บันทึกค่าลงไฟล์ '{CONFIG_FILE}' เรียบร้อยแล้ว!")
    except Exception as e:
        print(f"[!] บันทึกไฟล์ล้มเหลว: {e}")


def nothing(x):
    pass


def hsv_to_bgr(h, s, v):
    """แปลงค่า HSV (H:0-179, S:0-255, V:0-255) เป็นสี BGR สำหรับวาดบน OpenCV"""
    h = max(0, min(179, int(h)))
    s = max(0, min(255, int(s)))
    v = max(0, min(255, int(v)))
    pixel = np.uint8([[[h, s, v]]])
    bgr = cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


def describe_hsv_color(h, s, v):
    """
    วิเคราะห์และบอกชื่อเฉดสีโดยประมาณ (ภาษาไทย + อังกฤษ) จากค่า HSV
    """
    if v < 35:
        return "Black / Dark (สีดำ / มืดมาก)"
    if s < 32:
        if v > 200:
            return "White (สีขาว)"
        elif v > 120:
            return "Light Gray (สีเทาสว่าง)"
        else:
            return "Dark Gray (สีเทาเข้ม)"

    # แยกชื่อสีหลักตามค่า Hue (0 - 179)
    if h < 10 or h >= 168:
        base_name = "Red (สีแดง)"
    elif h < 20:
        base_name = "Orange (สีส้ม)"
    elif h < 34:
        base_name = "Yellow (สีเหลือง)"
    elif h < 45:
        base_name = "Yellow-Green (เขียวตองอ่อน/มะนาว)"
    elif h < 78:
        base_name = "Green (สีเขียว)"
    elif h < 95:
        base_name = "Cyan / Teal (สีฟ้าอมเขียว)"
    elif h < 110:
        base_name = "Sky Blue (สีฟ้าสด)"
    elif h < 132:
        base_name = "Blue / Navy (สีน้ำเงิน / กรมท่า)"
    elif h < 150:
        base_name = "Purple / Violet (สีม่วง)"
    elif h < 162:
        base_name = "Magenta (สีบานเย็น)"
    else:
        base_name = "Pink / Crimson (สีชมพูเข้ม)"

    # คุณลักษณะความสดและสว่าง
    adj = ""
    if s < 90:
        adj = "Pale "
    elif s > 180 and v > 180:
        adj = "Bright "
    elif v < 85:
        adj = "Dark "

    return f"{adj}{base_name}"


# ตัวแปรเก็บพิกัดคลิกและค่า HSV ล่าสุด
clicked_hsv = None
clicked_pos = None


def on_mouse_click(event, x, y, flags, param):
    global clicked_hsv, clicked_pos
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv_img = param
        if hsv_img is not None and 0 <= y < hsv_img.shape[0] and 0 <= x < hsv_img.shape[1]:
            h, s, v = hsv_img[y, x]
            clicked_hsv = (int(h), int(s), int(v))
            clicked_pos = (x, y)
            color_desc = describe_hsv_color(h, s, v)
            print(f">> [CLICK] ตำแหน่ง ({x}, {y}) -> HSV = [{h}, {s}, {v}] | {color_desc}")


def init_camera():
    """
    พยายามเชื่อมต่อกล้อง RoboMaster ก่อน หากไม่ได้ต่อหุ่นยนต์จะสลับไปใช้ Webcam ของคอมพิวเตอร์
    """
    print("กำลังตรวจสอบการเชื่อมต่อกล้อง RoboMaster...")
    try:
        from robomaster import robot
        ep_robot = robot.Robot()
        ep_robot.initialize(conn_type="ap")
        ep_camera = ep_robot.camera
        ep_camera.start_video_stream(display=False)
        time.sleep(1)
        test_frame = ep_camera.read_cv2_image(strategy="newest", timeout=1.0)
        if test_frame is not None:
            print(">> เชื่อมต่อกล้อง RoboMaster สำเร็จ!")
            return ("ROBOMASTER", ep_robot, ep_camera)
    except Exception as e:
        print(f">> ไม่พบหรือเชื่อมต่อ RoboMaster ไม่ได้ ({e})")

    print(">> สลับไปใช้กล้องเว็บแคม (Webcam Index 0) สำหรับการจูนสี...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[!] ไม่พบกล้องทั้ง RoboMaster และ Webcam! กรุณาตรวจสอบการเชื่อมต่อ")
        sys.exit(1)
    return ("WEBCAM", None, cap)


def draw_color_palette_card(h_min, h_max, s_min, s_max, v_min, v_max, active_color, config, current_range_idx):
    """
    สร้างรูปการ์ดแสดงผลตัวอย่างสี (Color Palette & Swatches)
    แสดงสีตัวอย่างจริงของช่วง HSV ที่กำลังปรับอย่างละเอียด
    """
    card_w = 540
    card_h = 320
    card = np.zeros((card_h, card_w, 3), dtype=np.uint8)
    card[:] = (28, 30, 36)  # พื้นหลังสีเทาเข้ม Dark Charcoal

    # คำนวณค่ากึ่งกลางช่วง (Midpoint Representative Color)
    h_mid = (h_min + h_max) // 2
    s_mid = (s_min + s_max) // 2
    v_mid = (v_min + v_max) // 2

    mid_bgr = hsv_to_bgr(h_mid, s_mid, v_mid)
    low_bgr = hsv_to_bgr(h_min, s_min, v_min)
    up_bgr = hsv_to_bgr(h_max, s_max, v_max)

    r_mid, g_mid, b_mid = mid_bgr[2], mid_bgr[1], mid_bgr[0]
    hex_mid = f"#{r_mid:02X}{g_mid:02X}{b_mid:02X}"
    color_desc = describe_hsv_color(h_mid, s_mid, v_mid)

    # 1. แถบหัวข้อแสดงสีที่เลือก
    num_ranges = len(config[active_color].get("ranges", []))
    range_txt = f"(Range {current_range_idx+1}/{num_ranges})" if num_ranges > 1 else ""
    cv2.putText(card, f"Target: [{active_color}] {config[active_color]['name_th']} {range_txt}", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    # 2. กล่องตัวอย่างสีหลัก (Representative Color Swatch Box)
    swatch_x1, swatch_y1 = 18, 45
    swatch_x2, swatch_y2 = 148, 125
    cv2.rectangle(card, (swatch_x1, swatch_y1), (swatch_x2, swatch_y2), mid_bgr, -1)
    cv2.rectangle(card, (swatch_x1, swatch_y1), (swatch_x2, swatch_y2), (255, 255, 255), 2)
    cv2.rectangle(card, (swatch_x1 - 1, swatch_y1 - 1), (swatch_x2 + 1, swatch_y2 + 1), (0, 0, 0), 1)

    # ข้อมูลประกอบสีหลัก
    info_x = 165
    cv2.putText(card, f"Tone: {color_desc}", (info_x, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
    cv2.putText(card, f"RGB: ({r_mid}, {g_mid}, {b_mid})  |  Hex: {hex_mid}", (info_x, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 255, 200), 1)
    cv2.putText(card, f"Midpoint HSV: [{h_mid}, {s_mid}, {v_mid}]", (info_x, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 220, 100), 1)

    # 3. ตัวอย่างสีขอบเขตล่าง (Lower) vs ขอบเขตบน (Upper)
    cv2.line(card, (18, 138), (card_w - 18, 138), (60, 65, 75), 1)

    # Lower swatch
    cv2.rectangle(card, (18, 148), (68, 180), low_bgr, -1)
    cv2.rectangle(card, (18, 148), (68, 180), (200, 200, 200), 1)
    cv2.putText(card, f"Lower: [{h_min}, {s_min}, {v_min}]", (78, 168), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1)

    # Upper swatch
    cv2.rectangle(card, (275, 148), (325, 180), up_bgr, -1)
    cv2.rectangle(card, (275, 148), (325, 180), (200, 200, 200), 1)
    cv2.putText(card, f"Upper: [{h_max}, {s_max}, {v_max}]", (335, 168), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1)

    # 4. แถบสเปกตรัมสี Hue Rainbow Spectrum (0 - 179)
    cv2.putText(card, "Hue Spectrum Coverage:", (18, 202), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 180, 180), 1)
    bar_x, bar_y = 18, 210
    bar_w, bar_h = 504, 24

    # สร้างแถบสี Hue 0-179
    x_indices = np.linspace(0, 179, bar_w).astype(np.uint8)
    hsv_gradient = np.zeros((bar_h, bar_w, 3), dtype=np.uint8)
    hsv_gradient[:, :, 0] = x_indices
    hsv_gradient[:, :, 1] = max(140, s_mid)
    hsv_gradient[:, :, 2] = max(160, v_mid)
    bgr_gradient = cv2.cvtColor(hsv_gradient, cv2.COLOR_HSV2BGR)
    card[bar_y:bar_y + bar_h, bar_x:bar_x + bar_w] = bgr_gradient
    cv2.rectangle(card, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 1)

    # วาดตัวชี้ช่วง H_min ถึง H_max บนสเปกตรัม
    pos_min = int(bar_x + (h_min / 179.0) * bar_w)
    pos_max = int(bar_x + (h_max / 179.0) * bar_w)

    cv2.line(card, (pos_min, bar_y - 3), (pos_min, bar_y + bar_h + 3), (0, 0, 0), 3)
    cv2.line(card, (pos_min, bar_y - 3), (pos_min, bar_y + bar_h + 3), (255, 255, 255), 1)
    cv2.line(card, (pos_max, bar_y - 3), (pos_max, bar_y + bar_h + 3), (0, 0, 0), 3)
    cv2.line(card, (pos_max, bar_y - 3), (pos_max, bar_y + bar_h + 3), (255, 255, 255), 1)

    # กรอบเน้นพื้นที่ที่เลือก
    if pos_min < pos_max:
        cv2.rectangle(card, (pos_min, bar_y), (pos_max, bar_y + bar_h), (255, 255, 255), 2)

    cv2.putText(card, f"Active H-Range: {h_min} -> {h_max} (out of 179)", (18, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 220, 100), 1)

    # 5. ตัวอย่างสีจุดที่คลิกเมาส์ (Eyedropper Sample)
    cv2.line(card, (18, 260), (card_w - 18, 260), (60, 65, 75), 1)
    if clicked_hsv is not None:
        ch, cs, cv_val = clicked_hsv
        clicked_bgr = hsv_to_bgr(ch, cs, cv_val)
        click_desc = describe_hsv_color(ch, cs, cv_val)
        cv2.rectangle(card, (18, 270), (58, 305), clicked_bgr, -1)
        cv2.rectangle(card, (18, 270), (58, 305), (255, 255, 255), 1)
        cv2.putText(card, f"Clicked Pixel: HSV [{ch}, {cs}, {cv_val}] | {click_desc}", (68, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1)
        cv2.putText(card, ">> Press [A] to Auto-Fit sliders to this color <<", (68, 303), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
    else:
        cv2.putText(card, "Click any pixel on the video image to sample color (Eyedropper)", (18, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 160, 160), 1)

    return card


def main():
    global clicked_hsv, clicked_pos

    config = load_config()
    color_keys = ["Red", "Green", "Blue", "Yellow"]
    current_color_idx = 3  # เริ่มต้นที่ Yellow
    current_range_idx = 0

    cam_type, ep_robot, cam_source = init_camera()

    window_ctrl = "HSV Tuning Controls"
    window_views = "HSV Live Preview"

    cv2.namedWindow(window_ctrl, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(window_views, cv2.WINDOW_AUTOSIZE)

    # สร้าง Trackbars สำหรับปรับ HSV
    cv2.createTrackbar("H_min", window_ctrl, 0, 179, nothing)
    cv2.createTrackbar("H_max", window_ctrl, 179, 179, nothing)
    cv2.createTrackbar("S_min", window_ctrl, 0, 255, nothing)
    cv2.createTrackbar("S_max", window_ctrl, 255, 255, nothing)
    cv2.createTrackbar("V_min", window_ctrl, 0, 255, nothing)
    cv2.createTrackbar("V_max", window_ctrl, 255, 255, nothing)

    def set_trackbars_from_config(color_name, range_idx=0):
        color_info = config.get(color_name, DEFAULT_CONFIG["Yellow"])
        ranges = color_info.get("ranges", [])
        if range_idx >= len(ranges):
            range_idx = 0
        cur_range = ranges[range_idx]
        low = cur_range["lower"]
        up = cur_range["upper"]

        cv2.setTrackbarPos("H_min", window_ctrl, int(low[0]))
        cv2.setTrackbarPos("S_min", window_ctrl, int(low[1]))
        cv2.setTrackbarPos("V_min", window_ctrl, int(low[2]))
        cv2.setTrackbarPos("H_max", window_ctrl, int(up[0]))
        cv2.setTrackbarPos("S_max", window_ctrl, int(up[1]))
        cv2.setTrackbarPos("V_max", window_ctrl, int(up[2]))

    # โหลดค่าสีเริ่มต้นขึ้น Trackbar
    set_trackbars_from_config(color_keys[current_color_idx], current_range_idx)

    print("\n" + "="*70)
    print("      โปรแกรมจูนและแสดงผลเฉดสี HSV (HSV Color Tuner with Swatches)")
    print("="*70)
    print("  การเลือกสีเพื่อปรับแต่ง:")
    print("   [1] : สีแดง (Red)        [2] : สีเขียว (Green)")
    print("   [3] : สีน้ำเงิน (Blue)    [4] : สีเหลือง (Yellow)")
    print("   [r] : สลับช่วง Range 1 / Range 2 (กรณีสีแดงมี 2 ช่วง)")
    print("  การดูดสีและปรับค่าอัตโนมัติ:")
    print("   - คลิกเมาส์ซ้ายบนภาพ เพื่อดูดค่า HSV และดูตัวอย่างสีจริงทันที")
    print("   [a] : Auto-Fit แถบเลื่อนรอบค่า HSV ของจุดที่คลิก")
    print("   [s] : บันทึกค่า Lower / Upper ลงไฟล์ hsv_config.json")
    print("   [p] : แสดงโค้ด Python (np.array) เพื่อนำไปก๊อปปี้ใช้งาน")
    print("   [q] : ออกจากโปรแกรม")
    print("="*70 + "\n")

    latest_hsv = None

    while True:
        # รับภาพจากกล้อง
        if cam_type == "ROBOMASTER":
            frame = cam_source.read_cv2_image(strategy="newest", timeout=0.5)
        else:
            ret, frame = cam_source.read()
            if not ret:
                continue

        if frame is None:
            continue

        h_orig, w_orig = frame.shape[:2]
        display_w = 460
        display_h = int(h_orig * (display_w / w_orig))
        frame_resized = cv2.resize(frame, (display_w, display_h))

        hsv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)
        latest_hsv = hsv
        cv2.setMouseCallback(window_views, on_mouse_click, latest_hsv)

        # อ่านค่าจาก Trackbars
        h_min = cv2.getTrackbarPos("H_min", window_ctrl)
        s_min = cv2.getTrackbarPos("S_min", window_ctrl)
        v_min = cv2.getTrackbarPos("V_min", window_ctrl)
        h_max = cv2.getTrackbarPos("H_max", window_ctrl)
        s_max = cv2.getTrackbarPos("S_max", window_ctrl)
        v_max = cv2.getTrackbarPos("V_max", window_ctrl)

        lower_bound = np.array([h_min, s_min, v_min])
        upper_bound = np.array([h_max, s_max, v_max])

        active_color = color_keys[current_color_idx]

        # --------------------------------------------------
        # 1. วาดการ์ดแสดงตัวอย่างสีในหน้าต่าง Trackbar Controls
        # --------------------------------------------------
        card_img = draw_color_palette_card(h_min, h_max, s_min, s_max, v_min, v_max, active_color, config, current_range_idx)
        cv2.imshow(window_ctrl, card_img)

        # --------------------------------------------------
        # 2. ประมวลผลภาพ Mask และ Result
        # --------------------------------------------------
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        kernel = np.ones((5, 5), np.uint8)
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_DILATE, kernel)
        result = cv2.bitwise_and(frame_resized, frame_resized, mask=mask_clean)

        # วาด Contour ตรวจจับ
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_view = frame_resized.copy()

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 400:
                perimeter = cv2.arcLength(cnt, True)
                circularity = (4.0 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
                bx, by, bw, bh = cv2.boundingRect(cnt)
                if circularity >= 0.65:
                    shape_str = "Circle"
                else:
                    aspect = float(bw) / float(bh) if bh > 0 else 1.0
                    if 0.85 <= aspect <= 1.18:
                        shape_str = "Square"
                    elif aspect > 1.18:
                        shape_str = "Rect_H"
                    else:
                        shape_str = "Rect_V"
                cv2.rectangle(contours_view, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                cv2.putText(contours_view, f"{shape_str} A:{int(area)}", (bx, max(15, by - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        # วาดมาร์กเกอร์จุดที่คลิก
        if clicked_pos is not None and clicked_hsv is not None:
            cx, cy = clicked_pos
            cv2.circle(contours_view, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(contours_view, f"HSV:{clicked_hsv}", (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # ใส่แถบข้อความบนภาพแต่ละส่วน
        cv2.putText(contours_view, f"Original + Detect [{active_color}]", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2)
        mask_bgr = cv2.cvtColor(mask_clean, cv2.COLOR_GRAY2BGR)
        cv2.putText(mask_bgr, f"Mask: H[{h_min}-{h_max}] S[{s_min}-{s_max}] V[{v_min}-{v_max}]", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1)
        cv2.putText(result, "Isolated Color Result", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2)

        # รวม 3 หน้าจอแนวนอน (Original | Mask | Result)
        combined_view = np.hstack([contours_view, mask_bgr, result])

        # --------------------------------------------------
        # 3. แถบแสดงตัวอย่างสีใต้หน้าจอหลัก (Bottom Color Swatch Footer)
        # --------------------------------------------------
        footer_h = 75
        footer = np.zeros((footer_h, combined_view.shape[1], 3), dtype=np.uint8)
        footer[:] = (22, 24, 28)

        # กล่องตัวอย่างสีที่กำลังปรับ (Swatch Box)
        h_mid = (h_min + h_max) // 2
        s_mid = (s_min + s_max) // 2
        v_mid = (v_min + v_max) // 2
        mid_color_bgr = hsv_to_bgr(h_mid, s_mid, v_mid)
        shade_name = describe_hsv_color(h_mid, s_mid, v_mid)
        r_m, g_m, b_m = mid_color_bgr[2], mid_color_bgr[1], mid_color_bgr[0]
        hex_code = f"#{r_m:02X}{g_m:02X}{b_m:02X}"

        # วาดกล่องสีตัวอย่างใหญ่ทางซ้าย
        cv2.rectangle(footer, (15, 10), (75, 65), mid_color_bgr, -1)
        cv2.rectangle(footer, (15, 10), (75, 65), (255, 255, 255), 2)

        # ข้อความระบุเฉดสี
        cv2.putText(footer, f"Selected Tone: {shade_name} | Hex: {hex_code} | RGB: ({r_m}, {g_m}, {b_m})", (88, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1)
        cv2.putText(footer, f"Lower: [{h_min}, {s_min}, {v_min}]  <--->  Upper: [{h_max}, {s_max}, {v_max}]  |  [S]: Save Config  |  [1-4]: Switch Color", (88, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)

        # กล่องตัวอย่างสีที่คลิกดูดไว้ (ถ้ามี)
        if clicked_hsv is not None:
            ch, cs, cv_val = clicked_hsv
            c_bgr = hsv_to_bgr(ch, cs, cv_val)
            c_pos_x = combined_view.shape[1] - 180
            cv2.rectangle(footer, (c_pos_x, 12), (c_pos_x + 45, 62), c_bgr, -1)
            cv2.rectangle(footer, (c_pos_x, 12), (c_pos_x + 45, 62), (255, 255, 255), 1)
            cv2.putText(footer, "Clicked", (c_pos_x + 52, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
            cv2.putText(footer, f"[{ch},{cs},{cv_val}]", (c_pos_x + 52, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
            cv2.putText(footer, "[A]:AutoFit", (c_pos_x + 52, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 255, 180), 1)

        screen_final = np.vstack([combined_view, footer])
        cv2.imshow(window_views, screen_final)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("ปิดโปรแกรม HSV Tuner")
            break

        # เลือกสี 1=Red, 2=Green, 3=Blue, 4=Yellow
        elif key in (ord('1'), ord('2'), ord('3'), ord('4')):
            current_color_idx = key - ord('1')
            current_range_idx = 0
            active_color = color_keys[current_color_idx]
            print(f"\n>> เปลี่ยนเป้าหมายการจูนสีเป็น: '{active_color}' ({config[active_color]['name_th']})")
            set_trackbars_from_config(active_color, current_range_idx)

        # สลับช่วง Range สำหรับสีที่มีหลายช่วง (เช่น Red)
        elif key in (ord('r'), ord('R')):
            active_color = color_keys[current_color_idx]
            ranges = config[active_color].get("ranges", [])
            if len(ranges) > 1:
                current_range_idx = (current_range_idx + 1) % len(ranges)
                print(f">> สลับไปช่วง Range {current_range_idx+1} ของสี {active_color}")
                set_trackbars_from_config(active_color, current_range_idx)
            else:
                print(f">> สี {active_color} มีเพียง 1 ช่วง (Range 1)")

        # Auto-fit ตามจุดที่คลิก
        elif key in (ord('a'), ord('A')):
            if clicked_hsv is not None:
                ch, cs, cv_val = clicked_hsv
                h_low = max(0, ch - 12)
                h_up = min(179, ch + 12)
                s_low = max(40, cs - 60)
                s_up = min(255, cs + 60)
                v_low = max(50, cv_val - 60)
                v_up = min(255, cv_val + 60)

                cv2.setTrackbarPos("H_min", window_ctrl, h_low)
                cv2.setTrackbarPos("H_max", window_ctrl, h_up)
                cv2.setTrackbarPos("S_min", window_ctrl, s_low)
                cv2.setTrackbarPos("S_max", window_ctrl, s_up)
                cv2.setTrackbarPos("V_min", window_ctrl, v_low)
                cv2.setTrackbarPos("V_max", window_ctrl, v_up)
                print(f">> [AUTO-FIT] ปรับ Trackbar อัตโนมัติรอบจุดคลิก HSV {clicked_hsv}:")
                print(f"   H: [{h_low}, {h_up}], S: [{s_low}, {s_up}], V: [{v_low}, {v_up}]")
            else:
                print(">> กรุณาใช้เมาส์คลิกบนวัตถุสีที่ต้องการก่อนกด 'a'")

        # บันทึกค่าลง config
        elif key in (ord('s'), ord('S')):
            active_color = color_keys[current_color_idx]
            if "ranges" not in config[active_color]:
                config[active_color]["ranges"] = []

            while len(config[active_color]["ranges"]) <= current_range_idx:
                config[active_color]["ranges"].append({"lower": [0, 0, 0], "upper": [179, 255, 255]})

            config[active_color]["ranges"][current_range_idx]["lower"] = [int(h_min), int(s_min), int(v_min)]
            config[active_color]["ranges"][current_range_idx]["upper"] = [int(h_max), int(s_max), int(v_max)]
            save_config(config)

        # แสดงโค้ด Python
        elif key in (ord('p'), ord('P')):
            active_color = color_keys[current_color_idx]
            print("\n" + "-"*50)
            print(f"# โค้ด NumPy สำหรับสี: {active_color}")
            print(f"LOWER_{active_color.upper()} = np.array([{h_min}, {s_min}, {v_min}])")
            print(f"UPPER_{active_color.upper()} = np.array([{h_max}, {s_max}, {v_max}])")
            print("-"*50)

    # ปิดทรัพยากร
    cv2.destroyAllWindows()
    if cam_type == "ROBOMASTER":
        try:
            cam_source.stop_video_stream()
            ep_robot.close()
        except Exception:
            pass
    else:
        cam_source.release()

    print("ปิดโปรแกรม HSV Tuner สำเร็จ")


if __name__ == "__main__":
    main()
