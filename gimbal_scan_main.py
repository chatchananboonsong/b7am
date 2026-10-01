import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

_BUNDLED_SDK_SRC = Path(__file__).resolve().parent / "RoboMaster-SDK" / "src"
if _BUNDLED_SDK_SRC.is_dir() and str(_BUNDLED_SDK_SRC) not in sys.path:
    sys.path.insert(0, str(_BUNDLED_SDK_SRC))

from robomaster import robot
from gimbal_scan_controller import (
    run_4direction_scan_mission,
    format_filter_label,
    AVAILABLE_COLORS,
    AVAILABLE_SHAPES,
)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


class GimbalScanConfigDialog:
    def __init__(self, default_color="Red", default_shape="Circle", default_pitch=-18):
        self.selected_color = default_color
        self.selected_shape = default_shape
        self.tilt_pitch = default_pitch
        self.scan_sec = 2.5
        self.auto_fire = True
        self.fire_on_detection = True
        self.shots = 2
        self.target_size_cm = 0.0
        self.fire_mode = "ir"
        self.enabled_directions = ["front", "right", "left"]
        self.is_confirmed = False

        self.root = tk.Tk()
        self.root.title("🎯 ตั้งค่าภารกิจ Gimbal ก้มตรวจเช็กเป้าหมาย 3 ทิศทาง (RoboMaster 3-Direction Scan)")
        self.root.geometry("720x760")
        self.root.resizable(False, False)
        self.root.configure(bg="#1E1E2E")

        self.color_vars = {}
        for cid, _, _, _ in AVAILABLE_COLORS:
            self.color_vars[cid] = tk.BooleanVar(value=(cid == default_color if default_color != "ALL" else True))
        self.color_all_var = tk.BooleanVar(value=(default_color == "ALL"))

        self.shape_vars = {}
        for sid, _, _ in AVAILABLE_SHAPES:
            self.shape_vars[sid] = tk.BooleanVar(value=(sid == default_shape if default_shape != "ALL" else True))
        self.shape_all_var = tk.BooleanVar(value=(default_shape == "ALL"))

        self.pitch_var = tk.IntVar(value=self.tilt_pitch)
        self.scan_sec_var = tk.DoubleVar(value=self.scan_sec)
        self.auto_fire_var = tk.BooleanVar(value=True)
        self.fire_on_detection_var = tk.BooleanVar(value=True)
        self.shots_var = tk.IntVar(value=self.shots)
        self.target_size_cm_var = tk.DoubleVar(value=self.target_size_cm)
        self.fire_mode_var = tk.StringVar(value="ir")

        self.dir_vars = {
            "front": tk.BooleanVar(value=True),
            "right": tk.BooleanVar(value=True),
            "left":  tk.BooleanVar(value=True),
        }

        self._build_ui()

    def _build_ui(self):
        header = tk.Frame(self.root, bg="#2A2A3E", padx=16, pady=12)
        header.pack(fill="x")

        tk.Label(header, text="🧭 RoboMaster 3-Direction Gimbal Tilt & Scan",
                 font=("Segoe UI", 13, "bold"), fg="#00FFAA", bg="#2A2A3E").pack(anchor="w")
        tk.Label(header, text="ก้ม Gimbal ตามองศาที่กำหนด -> ตรวจเช็กเป้าหมาย 3 ทิศทาง -> สรุปรายงาน / เล็งยิง",
                 font=("Tahoma", 9), fg="#B0B0C8", bg="#2A2A3E").pack(anchor="w", pady=(2, 0))

        content = tk.Frame(self.root, bg="#1E1E2E", padx=18, pady=10)
        content.pack(fill="both", expand=True)

        quick_frame = tk.LabelFrame(content, text=" ⚡ เลือกประเภทเป้าหมายด่วน ",
                                    font=("Tahoma", 9, "bold"), fg="#FFDD00", bg="#252538", padx=8, pady=6)
        quick_frame.pack(fill="x", pady=(0, 8))

        presets = [
            ("🔴 วงกลม สีแดง", ["Red"], ["Circle"], "#FF4444"),
            ("🟢 วงกลม สีเขียว", ["Green"], ["Circle"], "#2ECC71"),
            ("🔵 วงกลม สีน้ำเงิน", ["Blue"], ["Circle"], "#3498DB"),
            ("🟡 วงกลม สีเหลือง", ["Yellow"], ["Circle"], "#F1C40F"),
            ("⬛ สี่เหลี่ยม สีแดง", ["Red"], ["Square"], "#FF6666"),
            ("▰ ผืนผ้านอน สีแดง", ["Red"], ["Rect_H"], "#FFAA44"),
            ("▮ ผืนผ้าตั้ง สีน้ำเงิน", ["Blue"], ["Rect_V"], "#44AAFF"),
            ("🎯 ทุกเป้าหมาย (ทุกสี/ทุกทรง)", "ALL", "ALL", "#00FFAA"),
        ]

        grid = tk.Frame(quick_frame, bg="#252538")
        grid.pack(fill="x")
        for i, (label, c, s, col) in enumerate(presets):
            btn = tk.Button(grid, text=label, font=("Tahoma", 8, "bold"), fg=col, bg="#1E1E2E",
                            activebackground="#3A3A55", relief="groove", bd=1, pady=3, cursor="hand2",
                            command=lambda c_val=c, s_val=s: self._set_preset(c_val, s_val))
            btn.grid(row=i // 4, column=i % 4, padx=3, pady=3, sticky="ew")
        for c_idx in range(4):
            grid.columnconfigure(c_idx, weight=1)

        opt_box = tk.LabelFrame(content, text=" 🎨 สีและรูปทรงที่ต้องการตรวจเช็ก ",
                                font=("Tahoma", 9, "bold"), fg="#00E5FF", bg="#252538", padx=8, pady=5)
        opt_box.pack(fill="x", pady=(0, 8))

        row_c = tk.Frame(opt_box, bg="#252538")
        row_c.pack(fill="x", pady=2)
        tk.Label(row_c, text="สีเป้าหมาย:", font=("Tahoma", 9, "bold"), fg="white", bg="#252538", width=12, anchor="w").pack(side="left")
        
        tk.Checkbutton(row_c, text="🌈 ทุกสี (ALL)", variable=self.color_all_var,
                       command=self._on_toggle_color_all,
                       font=("Tahoma", 8, "bold"), fg="#00FFAA", bg="#252538", selectcolor="#14141E").pack(side="left", padx=3)

        for cid, cth, chex, _ in AVAILABLE_COLORS:
            tk.Checkbutton(row_c, text=cth, variable=self.color_vars[cid],
                           command=self._on_individual_color_change,
                           font=("Tahoma", 8, "bold"), fg=chex, bg="#252538", selectcolor="#14141E").pack(side="left", padx=2)

        row_s = tk.Frame(opt_box, bg="#252538")
        row_s.pack(fill="x", pady=2)
        tk.Label(row_s, text="รูปทรงเป้า:", font=("Tahoma", 9, "bold"), fg="white", bg="#252538", width=12, anchor="w").pack(side="left")
        
        tk.Checkbutton(row_s, text="⭐ ทุกรูปทรง (ALL)", variable=self.shape_all_var,
                       command=self._on_toggle_shape_all,
                       font=("Tahoma", 8, "bold"), fg="#00FFAA", bg="#252538", selectcolor="#14141E").pack(side="left", padx=3)

        for sid, sth, sicon in AVAILABLE_SHAPES:
            tk.Checkbutton(row_s, text=f"{sicon} {sth}", variable=self.shape_vars[sid],
                           command=self._on_individual_shape_change,
                           font=("Tahoma", 8), fg="#EAEAEA", bg="#252538", selectcolor="#14141E").pack(side="left", padx=2)

        gimbal_box = tk.LabelFrame(content, text=" 📐 การตั้งค่า Gimbal ก้ม & ทิศทางสแกน ",
                                   font=("Tahoma", 9, "bold"), fg="#FFAA00", bg="#252538", padx=8, pady=6)
        gimbal_box.pack(fill="x", pady=(0, 8))

        pitch_row = tk.Frame(gimbal_box, bg="#252538")
        pitch_row.pack(fill="x", pady=3)
        tk.Label(pitch_row, text="มุมก้ม Gimbal (Pitch):", font=("Tahoma", 9, "bold"), fg="#FFDD66", bg="#252538").pack(side="left", padx=(0, 6))

        pitch_opts = [
            ("-18° (ก้มมากขึ้น)", -18),
            ("-15°", -15),
            ("-16°", -16),
            ("-13°", -13),
            ("-10°", -10),
        ]
        for p_label, p_val in pitch_opts:
            tk.Radiobutton(pitch_row, text=p_label, value=p_val, variable=self.pitch_var,
                           font=("Tahoma", 8), fg="#00FFAA" if p_val == -18 else "white",
                           bg="#252538", selectcolor="#14141E").pack(side="left", padx=2)

        tk.Label(pitch_row, text="| ระบุ:", font=("Tahoma", 8), fg="#AAAAAA", bg="#252538").pack(side="left", padx=(4, 2))
        tk.Spinbox(pitch_row, from_=-20, to=0, width=4, textvariable=self.pitch_var,
                   font=("Tahoma", 9, "bold"), bg="#1E1E2E", fg="#00FFAA").pack(side="left")
        tk.Label(pitch_row, text="°", font=("Tahoma", 9, "bold"), fg="#00FFAA", bg="#252538").pack(side="left")

        dir_row = tk.Frame(gimbal_box, bg="#252538")
        dir_row.pack(fill="x", pady=4)
        tk.Label(dir_row, text="ทิศทาง:", font=("Tahoma", 9, "bold"), fg="#FFDD66", bg="#252538").pack(side="left", padx=(0, 8))

        dir_labels = [
            ("front", "⬆️ หน้า (0°)"),
            ("right", "➡️️ ขวา (90°)"),
            ("left",  "⬅️ ซ้าย (-90°)"),
        ]
        for d_id, d_lbl in dir_labels:
            tk.Checkbutton(dir_row, text=d_lbl, variable=self.dir_vars[d_id],
                           font=("Tahoma", 8, "bold"), fg="#E0E0FF", bg="#252538", selectcolor="#14141E").pack(side="left", padx=5)

        time_row = tk.Frame(gimbal_box, bg="#252538")
        time_row.pack(fill="x", pady=2)
        tk.Label(time_row, text="เวลาตรวจต่อทิศ:", font=("Tahoma", 8), fg="#CCCCCC", bg="#252538").pack(side="left")
        tk.Spinbox(time_row, from_=1.0, to=10.0, increment=0.5, format="%.1f", width=5,
                   textvariable=self.scan_sec_var, font=("Tahoma", 9, "bold"), bg="#1E1E2E", fg="#00FFAA").pack(side="left", padx=4)
        tk.Label(time_row, text="วินาที", font=("Tahoma", 8), fg="#999999", bg="#252538").pack(side="left")

        action_box = tk.LabelFrame(content, text=" 💥 รูปแบบการกระทำ ",
                                   font=("Tahoma", 9, "bold"), fg="#FF5577", bg="#252538", padx=8, pady=6)
        action_box.pack(fill="x", pady=(0, 8))

        mode_row = tk.Frame(action_box, bg="#252538")
        mode_row.pack(fill="x", pady=2)

        tk.Radiobutton(mode_row, text="🔍 รายงานผลอย่างเดียว (Scan & Report)",
                       variable=self.auto_fire_var, value=False,
                       font=("Tahoma", 9, "bold"), fg="#00E5FF", bg="#252538", selectcolor="#14141E").pack(anchor="w", pady=2)

        tk.Radiobutton(mode_row, text="💥 ล็อกและยิงอัตโนมัติ (Auto-Aim & Fire)",
                       variable=self.auto_fire_var, value=True,
                       font=("Tahoma", 9, "bold"), fg="#FF5555", bg="#252538", selectcolor="#14141E").pack(anchor="w", pady=2)

        fire_timing_row = tk.Frame(action_box, bg="#252538")
        fire_timing_row.pack(fill="x", pady=2)
        tk.Label(fire_timing_row, text="จังหวะยิง:", font=("Tahoma", 8, "bold"), fg="#FFD700", bg="#252538").pack(side="left")
        tk.Radiobutton(fire_timing_row, text="ยิงทันทีเมื่อพบเป้า", value=True,
                       variable=self.fire_on_detection_var, font=("Tahoma", 8, "bold"),
                       fg="#FF7777", bg="#252538", selectcolor="#14141E").pack(side="left", padx=5)
        tk.Radiobutton(fire_timing_row, text="เล็งกลางจอก่อนยิง", value=False,
                       variable=self.fire_on_detection_var, font=("Tahoma", 8),
                       fg="white", bg="#252538", selectcolor="#14141E").pack(side="left", padx=5)

        fire_opt_row = tk.Frame(action_box, bg="#252538")
        fire_opt_row.pack(fill="x", pady=3)
        tk.Label(fire_opt_row, text="ยิงเป้าละ:", font=("Tahoma", 8, "bold"), fg="#FFD700", bg="#252538").pack(side="left")
        tk.Spinbox(fire_opt_row, from_=1, to=5, width=4, textvariable=self.shots_var,
                   font=("Tahoma", 9, "bold"), bg="#1E1E2E", fg="#00FFAA").pack(side="left", padx=4)
        tk.Label(fire_opt_row, text="| ขนาดจริงเป้า (ซม.):", font=("Tahoma", 8), fg="#CCCCCC", bg="#252538").pack(side="left", padx=2)
        tk.Spinbox(fire_opt_row, from_=0, to=200, increment=0.5, width=5,
                   textvariable=self.target_size_cm_var, font=("Tahoma", 9, "bold"),
                   bg="#1E1E2E", fg="#00FFAA").pack(side="left", padx=3)
        tk.Label(action_box, text="ใส่เส้นผ่านศูนย์กลาง/ความสูงจริงของเป้าเพื่อคำนวณระยะ; ใส่ 0 = ไม่ทราบระยะและงดยิง",
                 font=("Tahoma", 8), fg="#FFCC66", bg="#252538").pack(anchor="w", pady=(2, 0))
        tk.Label(fire_opt_row, text="นัด | โหมด:", font=("Tahoma", 8), fg="#CCCCCC", bg="#252538").pack(side="left", padx=2)

        tk.Radiobutton(fire_opt_row, text="IR", value="ir", variable=self.fire_mode_var,
                       font=("Tahoma", 8), fg="white", bg="#252538", selectcolor="#14141E").pack(side="left", padx=4)
        tk.Radiobutton(fire_opt_row, text="Water Gel", value="water", variable=self.fire_mode_var,
                       font=("Tahoma", 8), fg="#00FFAA", bg="#252538", selectcolor="#14141E").pack(side="left", padx=4)
        tk.Radiobutton(fire_opt_row, text="Dual", value="dual", variable=self.fire_mode_var,
                       font=("Tahoma", 8), fg="#FFAA00", bg="#252538", selectcolor="#14141E").pack(side="left", padx=4)

        btn_box = tk.Frame(self.root, bg="#1E1E2E", pady=10, padx=20)
        btn_box.pack(fill="x")

        start_btn = tk.Button(btn_box, text="🚀 เริ่มภารกิจ (START)",
                              font=("Segoe UI", 11, "bold"), bg="#00C853", fg="white",
                              activebackground="#00E676", pady=8, cursor="hand2", relief="flat", command=self._on_start)
        start_btn.pack(side="left", fill="x", expand=True, padx=(0, 10))

        cancel_btn = tk.Button(btn_box, text="ออก (Exit)", font=("Tahoma", 9),
                               bg="#555566", fg="white", activebackground="#777788", pady=8, padx=16,
                               cursor="hand2", relief="flat", command=self._on_cancel)
        cancel_btn.pack(side="right")

    def _on_toggle_color_all(self):
        state = self.color_all_var.get()
        for var in self.color_vars.values():
            var.set(state)

    def _on_individual_color_change(self):
        all_checked = all(var.get() for var in self.color_vars.values())
        self.color_all_var.set(all_checked)

    def _on_toggle_shape_all(self):
        state = self.shape_all_var.get()
        for var in self.shape_vars.values():
            var.set(state)

    def _on_individual_shape_change(self):
        all_checked = all(var.get() for var in self.shape_vars.values())
        self.shape_all_var.set(all_checked)

    def _set_preset(self, color, shape):
        if color == "ALL":
            self.color_all_var.set(True)
            self._on_toggle_color_all()
        else:
            c_list = color if isinstance(color, list) else [color]
            for cid, var in self.color_vars.items():
                var.set(cid in c_list)
            self._on_individual_color_change()

        if shape == "ALL":
            self.shape_all_var.set(True)
            self._on_toggle_shape_all()
        else:
            s_list = shape if isinstance(shape, list) else [shape]
            for sid, var in self.shape_vars.items():
                var.set(sid in s_list)
            self._on_individual_shape_change()

    def _on_start(self):
        if self.color_all_var.get():
            self.selected_color = "ALL"
        else:
            selected_c = [cid for cid, var in self.color_vars.items() if var.get()]
            if not selected_c or len(selected_c) == len(self.color_vars):
                self.selected_color = "ALL"
            elif len(selected_c) == 1:
                self.selected_color = selected_c[0]
            else:
                self.selected_color = selected_c

        if self.shape_all_var.get():
            self.selected_shape = "ALL"
        else:
            selected_s = [sid for sid, var in self.shape_vars.items() if var.get()]
            if not selected_s or len(selected_s) == len(self.shape_vars):
                self.selected_shape = "ALL"
            elif len(selected_s) == 1:
                self.selected_shape = selected_s[0]
            else:
                self.selected_shape = selected_s

        self.tilt_pitch = max(-20, min(0, int(self.pitch_var.get())))
        self.scan_sec = max(1.0, float(self.scan_sec_var.get()))
        self.auto_fire = self.auto_fire_var.get()
        self.fire_on_detection = self.fire_on_detection_var.get()
        self.shots = max(1, min(5, self.shots_var.get()))
        self.target_size_cm = max(0.0, min(200.0, float(self.target_size_cm_var.get())))
        self.fire_mode = self.fire_mode_var.get()

        self.enabled_directions = [d_id for d_id, var in self.dir_vars.items() if var.get()]
        if not self.enabled_directions:
            self.enabled_directions = ["front", "right", "left"]

        self.is_confirmed = True
        self.root.destroy()

    def _on_cancel(self):
        self.is_confirmed = False
        self.root.destroy()

    def run(self):
        self.root.mainloop()
        return (
            self.selected_color,
            self.selected_shape,
            self.tilt_pitch,
            self.scan_sec,
            self.auto_fire,
            self.fire_on_detection,
            self.shots,
            self.target_size_cm,
            self.fire_mode,
            self.enabled_directions,
            self.is_confirmed
        )


def main():
    print("="*80)
    print("  🧭 RoboMaster 3-Direction Gimbal Tilt & Scan System")
    print("="*80)

    dialog = GimbalScanConfigDialog(default_color="Red", default_shape="Circle", default_pitch=-18)
    (
        target_color,
        target_shape,
        tilt_pitch,
        scan_sec,
        auto_fire,
        fire_on_detection,
        shots_per_target,
        target_size_cm,
        fire_mode,
        enabled_dirs,
        confirmed
    ) = dialog.run()

    if not confirmed:
        print(">> ผู้ใช้กดยกเลิกภารกิจ -> ปิดโปรแกรม")
        return

    filter_label = format_filter_label(target_color, target_shape)
    print(f"\n📋 [เป้าหมาย]: {filter_label}")
    print(f"📐 [Pitch]: {tilt_pitch}° | 🧭 [ทิศทาง]: {', '.join(enabled_dirs)}")
    print(f"⚙️ [โหมด]: {'Auto-Aim & Fire' if auto_fire else 'Scan & Report'}")
    if auto_fire:
        print(f"🎯 [จังหวะยิง]: {'ยิงทันทีเมื่อพบเป้า' if fire_on_detection else 'เล็งกลางจอก่อนยิง'}")

    print("\nกำลังเชื่อมต่อกับหุ่นยนต์ RoboMaster EP...")
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    try:
        print("กำลังเปิดระบบสตรีมกล้อง...")
        ep_robot.camera.start_video_stream(display=False)
        
        img_ready = False
        print("รอสัญญาณภาพจากกล้อง...")
        for i in range(40):
            try:
                frame = ep_robot.camera.read_cv2_image(strategy="newest", timeout=0.1)
                if frame is not None:
                    img_ready = True
                    print(f"✅ เชื่อมต่อภาพสำเร็จใน {i*0.1:.1f} วินาที!")
                    break
            except Exception:
                pass
            time.sleep(0.1)

        if not img_ready:
            print("❌ ไม่สามารถรับสัญญาณภาพจากกล้องได้")
            return

        run_4direction_scan_mission(
            ep_robot=ep_robot,
            color_name=target_color,
            shape_name=target_shape,
            tilt_pitch=tilt_pitch,
            scan_sec=scan_sec,
            auto_fire=auto_fire,
            fire_on_detection=fire_on_detection,
            shots_per_target=shots_per_target,
            fire_mode=fire_mode,
            enabled_directions=enabled_dirs,
            target_physical_size_m=target_size_cm / 100.0,
        )

    except KeyboardInterrupt:
        print("\n[!] หยุดการทำงานด้วย Ctrl+C")
    except Exception as e:
        print(f"\n[!] เกิดข้อผิดพลาด: {e}")
    finally:
        print("กำลังปิดการเชื่อมต่อหุ่นยนต์...")
        try:
            ep_robot.camera.stop_video_stream()
            ep_robot.close()
        except Exception:
            pass
        print(">> สิ้นสุดการทำงานอย่างปลอดภัย")


if __name__ == "__main__":
    main()





