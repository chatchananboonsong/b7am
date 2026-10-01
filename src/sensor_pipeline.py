#!/usr/bin/env python3
"""Sensor pipeline, filters, calibration, and Thread 1 (Sensor Collector) for RoboMaster EP.

Step 2 Requirement:
- Thread 1: Collects raw sensor data (Sharp Left/Right, ToF, IMU, Attitude, Position, Velocity, ESC, Status, Gripper),
  applies filtering (Median, Moving Average / EMA, Outlier rejection), converts to engineering units (mm, deg),
  and exposes clean, ready-to-use snapshots for mapping and real-time controller without redundant hardware calls.
- Telemetry integration: Records time-series data for post-run analysis and mapping.
"""

import collections
import copy
import json
import math
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Calibration Manager
# ---------------------------------------------------------------------------

class CalibrationManager:
    """Loads calibration polynomial curves and converts raw sensor values to mm."""

    def __init__(self, calibration_file: Optional[str] = "calibration_output/calibration.json"):
        self.calibration_file = Path(calibration_file) if calibration_file else None
        self.models: Dict[str, Dict[str, Any]] = {}
        self.load()

    def load(self) -> bool:
        if self.calibration_file:
            path = self.calibration_file
            if not path.exists():
                for cand in [Path("calibration_output/calibration.json"), Path("..") / "calibration_output/calibration.json"]:
                    if cand.exists():
                        path = cand
                        break
            if path.exists():
                try:
                    with path.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                        self.models = data.get("sensors", {})
                        return True
                except Exception as exc:
                    print(f"[CalibrationManager] Warning: failed to load {path}: {exc}")
        return False

    def raw_to_mm(self, sensor_name: str, raw_value: Optional[float]) -> Optional[float]:
        """Converts raw sensor reading to physical distance (mm)."""
        if raw_value is None or not math.isfinite(raw_value):
            return None

        # If polynomial calibration exists for sensor
        if sensor_name in self.models:
            fit = self.models[sensor_name]
            coeffs = fit.get("coefficients", [])
            if coeffs:
                # Polynomial evaluation: c_n * x^n + ... + c_1 * x + c_0
                val = 0.0
                for c in coeffs:
                    val = val * raw_value + c
                # Clamp to reasonable physical bounds
                min_ref = fit.get("reference_min_mm", 20.0)
                max_ref = fit.get("reference_max_mm", 400.0)
                return max(0.0, min(float(val), max_ref * 1.5))

        # Default fallback conversions if calibration model is missing
        if sensor_name == "tof":
            # ToF in RoboMaster SDK is already in mm (or cm depending on firmware; standard is mm)
            return float(raw_value)
        elif sensor_name.startswith("sharp"):
            # Generic 4-30cm Sharp IR inverse approximation if no calibration curve
            if raw_value <= 20:
                return 400.0
            return max(30.0, min(400.0, (1000.0 / (raw_value + 10.0)) * 10.0))
        elif sensor_name == "gripper":
            return float(raw_value)

        return float(raw_value)


# ---------------------------------------------------------------------------
# Filter Implementations
# ---------------------------------------------------------------------------

class MovingAverageFilter:
    """Moving average filter over a sliding window."""

    def __init__(self, window_size: int = 5):
        self.window_size = max(1, window_size)
        self.buffer = collections.deque(maxlen=self.window_size)

    def update(self, value: float) -> float:
        self.buffer.append(value)
        return sum(self.buffer) / len(self.buffer)

    def reset(self):
        self.buffer.clear()


class MedianFilter:
    """Median filter to reject impulsive sensor noise/spikes."""

    def __init__(self, window_size: int = 5):
        self.window_size = max(1, window_size)
        self.buffer = collections.deque(maxlen=self.window_size)

    def update(self, value: float) -> float:
        self.buffer.append(value)
        sorted_vals = sorted(self.buffer)
        n = len(sorted_vals)
        if n % 2 == 1:
            return sorted_vals[n // 2]
        else:
            return (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0

    def reset(self):
        self.buffer.clear()


class ExponentialMovingAverageFilter:
    """Exponential moving average (EMA / Low-pass filter)."""

    def __init__(self, alpha: float = 0.3):
        self.alpha = min(1.0, max(0.01, alpha))
        self.current_value: Optional[float] = None

    def update(self, value: float) -> float:
        if self.current_value is None:
            self.current_value = value
        else:
            self.current_value = self.alpha * value + (1.0 - self.alpha) * self.current_value
        return self.current_value

    def reset(self):
        self.current_value = None


class OutlierRejectionFilter:
    """Rejects out-of-bounds or physically impossible sensor jumps."""

    def __init__(self, min_valid: float, max_valid: float, max_rate_of_change: Optional[float] = None):
        self.min_valid = min_valid
        self.max_valid = max_valid
        self.max_rate_of_change = max_rate_of_change
        self.last_valid: Optional[float] = None

    def update(self, value: float) -> Tuple[float, bool]:
        if not (self.min_valid <= value <= self.max_valid):
            # Out of bounds
            return (self.last_valid if self.last_valid is not None else value, False)

        if self.max_rate_of_change is not None and self.last_valid is not None:
            if abs(value - self.last_valid) > self.max_rate_of_change:
                # Spike detected, reject or limit
                return (self.last_valid, False)

        self.last_valid = value
        return (value, True)

    def reset(self):
        self.last_valid = None


class SensorFilterPipeline:
    """Composite filter pipeline combining Outlier Rejection, Median, and EMA."""

    def __init__(
        self,
        min_valid: float = 0.0,
        max_valid: float = 1023.0,
        median_window: int = 5,
        ema_alpha: float = 0.35,
    ):
        self.outlier = OutlierRejectionFilter(min_valid=min_valid, max_valid=max_valid)
        self.median = MedianFilter(window_size=median_window)
        self.ema = ExponentialMovingAverageFilter(alpha=ema_alpha)

    def filter(self, raw_value: Optional[float]) -> Tuple[Optional[float], bool]:
        if raw_value is None or not math.isfinite(raw_value):
            return None, False

        checked_val, is_valid = self.outlier.update(raw_value)
        median_val = self.median.update(checked_val)
        filtered_val = self.ema.update(median_val)
        return filtered_val, is_valid

    def reset(self):
        self.outlier.reset()
        self.median.reset()
        self.ema.reset()


# ---------------------------------------------------------------------------
# Data Models: Sensor Snapshot & Telemetry Entry
# ---------------------------------------------------------------------------

@dataclass
class RobotSensorSnapshot:
    """Immutable data snapshot of all filtered sensor states at a point in time."""

    timestamp: float = field(default_factory=time.time)
    monotonic_time: float = field(default_factory=time.monotonic)
    frame_index: int = 0

    # Sharp IR sensors (ADC raw & calibrated mm)
    sharp_left_raw: Optional[float] = None
    sharp_left_filtered_raw: Optional[float] = None
    sharp_left_mm: Optional[float] = None
    sharp_left_valid: bool = False

    sharp_right_raw: Optional[float] = None
    sharp_right_filtered_raw: Optional[float] = None
    sharp_right_mm: Optional[float] = None
    sharp_right_valid: bool = False

    # ToF front sensor (mm)
    tof_raw: Optional[float] = None
    tof_filtered_mm: Optional[float] = None
    tof_valid: bool = False
    tof_sample_id: int = 0

    # IMU / Attitude (degrees)
    yaw: float = 0.0      # Relative locked yaw (starts at 0.0 deg on startup)
    yaw_raw: float = 0.0  # Raw IMU yaw from robot SDK
    pitch: float = 0.0
    roll: float = 0.0

    # Chassis Odometry Position (m)
    pos_x: float = 0.0      # Local zeroed X (starts at 0.000 m along initial forward axis)
    pos_y: float = 0.0      # Local zeroed Y (starts at 0.000 m along initial lateral axis)
    pos_z: float = 0.0
    pos_x_raw: float = 0.0  # Raw SDK world X
    pos_y_raw: float = 0.0  # Raw SDK world Y

    # Chassis Velocity (m/s)
    vel_vx: float = 0.0
    vel_vy: float = 0.0
    vel_vz: float = 0.0

    # IMU Accelerometer & Gyroscope
    acc_x: float = 0.0
    acc_y: float = 0.0
    acc_z: float = 0.0
    gyro_x: float = 0.0
    gyro_y: float = 0.0
    gyro_z: float = 0.0

    # ESC Motors
    esc_speeds: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    esc_angles: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])

    # Status flags
    is_static: bool = True
    impact_detected: bool = False
    slip_detected: bool = False
    gimbal_yaw: float = 0.0    # Gimbal yaw relative to chassis (deg)
    gimbal_pitch: float = 0.0  # Gimbal pitch relative to chassis (deg)

    # Derived Wall Classifications for Grid Navigation (Req 3 & 4)
    wall_left_detected: bool = False
    wall_right_detected: bool = False
    wall_front_detected: bool = False
    sharp_diff_mm: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Converts snapshot to dictionary for logging/serialization."""
        return {
            "timestamp": self.timestamp,
            "monotonic_time": self.monotonic_time,
            "frame_index": self.frame_index,
            "sharp_left_raw": self.sharp_left_raw,
            "sharp_left_mm": self.sharp_left_mm,
            "sharp_left_valid": self.sharp_left_valid,
            "sharp_right_raw": self.sharp_right_raw,
            "sharp_right_mm": self.sharp_right_mm,
            "sharp_right_valid": self.sharp_right_valid,
            "tof_raw": self.tof_raw,
            "tof_filtered_mm": self.tof_filtered_mm,
            "tof_valid": self.tof_valid,
            "tof_sample_id": self.tof_sample_id,
            "yaw": self.yaw,
            "yaw_raw": self.yaw_raw,
            "pitch": self.pitch,
            "roll": self.roll,
            "pos_x": self.pos_x,
            "pos_y": self.pos_y,
            "pos_z": self.pos_z,
            "pos_x_raw": self.pos_x_raw,
            "pos_y_raw": self.pos_y_raw,
            "vel_vx": self.vel_vx,
            "vel_vy": self.vel_vy,
            "gimbal_yaw": self.gimbal_yaw,
            "gimbal_pitch": self.gimbal_pitch,
            "wall_left": self.wall_left_detected,
            "wall_right": self.wall_right_detected,
            "wall_front": self.wall_front_detected,
            "sharp_diff_mm": self.sharp_diff_mm,
            "is_static": self.is_static,
        }


# ---------------------------------------------------------------------------
# Sensor Hub (Thread-safe Shared Memory)
# ---------------------------------------------------------------------------

class SensorHub:
    """Thread-safe container providing synchronized sensor snapshots to Thread 2."""

    def __init__(self, max_history: int = 1000):
        self._lock = threading.Lock()
        self._new_data_event = threading.Event()
        self._latest_state = RobotSensorSnapshot()
        self._history = collections.deque(maxlen=max_history)
        self._listeners: List[Callable[[RobotSensorSnapshot], None]] = []

    def update_state(self, snapshot: RobotSensorSnapshot):
        """Thread 1 updates state atomically."""
        with self._lock:
            self._latest_state = snapshot
            self._history.append(snapshot)
            listeners = list(self._listeners)

        self._new_data_event.set()
        for listener in listeners:
            try:
                listener(snapshot)
            except Exception as e:
                print(f"[SensorHub] Listener error: {e}")

    def get_latest_state(self) -> RobotSensorSnapshot:
        """Thread 2 reads the latest clean state safely without calling hardware."""
        with self._lock:
            return self._latest_state

    def wait_for_next_state(self, timeout: Optional[float] = 1.0) -> Optional[RobotSensorSnapshot]:
        """Wait until Thread 1 pushes a new sensor reading."""
        self._new_data_event.clear()
        if self._new_data_event.wait(timeout=timeout):
            return self.get_latest_state()
        return None

    def get_history_snapshot(self) -> List[RobotSensorSnapshot]:
        """Returns a copy of historical snapshots for mapping / analytics."""
        with self._lock:
            return list(self._history)

    def add_listener(self, callback: Callable[[RobotSensorSnapshot], None]):
        """Register a callback when new sensor data arrives."""
        with self._lock:
            self._listeners.append(callback)


# ---------------------------------------------------------------------------
# Thread 1: Sensor Collector & Filter Thread
# ---------------------------------------------------------------------------

class SensorCollectorThread(threading.Thread):
    """Thread 1: Collects raw sensors from RoboMaster SDK, filters, calibrates,
    and updates SensorHub continuously.
    """

    def __init__(
        self,
        sensor_hub: SensorHub,
        robot: Any = None,
        calibration_manager: Optional[CalibrationManager] = None,
        telemetry_recorder: Any = None,
        update_rate_hz: float = 20.0,
        mock_mode: bool = False,
    ):
        super().__init__(name="SensorCollectorThread-1", daemon=True)
        self.sensor_hub = sensor_hub
        self.robot = robot
        self.calibration_manager = calibration_manager or CalibrationManager()
        self.telemetry_recorder = telemetry_recorder
        self.update_interval = 1.0 / max(1.0, update_rate_hz)
        self.mock_mode = mock_mode

        self._running = threading.Event()
        self._frame_count = 0

        # Filter pipelines
        self.sharp_left_filter = SensorFilterPipeline(min_valid=20.0, max_valid=1020.0, median_window=5, ema_alpha=0.35)
        self.sharp_right_filter = SensorFilterPipeline(min_valid=20.0, max_valid=1020.0, median_window=5, ema_alpha=0.35)
        self.tof_filter = SensorFilterPipeline(min_valid=10.0, max_valid=4000.0, median_window=5, ema_alpha=0.40)

        # Internal raw cache updated via RoboMaster SDK callbacks
        self._raw_lock = threading.Lock()
        self._raw_sharp_left: Optional[float] = None
        self._raw_sharp_right: Optional[float] = None
        self._raw_tof: Optional[float] = None
        self._tof_sample_id: int = 0
        self._raw_attitude: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._raw_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._raw_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._raw_imu: Tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self._raw_esc_speeds: List[float] = [0.0, 0.0, 0.0, 0.0]
        self._raw_esc_angles: List[float] = [0.0, 0.0, 0.0, 0.0]
        self._is_static: bool = True
        self._impact: bool = False
        self._slip: bool = False
        self._raw_gimbal_yaw: float = 0.0
        self._raw_gimbal_pitch: float = 0.0
        self._initial_yaw_offset: Optional[float] = None
        self._initial_pos_offset: Optional[Tuple[float, float]] = None

    # SDK Subscription Callbacks
    def _cb_distance(self, distance_info):
        """ToF callback."""
        with self._raw_lock:
            if isinstance(distance_info, (list, tuple)) and len(distance_info) > 0:
                self._raw_tof = float(distance_info[0])
                self._tof_sample_id += 1
            elif isinstance(distance_info, (int, float)):
                self._raw_tof = float(distance_info)
                self._tof_sample_id += 1

    def _cb_adapter(self, adapter_info):
        """Sensor adapter callback (contains IO & ADC for 6 adapter ports)."""
        with self._raw_lock:
            if isinstance(adapter_info, (list, tuple)) and len(adapter_info) >= 2:
                ad_values = adapter_info[1]
                if isinstance(ad_values, (list, tuple)) and len(ad_values) >= 4:
                    # id1 port1 -> index 0 (Left Sharp)
                    # id2 port2 -> index 3 (Right Sharp)
                    self._raw_sharp_left = float(ad_values[0])
                    self._raw_sharp_right = float(ad_values[3])

    def _cb_attitude(self, attitude_info):
        with self._raw_lock:
            if len(attitude_info) >= 3:
                self._raw_attitude = (float(attitude_info[0]), float(attitude_info[1]), float(attitude_info[2]))

    def _cb_position(self, pos_info):
        with self._raw_lock:
            if len(pos_info) >= 3:
                self._raw_position = (float(pos_info[0]), float(pos_info[1]), float(pos_info[2]))

    def _cb_velocity(self, vel_info):
        with self._raw_lock:
            if len(vel_info) >= 3:
                self._raw_velocity = (float(vel_info[0]), float(vel_info[1]), float(vel_info[2]))

    def _cb_imu(self, imu_info):
        with self._raw_lock:
            if len(imu_info) >= 6:
                self._raw_imu = tuple(float(x) for x in imu_info[:6])

    def _cb_esc(self, esc_info):
        with self._raw_lock:
            if len(esc_info) >= 2:
                speeds, angles = esc_info[0], esc_info[1]
                self._raw_esc_speeds = [float(s) for s in speeds]
                self._raw_esc_angles = [float(a) for a in angles]

    def _cb_status(self, status_info):
        with self._raw_lock:
            if len(status_info) >= 6:
                self._is_static = bool(status_info[0])
                self._slip = bool(status_info[5])
                if len(status_info) >= 9:
                    self._impact = any(abs(float(status_info[i])) > 0 for i in (6, 7, 8))

    def _cb_gimbal_angle(self, angle_info):
        with self._raw_lock:
            if isinstance(angle_info, (list, tuple)) and len(angle_info) >= 2:
                self._raw_gimbal_pitch = float(angle_info[0])
                self._raw_gimbal_yaw = float(angle_info[1])

    def setup_subscriptions(self):
        """Subscribes to RoboMaster SDK telemetry streams."""
        if self.mock_mode or self.robot is None:
            return

        try:
            if hasattr(self.robot, "sensor"):
                self.robot.sensor.sub_distance(freq=20, callback=self._cb_distance)
            if hasattr(self.robot, "sensor_adaptor"):
                self.robot.sensor_adaptor.sub_adapter(freq=20, callback=self._cb_adapter)
            if hasattr(self.robot, "chassis"):
                self.robot.chassis.sub_attitude(freq=20, callback=self._cb_attitude)
                self.robot.chassis.sub_position(freq=20, callback=self._cb_position)
                self.robot.chassis.sub_velocity(freq=20, callback=self._cb_velocity)
                self.robot.chassis.sub_imu(freq=20, callback=self._cb_imu)
                self.robot.chassis.sub_esc(freq=20, callback=self._cb_esc)
                self.robot.chassis.sub_status(freq=20, callback=self._cb_status)
            if hasattr(self.robot, "gimbal"):
                self.robot.gimbal.sub_angle(freq=20, callback=self._cb_gimbal_angle)
        except Exception as exc:
            print(f"[SensorCollectorThread] Subscription warning: {exc}")

    def unsubscribe_all(self):
        """Unsubscribes from RoboMaster SDK streams on shutdown."""
        if self.mock_mode or self.robot is None:
            return

        try:
            if hasattr(self.robot, "sensor"):
                self.robot.sensor.unsub_distance()
            if hasattr(self.robot, "sensor_adaptor"):
                self.robot.sensor_adaptor.unsub_adapter()
            if hasattr(self.robot, "chassis"):
                self.robot.chassis.unsub_attitude()
                self.robot.chassis.unsub_position()
                self.robot.chassis.unsub_velocity()
                self.robot.chassis.unsub_imu()
                self.robot.chassis.unsub_esc()
                self.robot.chassis.unsub_status()
            if hasattr(self.robot, "gimbal"):
                self.robot.gimbal.unsub_angle()
        except Exception as exc:
            print(f"[SensorCollectorThread] Unsubscribe warning: {exc}")

    def _poll_adcs_if_needed(self):
        """Direct polling fallback for Sharp sensors if adapter subscription not streaming."""
        if self.mock_mode or self.robot is None:
            return
        if not hasattr(self.robot, "sensor_adaptor"):
            return

        with self._raw_lock:
            need_poll = (self._raw_sharp_left is None or self._raw_sharp_right is None)

        if need_poll:
            try:
                adc_l = self.robot.sensor_adaptor.get_adc(id=1, port=1)
                adc_r = self.robot.sensor_adaptor.get_adc(id=2, port=2)
                with self._raw_lock:
                    if adc_l is not None:
                        self._raw_sharp_left = float(adc_l)
                    if adc_r is not None:
                        self._raw_sharp_right = float(adc_r)
            except Exception:
                pass

    def start_collecting(self):
        self._running.set()
        self.setup_subscriptions()
        self.start()

    def stop_collecting(self):
        self._running.clear()
        self.unsubscribe_all()

    def reset_heading_zero(self, yaw_deg: Optional[float] = None):
        """Sets the zero reference angle to current yaw or specified degree."""
        with self._raw_lock:
            if yaw_deg is not None:
                self._initial_yaw_offset = yaw_deg
            elif self._raw_attitude[0] is not None:
                self._initial_yaw_offset = self._raw_attitude[0]
            else:
                self._initial_yaw_offset = 0.0
        print(f"[SensorCollectorThread] Heading Zero Reset to {self._initial_yaw_offset:.2f}°")

    def reset_position_zero(self, pos_xy: Optional[Tuple[float, float]] = None):
        """Sets origin (0, 0) to current position or specified coordinates."""
        with self._raw_lock:
            if pos_xy is not None:
                self._initial_pos_offset = pos_xy
            elif self._raw_position is not None and len(self._raw_position) >= 2:
                self._initial_pos_offset = (self._raw_position[0], self._raw_position[1])
            else:
                self._initial_pos_offset = (0.0, 0.0)
        print(f"[SensorCollectorThread] Position Zero Reset to ({self._initial_pos_offset[0]:.3f}, {self._initial_pos_offset[1]:.3f}) -> (0, 0)")

    def run(self):
        """Thread 1 main execution loop."""
        while self._running.is_set():
            t_start = time.monotonic()

            if not self.mock_mode:
                self._poll_adcs_if_needed()

            # Acquire snapshot of raw data
            with self._raw_lock:
                raw_sl = self._raw_sharp_left
                raw_sr = self._raw_sharp_right
                raw_tof = self._raw_tof
                tof_sample_id = self._tof_sample_id
                att = self._raw_attitude
                pos = self._raw_position
                vel = self._raw_velocity
                imu = self._raw_imu
                esc_spd = list(self._raw_esc_speeds)
                esc_ang = list(self._raw_esc_angles)
                is_stat = self._is_static
                impact = self._impact
                slip = self._slip
                g_yaw = self._raw_gimbal_yaw
                g_pitch = self._raw_gimbal_pitch

            # Heading (Yaw) Zeroing: Locks initial robot heading to 0.0 deg
            raw_yaw = att[0] if att and len(att) > 0 else 0.0
            if raw_yaw is not None and math.isfinite(raw_yaw):
                if self._initial_yaw_offset is None:
                    self._initial_yaw_offset = raw_yaw
                    print(f"[Thread 1 Sensor] Auto-Locked Initial Heading: {self._initial_yaw_offset:.2f}° -> 0.0°")
                norm_yaw = (raw_yaw - self._initial_yaw_offset + 180.0) % 360.0 - 180.0
            else:
                norm_yaw = 0.0

            # Position (0, 0) Zeroing: Locks initial robot position to (0, 0)
            raw_x = pos[0] if pos and len(pos) > 0 else 0.0
            raw_y = pos[1] if pos and len(pos) > 1 else 0.0
            if self.mock_mode:
                local_x = raw_x
                local_y = raw_y
            else:
                if self._initial_pos_offset is None:
                    self._initial_pos_offset = (raw_x, raw_y)
                    print(f"[Thread 1 Sensor] Auto-Locked Initial Position: ({raw_x:.3f}, {raw_y:.3f}) -> (0.000, 0.000)")

                dx_raw = raw_x - self._initial_pos_offset[0]
                dy_raw = raw_y - self._initial_pos_offset[1]
                # Rotate world odometry into robot's initial baseline frame (where forward = +X)
                theta0_rad = math.radians(self._initial_yaw_offset if self._initial_yaw_offset is not None else 0.0)
                local_x = dx_raw * math.cos(theta0_rad) + dy_raw * math.sin(theta0_rad)
                local_y = -dx_raw * math.sin(theta0_rad) + dy_raw * math.cos(theta0_rad)

            # Filtering raw signals
            filt_sl, sl_valid = self.sharp_left_filter.filter(raw_sl)
            filt_sr, sr_valid = self.sharp_right_filter.filter(raw_sr)
            filt_tof, tof_valid = self.tof_filter.filter(raw_tof)

            # Polynomial calibration conversion to physical units (mm)
            mm_left = self.calibration_manager.raw_to_mm("sharp_left", filt_sl)
            mm_right = self.calibration_manager.raw_to_mm("sharp_right", filt_sr)
            mm_tof = self.calibration_manager.raw_to_mm("tof", filt_tof)

            # Calculate wall detection & lateral alignment difference (Req 3)
            sharp_diff = 0.0
            wall_left = False
            wall_right = False
            wall_front = False

            if mm_left is not None and mm_left < 450.0:
                wall_left = True
            if mm_right is not None and mm_right < 450.0:
                wall_right = True
            if mm_tof is not None and mm_tof < 450.0:
                wall_front = True

            if mm_left is not None and mm_right is not None and wall_left and wall_right:
                sharp_diff = mm_left - mm_right

            self._frame_count += 1

            # Construct clean, immutable snapshot
            snapshot = RobotSensorSnapshot(
                timestamp=time.time(),
                monotonic_time=t_start,
                frame_index=self._frame_count,
                sharp_left_raw=raw_sl,
                sharp_left_filtered_raw=filt_sl,
                sharp_left_mm=mm_left,
                sharp_left_valid=sl_valid and (mm_left is not None),
                sharp_right_raw=raw_sr,
                sharp_right_filtered_raw=filt_sr,
                sharp_right_mm=mm_right,
                sharp_right_valid=sr_valid and (mm_right is not None),
                tof_raw=raw_tof,
                tof_filtered_mm=mm_tof,
                tof_valid=tof_valid and (mm_tof is not None),
                tof_sample_id=tof_sample_id,
                yaw=norm_yaw,
                yaw_raw=raw_yaw,
                pitch=att[1],
                roll=att[2],
                pos_x=local_x,
                pos_y=local_y,
                pos_z=pos[2],
                pos_x_raw=raw_x,
                pos_y_raw=raw_y,
                vel_vx=vel[0],
                vel_vy=vel[1],
                vel_vz=vel[2],
                acc_x=imu[0],
                acc_y=imu[1],
                acc_z=imu[2],
                gyro_x=imu[3],
                gyro_y=imu[4],
                gyro_z=imu[5],
                esc_speeds=esc_spd,
                esc_angles=esc_ang,
                is_static=is_stat,
                impact_detected=impact,
                slip_detected=slip,
                gimbal_yaw=g_yaw,
                gimbal_pitch=g_pitch,
                wall_left_detected=wall_left,
                wall_right_detected=wall_right,
                wall_front_detected=wall_front,
                sharp_diff_mm=sharp_diff,
            )

            # Update shared state in SensorHub for Thread 2
            self.sensor_hub.update_state(snapshot)

            # Record telemetry if recorder is attached
            if self.telemetry_recorder is not None:
                self.telemetry_recorder.record_snapshot(snapshot)

            # Sleep to maintain stable update rate
            elapsed = time.monotonic() - t_start
            sleep_time = self.update_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    # Simulation helper to inject synthetic sensor values
    def inject_mock_data(
        self,
        sharp_left_adc: Optional[float] = None,
        sharp_right_adc: Optional[float] = None,
        tof_dist: Optional[float] = None,
        yaw: float = 0.0,
        pos_x: float = 0.0,
        pos_y: float = 0.0,
        gimbal_yaw: float = 0.0,
        gimbal_pitch: float = 0.0,
    ):
        with self._raw_lock:
            if sharp_left_adc is not None:
                self._raw_sharp_left = sharp_left_adc
            if sharp_right_adc is not None:
                self._raw_sharp_right = sharp_right_adc
            if tof_dist is not None:
                self._raw_tof = tof_dist
            self._raw_attitude = (yaw, self._raw_attitude[1], self._raw_attitude[2])
            self._raw_position = (pos_x, pos_y, self._raw_position[2])
            self._raw_gimbal_yaw = gimbal_yaw
            self._raw_gimbal_pitch = gimbal_pitch
