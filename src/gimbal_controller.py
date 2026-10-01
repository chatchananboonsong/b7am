#!/usr/bin/env python3
"""RoboMaster EP Gimbal Controller for 4-Direction Wall Scanning.

Replaces the gripper system with active gimbal rotation (Front, Right, Back, Left)
to detect walls in all 4 cardinal directions using the ToF distance sensor.
"""

import statistics
import time
from typing import Any, Callable, Dict, Optional, Tuple


class GimbalScanController:
    """Controls RoboMaster EP Gimbal to scan surroundings and detect walls in 4 directions."""

    # Relative yaw angles (degrees): Front=0, Right=-90, Back=180, Left=+90
    SCAN_ANGLES = {
        "front": 0.0,
        "right": 90.0,
        "back": 180.0,
        "left": -90.0,
    }

    WALL_DETECT_THRESHOLD_MM = 450.0  # 45 cm
    FRESH_TOF_SAMPLES = 3
    FRESH_TOF_TIMEOUT_SEC = 1.2

    def __init__(
        self,
        ep_robot: Any = None,
        sensor_hub: Optional[Any] = None,
        mock_mode: bool = False,
        yaw_speed: float = 180.0,
        settle_delay_sec: float = 0.15,
    ):
        self.robot = ep_robot
        self.sensor_hub = sensor_hub
        self.mock_mode = mock_mode
        self.yaw_speed = yaw_speed
        self.settle_delay = settle_delay_sec
        self.current_yaw = 0.0
        self.current_pitch = 0.0
        self.last_scan_validity: Dict[str, bool] = {}

    def rotate_to(self, yaw_deg: float, pitch_deg: float = 0.0, speed: Optional[float] = None) -> bool:
        """Rotates gimbal to a specific yaw and pitch angle relative to robot chassis."""
        spd = speed or self.yaw_speed
        self.current_yaw = yaw_deg
        self.current_pitch = pitch_deg

        if self.mock_mode or self.robot is None:
            time.sleep(self.settle_delay)
            return True

        if hasattr(self.robot, "gimbal"):
            try:
                action = self.robot.gimbal.moveto(
                    pitch=int(pitch_deg),
                    yaw=int(yaw_deg),
                    pitch_speed=int(spd),
                    yaw_speed=int(spd),
                )
                action.wait_for_completed()
                if hasattr(action, "has_succeeded") and not action.has_succeeded:
                    print(f"[GimbalScanController] Gimbal action failed: {action.state}")
                    return False
                time.sleep(self.settle_delay)
                return True
            except Exception as e:
                print(f"[GimbalScanController] moveto error: {e}")
                return False
        return False

    def recenter(self, speed: Optional[float] = None) -> bool:
        """Recenters gimbal to 0° (facing front)."""
        spd = speed or self.yaw_speed
        self.current_yaw = 0.0
        self.current_pitch = 0.0

        if self.mock_mode or self.robot is None:
            time.sleep(self.settle_delay)
            return True

        if hasattr(self.robot, "gimbal"):
            try:
                action = self.robot.gimbal.recenter(pitch_speed=int(spd), yaw_speed=int(spd))
                action.wait_for_completed()
                time.sleep(self.settle_delay)
                return True
            except Exception as e:
                print(f"[GimbalScanController] recenter error: {e}")
                return False
        return False

    def scan_4_directions(
        self,
        mock_distance_provider: Optional[Callable[[str], float]] = None,
    ) -> Dict[str, Tuple[bool, float]]:
        """Rotates gimbal sequentially to Front, Right, Back, Left to measure wall distances.

        Returns a dictionary:
            {
                'front': (has_wall, distance_mm),
                'right': (has_wall, distance_mm),
                'back':  (has_wall, distance_mm),
                'left':  (has_wall, distance_mm),
            }
        """
        results: Dict[str, Tuple[bool, float]] = {}
        self.last_scan_validity = {}

        for dir_name in ("front", "right", "back", "left"):
            target_yaw = self.SCAN_ANGLES[dir_name]
            before = self.sensor_hub.get_latest_state() if self.sensor_hub else None
            before_sample_id = getattr(before, "tof_sample_id", 0)
            moved = self.rotate_to(yaw_deg=target_yaw)

            # Get a median of fresh ToF samples received only after the gimbal
            # reached this direction. Never label a stale previous-direction
            # reading as the wall result for this direction.
            if self.mock_mode and mock_distance_provider:
                dist_mm = mock_distance_provider(dir_name)
                is_valid = True
            elif self.sensor_hub and moved:
                readings = []
                last_sample_id = before_sample_id
                deadline = time.monotonic() + self.FRESH_TOF_TIMEOUT_SEC
                while time.monotonic() < deadline and len(readings) < self.FRESH_TOF_SAMPLES:
                    state = self.sensor_hub.get_latest_state()
                    sample_id = getattr(state, "tof_sample_id", 0)
                    if sample_id > last_sample_id:
                        last_sample_id = sample_id
                        if state.tof_valid and state.tof_filtered_mm is not None:
                            readings.append(float(state.tof_filtered_mm))
                    if len(readings) < self.FRESH_TOF_SAMPLES:
                        time.sleep(0.01)
                is_valid = len(readings) >= self.FRESH_TOF_SAMPLES
                dist_mm = statistics.median(readings) if is_valid else float("nan")
            else:
                dist_mm = float("nan")
                is_valid = False

            self.last_scan_validity[dir_name] = is_valid
            has_wall = is_valid and dist_mm < self.WALL_DETECT_THRESHOLD_MM
            results[dir_name] = (has_wall, dist_mm)

        # Always return gimbal to front center after scanning
        self.recenter()

        return results
