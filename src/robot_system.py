#!/usr/bin/env python3
"""Master Robot System Coordinator for RoboMaster EP Multi-Threading.

Manages Robot connection, Thread 1 (Sensor Collection & Filtering),
Thread 2 (Robot Motion Controller), and Telemetry Logging.
"""

import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Optional

try:
    from . import calibrate
    from .robot_controller import RobotControllerThread
    from .sensor_pipeline import CalibrationManager, SensorCollectorThread, SensorHub
    from .telemetry import TelemetryAnalyzer, TelemetryRecorder
except (ImportError, ValueError):
    import calibrate
    from robot_controller import RobotControllerThread
    from sensor_pipeline import CalibrationManager, SensorCollectorThread, SensorHub
    from telemetry import TelemetryAnalyzer, TelemetryRecorder


class RobotSystem:
    """Master orchestrator for Step 2 Multi-Threading Architecture."""

    def __init__(
        self,
        calibration_file: str = "calibration_output/calibration.json",
        telemetry_dir: str = "telemetry_logs",
        sensor_rate_hz: float = 20.0,
        mock_mode: bool = False,
        conn_type: str = "ap",
    ):
        self.mock_mode = mock_mode
        self.conn_type = conn_type
        self.robot = None

        # Core subsystems
        self.calibration_mgr = CalibrationManager(calibration_file)
        self.telemetry = TelemetryRecorder(output_dir=telemetry_dir)
        self.sensor_hub = SensorHub(max_history=1000)

        # Multi-threading workers
        self.thread_1_sensor: Optional[SensorCollectorThread] = None
        self.thread_2_controller: Optional[RobotControllerThread] = None
        self.sensor_rate_hz = sensor_rate_hz

    def connect_robot(self) -> bool:
        """Initializes connection to RoboMaster EP hardware."""
        if self.mock_mode:
            print("[RobotSystem] Running in SIMULATION / MOCK mode (No physical hardware needed).")
            return True

        print(f"[RobotSystem] Connecting to RoboMaster EP via {self.conn_type.upper()}...")
        try:
            robot_mod = calibrate.load_robot_sdk()
            self.robot = robot_mod.Robot()
            self.robot.initialize(conn_type=self.conn_type)
            try:
                self.robot.set_robot_mode(mode=robot_mod.FREE)
            except Exception as e:
                print(f"[RobotSystem] Note on robot mode: {e}")
            print("[RobotSystem] Successfully connected to RoboMaster EP!")
            return True
        except Exception as exc:
            print(f"[RobotSystem] Connection failed: {exc}")
            self.robot = None
            return False

    def setup_threads(self, plan_file: Optional[str] = None):
        """Spawns Thread 1 and Thread 2 with thread-safe shared memory."""
        # Thread 1: Sensor Collection + Filtering
        self.thread_1_sensor = SensorCollectorThread(
            sensor_hub=self.sensor_hub,
            robot=self.robot,
            calibration_manager=self.calibration_mgr,
            telemetry_recorder=self.telemetry,
            update_rate_hz=self.sensor_rate_hz,
            mock_mode=self.mock_mode,
        )

        # Thread 2: Robot Motion Controller
        self.thread_2_controller = RobotControllerThread(
            sensor_hub=self.sensor_hub,
            robot=self.robot,
            mock_mode=self.mock_mode,
        )

        if self.mock_mode and self.thread_2_controller.mock_actuator:
            self.thread_2_controller.mock_actuator.collector = self.thread_1_sensor

        if plan_file:
            try:
                self.thread_2_controller.load_plan_from_file(plan_file)
            except Exception as e:
                print(f"[RobotSystem] Plan loading warning: {e}")

    def start(self):
        """Starts both Thread 1 and Thread 2."""
        if not self.thread_1_sensor or not self.thread_2_controller:
            self.setup_threads()

        print("[RobotSystem] Starting Thread 1 (Sensor Collection & Filtering)...")
        self.thread_1_sensor.start_collecting()

        # Let sensor buffers warm up
        time.sleep(0.3)

        print("[RobotSystem] Starting Thread 2 (Robot Motion Controller)...")
        self.thread_2_controller.start_running()

    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """Blocks until Thread 2 finishes all plan commands or timeout."""
        start_t = time.time()
        while True:
            if self.thread_2_controller and self.thread_2_controller.plan_completed:
                return True
            if timeout and (time.time() - start_t) > timeout:
                return False
            time.sleep(0.1)

    def shutdown(self, save_telemetry: bool = True, run_analysis: bool = True):
        """Gracefully shuts down both threads and exports run telemetry."""
        print("\n[RobotSystem] Shutting down multi-threading workers...")
        if self.thread_2_controller:
            self.thread_2_controller.stop_running()
        if self.thread_1_sensor:
            self.thread_1_sensor.stop_collecting()

        if self.robot is not None:
            try:
                self.robot.close()
                print("[RobotSystem] RoboMaster SDK connection closed.")
            except Exception:
                pass

        if save_telemetry:
            json_p, csv_p = self.telemetry.export()
            if run_analysis:
                TelemetryAnalyzer.analyze_file(str(json_p), save_plot=True)
