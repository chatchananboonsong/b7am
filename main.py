#!/usr/bin/env python3
"""RoboMaster EP Autonomous Grid Navigation System - Master CLI Entry Point.

Usage:
    # 1. Step 3: Run live robot navigation with PID
    python main.py run --conn-type ap --plan data/robot_map_plan.json

    # 2. Step 3: Run simulation / dry-run with PID
    python main.py simulate --plan data/robot_map_plan.json

    # 3. Step 3: Test single grid cell move with PID centering
    python main.py step-test --cells 1 --conn-type ap

    # 4. Step 2: Live monitor sensors (Thread 1)
    python main.py monitor --conn-type ap

    # 5. Step 2: Analyze post-run telemetry
    python main.py analyze telemetry_logs/run1
    # or python main.py analyze telemetry_logs/run1/run1.json

    # 6. Step 1: Fit sensor calibration curves
    python main.py calibrate fit data/calibration_measurements.csv

    # 7. Map Planner: Launch interactive Pygame Grid Map & A* Planner
    python main.py map
"""

import argparse
import os
import signal
import sys
import time
from pathlib import Path

# Add src to sys.path
_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


import json
from typing import List

from src.gimbal_controller import GimbalScanController
from src.robot_system import RobotSystem
from src.telemetry import TelemetryAnalyzer


def parse_custom_commands(cmd_input: str) -> List[str]:
    """Parses arbitrary command string or JSON list into robot controller commands."""
    if not cmd_input:
        return []
    s = cmd_input.strip()
    if s.startswith("["):
        try:
            return json.loads(s)
        except Exception:
            pass

    raw_items = [c.strip() for c in s.replace(";", ",").split(",") if c.strip()]
    parsed = []
    for item in raw_items:
        low = item.lower()
        if any(w in low for w in ("fwd", "forward", "move", "cell")):
            nums = [int(tok) for tok in item.split() if tok.isdigit()]
            cells = nums[0] if nums else 1
            parsed.append(f"Move Forward: {cells} cells")
        elif "left" in low:
            parsed.append("Turn Left (90 deg)")
        elif "right" in low:
            parsed.append("Turn Right (90 deg)")
        elif "around" in low or "180" in low:
            parsed.append("Turn Around (180 deg)")
        elif "recenter" in low:
            parsed.append("Gimbal Recenter")
        else:
            parsed.append(item)
    return parsed


def confirm_action(prompt: str, auto_yes: bool = False) -> bool:
    if auto_yes:
        return True
    try:
        answer = input(f"{prompt} [y/N]: ").strip().lower()
        if answer not in ("y", "yes"):
            print("[main] Operation cancelled.")
            return False
        return True
    except (EOFError, KeyboardInterrupt):
        return False


def run_navigation_plan(sys_runner: RobotSystem, args):
    """Executes navigation plan with Step 3 PID centering (without gripper actions)."""
    auto_yes = getattr(args, "yes", False)
    if not confirm_action("Start following the map now?", auto_yes=auto_yes):
        return None

    # Setup threads and command queue
    sys_runner.setup_threads(plan_file=args.plan)
    if hasattr(args, "commands") and args.commands and sys_runner.thread_2_controller:
        custom_cmds = parse_custom_commands(args.commands)
        sys_runner.thread_2_controller.set_commands(custom_cmds)
    elif getattr(args, "backward_cm", None) is not None and sys_runner.thread_2_controller:
        sys_runner.thread_2_controller.set_commands([f"Move Backward: {args.backward_cm} cm"])

    if sys_runner.thread_2_controller:
        sys_runner.thread_2_controller.base_speed = args.speed
        sys_runner.thread_2_controller.wall_pid.nominal_side_dist_mm = args.nominal_side

    # Start multi-threading navigation
    sys_runner.start()
    reached_goal = sys_runner.wait_for_completion(
        timeout=args.duration if args.duration > 0 else None
    )
    if reached_goal:
        print("[main] Navigation completed successfully!")
    else:
        print("[main] Navigation stopped or timed out.")

    return reached_goal


def cmd_simulate(args):
    print("=" * 65)
    print("🤖 STARTING STEP 3 MULTI-THREADING SIMULATION (PID GRID NAVIGATION)")
    print("=" * 65)
    sys_runner = RobotSystem(
        calibration_file=args.calibration,
        sensor_rate_hz=args.rate,
        mock_mode=True,
    )
    sys_runner.connect_robot()

    def sig_handler(sig, frame):
        print("\nInterrupt received. Stopping...")
        sys_runner.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    try:
        run_navigation_plan(sys_runner, args)
    finally:
        sys_runner.shutdown()


def cmd_run(args):
    print("=" * 65)
    print("🤖 STARTING STEP 3 LIVE MULTI-THREADING RUN (PID GRID CONTROL)")
    print("=" * 65)
    sys_runner = RobotSystem(
        calibration_file=args.calibration,
        sensor_rate_hz=args.rate,
        mock_mode=False,
        conn_type=args.conn_type,
    )
    connected = sys_runner.connect_robot()
    if not connected and not args.allow_mock_fallback:
        print("[Error] Failed to connect to robot hardware. Exiting.")
        return 1

    def sig_handler(sig, frame):
        print("\nInterrupt received. Emergency stop initiated...")
        sys_runner.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    try:
        run_navigation_plan(sys_runner, args)
    finally:
        sys_runner.shutdown()
    return 0


def cmd_step_test(args):
    print("=" * 65)
    print(f"🎯 TESTING {args.cells} GRID CELL(S) WITH STEP 3 PID CENTERING")
    print("=" * 65)
    sys_runner = RobotSystem(
        calibration_file=args.calibration,
        sensor_rate_hz=args.rate,
        mock_mode=args.mock,
        conn_type=args.conn_type,
    )
    sys_runner.connect_robot()
    sys_runner.setup_threads()

    if sys_runner.thread_2_controller:
        sys_runner.thread_2_controller.base_speed = args.speed
        sys_runner.thread_2_controller.wall_pid.nominal_side_dist_mm = args.nominal_side
        sys_runner.thread_2_controller.set_commands([f"Move Forward: {args.cells} cells"])

    def sig_handler(sig, frame):
        print("\nInterrupt received. Stopping...")
        sys_runner.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    sys_runner.start()
    sys_runner.wait_for_completion(timeout=30.0)
    sys_runner.shutdown()


def cmd_turn_test(args):
    direction_str = str(args.direction).lower()
    if direction_str == "left":
        deg = 90.0
        cmd_text = "Turn Left (90 deg)"
    elif direction_str == "right":
        deg = -90.0
        cmd_text = "Turn Right (90 deg)"
    elif direction_str == "around":
        deg = 180.0
        cmd_text = "Turn Around (180 deg)"
    else:
        try:
            deg = float(direction_str)
            cmd_text = f"Turn {deg:+.0f} deg"
        except ValueError:
            print(f"Unknown direction '{args.direction}'. Use 'left', 'right', 'around', or degrees like '90' or '-90'.")
            return 1

    print("=" * 65)
    print(f"🔄 TESTING TURN: {cmd_text} (z={deg:+.0f}°)")
    print("=" * 65)

    sys_runner = RobotSystem(
        calibration_file=args.calibration,
        sensor_rate_hz=args.rate,
        mock_mode=args.mock,
        conn_type=args.conn_type,
    )
    sys_runner.connect_robot()
    sys_runner.setup_threads()

    if sys_runner.thread_2_controller:
        sys_runner.thread_2_controller.set_commands([cmd_text])

    def sig_handler(sig, frame):
        print("\nInterrupt received. Stopping...")
        sys_runner.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    sys_runner.start()
    sys_runner.wait_for_completion(timeout=15.0)
    sys_runner.shutdown()
    return 0


def cmd_monitor(args):
    print("=" * 65)
    print("📡 LIVE SENSOR MONITOR (THREAD 1)")
    print("=" * 65)
    sys_runner = RobotSystem(
        calibration_file=args.calibration,
        sensor_rate_hz=args.rate,
        mock_mode=args.mock,
        conn_type=args.conn_type,
    )
    sys_runner.connect_robot()
    sys_runner.setup_threads()

    def sig_handler(sig, frame):
        sys_runner.shutdown(save_telemetry=False)
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    sys_runner.thread_1_sensor.start_collecting()

    try:
        print(f"{'Frame':<8} | {'Sharp L (mm)':<13} | {'Sharp R (mm)':<13} | {'ToF (mm)':<10} | {'Yaw (deg)':<10} | {'Walls (L/F/R)'}")
        print("-" * 75)
        while True:
            state = sys_runner.sensor_hub.wait_for_next_state(timeout=1.0)
            if state:
                sl = f"{state.sharp_left_mm:.1f}" if state.sharp_left_mm is not None else "N/A"
                sr = f"{state.sharp_right_mm:.1f}" if state.sharp_right_mm is not None else "N/A"
                tof = f"{state.tof_filtered_mm:.1f}" if state.tof_filtered_mm is not None else "N/A"
                walls = f"{'L' if state.wall_left_detected else '-'}/{'F' if state.wall_front_detected else '-'}/{'R' if state.wall_right_detected else '-'}"
                print(f"{state.frame_index:<8} | {sl:<13} | {sr:<13} | {tof:<10} | {state.yaw:<10.1f} | {walls}")
            time.sleep(0.1)
    finally:
        sys_runner.shutdown(save_telemetry=False)


def cmd_analyze(args):
    print(f"Analyzing telemetry log: {args.file}")
    TelemetryAnalyzer.analyze_file(args.file, save_plot=not args.no_plot)


def cmd_calibrate(args):
    try:
        from src.calibrate import collect_live, fit_command, init_csv
    except ImportError:
        from calibrate import collect_live, fit_command, init_csv
    if args.cal_cmd == "init-csv":
        init_csv(args.path)
    elif args.cal_cmd == "collect-live":
        collect_live(args.sensor, args.output, args.board_id, args.port, args.tof_index, args.samples, args.conn_type)
    elif args.cal_cmd == "fit":
        fit_command(args.input, args.output_dir)


def cmd_map(args):
    print("Launching Pygame Grid Map & A* Path Planner...")
    from src.map_planner import main as launch_map_gui
    launch_map_gui()


def cmd_explore(args):
    from src.explorer import main as launch_explorer
    cli_args = []
    if args.mock:
        cli_args.append("--mock")
    if args.conn_type:
        cli_args.extend(["--conn-type", args.conn_type])
    if args.strategy:
        cli_args.extend(["--strategy", args.strategy])
    if getattr(args, "grid", None):
        cli_args.extend(["--grid", str(args.grid)])
    if args.rows is not None:
        cli_args.extend(["--rows", str(args.rows)])
    if args.cols is not None:
        cli_args.extend(["--cols", str(args.cols)])
    if getattr(args, "cell_size", None) is not None:
        cli_args.extend(["--cell-size", str(args.cell_size)])
    if getattr(args, "cell_size_cm", None) is not None:
        cli_args.extend(["--cell-size-cm", str(args.cell_size_cm)])
    if args.start_col is not None:
        cli_args.extend(["--start-col", str(args.start_col)])
    if args.start_row is not None:
        cli_args.extend(["--start-row", str(args.start_row)])
    if args.output:
        cli_args.extend(["--output", args.output])
    if args.sim_map:
        cli_args.extend(["--sim-map", args.sim_map])
    if args.speed:
        cli_args.extend(["--speed", str(args.speed)])
    if args.step_delay is not None:
        cli_args.extend(["--step-delay", str(args.step_delay)])
    if args.max_steps:
        cli_args.extend(["--max-steps", str(args.max_steps)])
    return launch_explorer(cli_args)


def main():
    parser = argparse.ArgumentParser(description="RoboMaster EP Autonomous Navigation System (Steps 1-3)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. Run live
    run_p = subparsers.add_parser("run", help="Run live robot navigation with Step 3 PID")
    run_p.add_argument("--plan", default="data/robot_map_plan.json", help="Path to map plan json")
    run_p.add_argument("--commands", default="", help="Custom comma-separated command sequence (e.g. 'fwd 1, left, fwd 1')")
    run_p.add_argument("--calibration", default="calibration_output/calibration.json")
    run_p.add_argument("--conn-type", choices=("ap", "sta"), default="ap")
    run_p.add_argument("--rate", type=float, default=20.0, help="Sensor collection rate Hz")
    run_p.add_argument("--speed", type=float, default=0.25, help="Base cruising speed (m/s)")
    run_p.add_argument("--nominal-side", type=float, default=140.0, help="Nominal distance to single wall (mm)")
    run_p.add_argument("--duration", type=float, default=0.0, help="Max duration in seconds")
    run_p.add_argument("--backward-cm", type=float, help="Run one backward move for this distance in cm")
    run_p.add_argument("-y", "--yes", action="store_true", help="Auto-confirm all interactive prompts")
    run_p.add_argument("--allow-mock-fallback", action="store_true", help="Fallback to mock if robot unavailable")

    # 2. Simulate
    sim_p = subparsers.add_parser("simulate", help="Run simulation / dry-run with Step 3 PID")
    sim_p.add_argument("--plan", default="data/robot_map_plan.json", help="Path to map plan json")
    sim_p.add_argument("--commands", default="", help="Custom comma-separated command sequence (e.g. 'fwd 1, left, fwd 1')")
    sim_p.add_argument("--calibration", default="calibration_output/calibration.json")
    sim_p.add_argument("--rate", type=float, default=20.0, help="Sensor collection rate Hz")
    sim_p.add_argument("--speed", type=float, default=0.25, help="Base cruising speed (m/s)")
    sim_p.add_argument("--nominal-side", type=float, default=140.0, help="Nominal distance to single wall (mm)")
    sim_p.add_argument("--duration", type=float, default=0.0, help="Max duration in seconds")
    sim_p.add_argument("--backward-cm", type=float, help="Run one backward move for this distance in cm")
    sim_p.add_argument("-y", "--yes", action="store_true", help="Auto-confirm all interactive prompts")

    # 3. Step-test
    step_p = subparsers.add_parser("step-test", help="Test N grid cell move with PID centering")
    step_p.add_argument("--cells", type=int, default=1, help="Number of cells to move")
    step_p.add_argument("--conn-type", choices=("ap", "sta"), default="ap")
    step_p.add_argument("--calibration", default="calibration_output/calibration.json")
    step_p.add_argument("--rate", type=float, default=20.0)
    step_p.add_argument("--speed", type=float, default=0.20)
    step_p.add_argument("--nominal-side", type=float, default=140.0)
    step_p.add_argument("--mock", action="store_true")

    # 4. Turn-test
    turn_p = subparsers.add_parser("turn-test", help="Test in-place turn (+90 right, -90 left, 180 around)")
    turn_p.add_argument("--direction", choices=("left", "right", "around"), default="right", help="Turn direction: left (z=-90), right (z=+90), around (z=180)")
    turn_p.add_argument("--conn-type", choices=("ap", "sta"), default="ap")
    turn_p.add_argument("--calibration", default="calibration_output/calibration.json")
    turn_p.add_argument("--rate", type=float, default=20.0)
    turn_p.add_argument("--mock", action="store_true")

    # 5. Monitor
    mon_p = subparsers.add_parser("monitor", help="Live stream sensor telemetry (Thread 1)")
    mon_p.add_argument("--conn-type", choices=("ap", "sta"), default="ap")
    mon_p.add_argument("--calibration", default="calibration_output/calibration.json")
    mon_p.add_argument("--rate", type=float, default=20.0)
    mon_p.add_argument("--mock", action="store_true", help="Monitor mock data")

    # 6. Analyze
    ana_p = subparsers.add_parser("analyze", help="Analyze telemetry log and generate graphs")
    ana_p.add_argument("file", help="Path to telemetry JSON file or run folder (e.g. telemetry_logs/run1)")
    ana_p.add_argument("--no-plot", action="store_true", help="Skip plot generation")

    # 7. Calibrate
    cal_p = subparsers.add_parser("calibrate", help="Sensor calibration tools (Step 1)")
    cal_sub = cal_p.add_subparsers(dest="cal_cmd", required=True)
    c_init = cal_sub.add_parser("init-csv", help="create CSV template")
    c_init.add_argument("path", nargs="?", default="data/calibration_measurements.csv")
    c_live = cal_sub.add_parser("collect-live", help="collect live sensor values")
    c_live.add_argument("sensor", choices=("sharp_left", "sharp_right", "tof"))
    c_live.add_argument("--output", default="data/calibration_measurements.csv")
    c_live.add_argument("--board-id", type=int)
    c_live.add_argument("--port", type=int)
    c_live.add_argument("--tof-index", type=int, default=0)
    c_live.add_argument("--samples", type=int, default=10)
    c_live.add_argument("--conn-type", choices=("ap", "sta"), default="ap")
    c_fit = cal_sub.add_parser("fit", help="fit polynomial curves")
    c_fit.add_argument("input", default="data/calibration_measurements.csv")
    c_fit.add_argument("--output-dir", default="calibration_output")

    # 8. Map GUI
    subparsers.add_parser("map", help="Launch interactive Grid Map & A* Planner GUI")

    # 9. Autonomous Explore
    exp_p = subparsers.add_parser("explore", help="Autonomous Grid Maze Exploration with Wall Avoidance")
    exp_p.add_argument("--mock", action="store_true", help="Run in mock simulation mode")
    exp_p.add_argument("--conn-type", choices=("ap", "sta"), default="ap", help="Connection mode")
    exp_p.add_argument("--strategy", choices=("dfs", "wall-follow"), default="dfs", help="Exploration strategy")
    exp_p.add_argument("--grid", "--size", dest="grid", default=None, help="Grid size formatted as COLSxROWS (e.g. 4x4, 5x6, 6x6)")
    exp_p.add_argument("--rows", type=int, default=None, help="Grid rows count (default: 6)")
    exp_p.add_argument("--cols", type=int, default=None, help="Grid columns count (default: 5)")
    exp_p.add_argument("--cell-size", type=float, default=0.60, help="Grid cell size in meters (default: 0.60)")
    exp_p.add_argument("--cell-size-cm", type=float, default=None, help="Grid cell size in centimeters (e.g. 60 or 50)")
    exp_p.add_argument("--start-col", type=int, default=None, help="Initial column index (default: 0)")
    exp_p.add_argument("--start-row", type=int, default=None, help="Initial row index (default: rows - 1)")
    exp_p.add_argument("--output", default="data/explored_map.json", help="Output file path to save discovered map")
    exp_p.add_argument("--sim-map", default="data/robot_map_plan.json", help="Ground truth map for simulation mode")
    exp_p.add_argument("--speed", type=float, default=0.22, help="Base forward speed (m/s)")
    exp_p.add_argument("--step-delay", type=float, default=0.4, help="Pause delay between steps (s)")
    exp_p.add_argument("--max-steps", type=int, default=120, help="Maximum number of steps before stopping")

    args = parser.parse_args()

    if args.command == "simulate":
        return cmd_simulate(args)
    elif args.command == "run":
        return cmd_run(args)
    elif args.command == "step-test":
        return cmd_step_test(args)
    elif args.command == "turn-test":
        return cmd_turn_test(args)
    elif args.command == "monitor":
        return cmd_monitor(args)
    elif args.command == "analyze":
        return cmd_analyze(args)
    elif args.command == "calibrate":
        return cmd_calibrate(args)
    elif args.command == "map":
        return cmd_map(args)
    elif args.command == "explore":
        return cmd_explore(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
