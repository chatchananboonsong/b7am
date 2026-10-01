#!/usr/bin/env python3
"""Step 1 calibration tools for RoboMaster EP sensors.

CSV input columns:
    sensor,raw_value,reference_mm[,sample_id]

Run ``python calibrate.py init-csv`` to create a template.  The fitting
command intentionally works offline, so measurements can be collected by
hand and verified before connecting to the robot.
"""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path


# Prefer this project's bundled SDK so runtime fixes are not shadowed by an
# older copy installed in the virtual environment's site-packages.
_BUNDLED_SDK_SRC = Path(__file__).resolve().parents[1] / "RoboMaster-SDK" / "src"
if _BUNDLED_SDK_SRC.is_dir() and str(_BUNDLED_SDK_SRC) not in sys.path:
    sys.path.insert(0, str(_BUNDLED_SDK_SRC))


SENSORS = ("sharp_left", "sharp_right", "tof", "gripper")
DEFAULT_DEGREES = {"sharp_left": 2, "sharp_right": 2, "tof": 1, "gripper": 1}


def read_measurements(path):
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for number, row in enumerate(csv.DictReader(stream), start=2):
            raw_text = (row.get("raw_value") or "").strip()
            reference_text = (row.get("reference_mm") or "").strip()
            if not raw_text and not reference_text:
                continue
            try:
                sensor = row["sensor"].strip().lower()
                raw = float(raw_text)
                reference = float(reference_text)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("invalid CSV row {}: {}".format(number, exc))
            if sensor not in SENSORS:
                raise ValueError("row {}: sensor must be one of {}".format(number, SENSORS))
            if not math.isfinite(raw) or not math.isfinite(reference) or reference <= 0:
                raise ValueError("row {}: values must be finite and reference_mm > 0".format(number))
            rows.append((sensor, raw, reference))
    if not rows:
        raise ValueError("CSV contains no measurements")
    return rows


def fit_polynomial(rows, degree):
    import numpy as np

    if len(rows) < degree + 1:
        raise ValueError("need at least {} measurements for degree {}".format(degree + 1, degree))
    raw = np.asarray([item[1] for item in rows], dtype=float)
    reference = np.asarray([item[2] for item in rows], dtype=float)
    coefficients = np.polyfit(raw, reference, degree)
    predicted = np.polyval(coefficients, raw)
    residual = reference - predicted
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((reference - np.mean(reference)) ** 2))
    return {
        "degree": degree,
        "coefficients": [float(value) for value in coefficients],
        "rmse_mm": float(np.sqrt(np.mean(residual ** 2))),
        "r2": 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot,
        "samples": len(rows),
        "raw_min": float(np.min(raw)),
        "raw_max": float(np.max(raw)),
        "reference_min_mm": float(np.min(reference)),
        "reference_max_mm": float(np.max(reference)),
    }


def plot_sensor(sensor, rows, fit, output):
    import matplotlib.pyplot as plt
    import numpy as np

    raw = np.asarray([item[1] for item in rows])
    reference = np.asarray([item[2] for item in rows])
    order = np.argsort(raw)
    x = np.linspace(raw.min(), raw.max(), 200)
    y = np.polyval(fit["coefficients"], x)
    plt.figure(figsize=(7, 4.5))
    plt.scatter(raw, reference, label="measurement")
    plt.plot(x, y, label="fit (degree {})".format(fit["degree"]))
    plt.xlabel("raw sensor value")
    plt.ylabel("reference distance (mm)")
    plt.title("{} calibration | RMSE {:.2f} mm | R² {:.4f}".format(sensor, fit["rmse_mm"], fit["r2"]))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def init_csv(path):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("sensor", "raw_value", "reference_mm", "sample_id"))
        writer.writerow(("sharp_left", "", "", "1"))
    print("created {}".format(target))


def append_measurement(path, sensor, raw_value, reference_mm, sample_id):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    new_file = not target.exists() or target.stat().st_size == 0
    with target.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        if new_file:
            writer.writerow(("sensor", "raw_value", "reference_mm", "sample_id"))
        writer.writerow((sensor, raw_value, reference_mm, sample_id))


def load_robot_sdk():
    """Load RoboMaster SDK while keeping camera media codec optional.

    The DJI SDK imports camera/media from robomaster.robot even when this
    calibration tool only needs sensor_adaptor and distance sensor modules.
    Some Linux installs do not ship libmedia_codec, so provide a small no-op
    codec module that lets Robot() construct camera/liveview objects. Camera
    streaming remains unavailable in that environment, but sensor calibration
    does not use it.
    """
    try:
        from robomaster import robot
        return robot
    except ModuleNotFoundError as exc:
        if exc.name != "libmedia_codec":
            raise RuntimeError("RoboMaster SDK is not installed correctly: {}".format(exc))

    import types

    class _NoCameraCodec(object):
        def __init__(self, *args, **kwargs):
            pass

        def decode(self, *args, **kwargs):
            return None

        def stop(self, *args, **kwargs):
            pass

        def display(self, *args, **kwargs):
            pass

    codec = types.ModuleType("libmedia_codec")
    codec.H264Decoder = _NoCameraCodec
    codec.OpusDecoder = _NoCameraCodec
    codec.AudioDecoder = _NoCameraCodec
    sys.modules["libmedia_codec"] = codec

    try:
        from robomaster import robot
        return robot
    except ImportError as exc:
        raise RuntimeError("RoboMaster SDK is not installed correctly: {}".format(exc))


def collect_live(sensor, output, board_id, port, tof_index, samples, conn_type):
    """Collect raw values from a connected EP and append them to CSV."""
    robot = load_robot_sdk()
    ep_robot = robot.Robot()
    latest_tof = [None]

    def tof_callback(distance):
        if tof_index >= len(distance):
            raise ValueError("tof index {} is not present in {}".format(tof_index, distance))
        latest_tof[0] = distance[tof_index]

    try:
        ep_robot.initialize(conn_type=conn_type)
        if sensor in ("sharp_left", "sharp_right"):
            sensor_id = 1 if sensor == "sharp_left" else 2
            sensor_port = port if port is not None else sensor_id
            for sample_id in range(1, samples + 1):
                reference = float(input("{} sample {} reference distance (mm): ".format(sensor, sample_id)))
                raw = ep_robot.sensor_adaptor.get_adc(id=board_id if board_id else sensor_id, port=sensor_port)
                if raw is None:
                    raise RuntimeError("sensor adapter returned no ADC value")
                append_measurement(output, sensor, raw, reference, sample_id)
                print("saved raw={} reference={}mm".format(raw, reference))
        elif sensor == "tof":
            ep_robot.sensor.sub_distance(freq=10, callback=tof_callback)
            time.sleep(0.5)
            for sample_id in range(1, samples + 1):
                reference = float(input("tof sample {} reference distance (mm): ".format(sample_id)))
                time.sleep(0.2)
                if latest_tof[0] is None:
                    raise RuntimeError("no ToF callback value received")
                append_measurement(output, sensor, latest_tof[0], reference, sample_id)
                print("saved raw={} reference={}mm".format(latest_tof[0], reference))
            ep_robot.sensor.unsub_distance()
        else:
            raise ValueError("live collection supports sharp_left, sharp_right, and tof; gripper position must be measured manually")
    finally:
        ep_robot.close()


def fit_command(input_path, output_dir):
    rows = read_measurements(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration = {"schema": 1, "source_csv": str(input_path), "sensors": {}}
    for sensor in SENSORS:
        sensor_rows = [row for row in rows if row[0] == sensor]
        if not sensor_rows:
            continue
        fit = fit_polynomial(sensor_rows, DEFAULT_DEGREES[sensor])
        calibration["sensors"][sensor] = fit
        plot_sensor(sensor, sensor_rows, fit, output_dir / (sensor + "_calibration.png"))
    if not calibration["sensors"]:
        raise ValueError("CSV has no supported sensor measurements")
    result = output_dir / "calibration.json"
    result.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
    print("wrote {}".format(result))
    for sensor, fit in calibration["sensors"].items():
        print("{}: {} samples, RMSE {:.2f} mm, R² {:.4f}".format(sensor, fit["samples"], fit["rmse_mm"], fit["r2"]))


def main():
    parser = argparse.ArgumentParser(description="RoboMaster EP Step 1 calibration")
    subparsers = parser.add_subparsers(dest="command", required=True)
    init_parser = subparsers.add_parser("init-csv", help="create a measurement CSV template")
    init_parser.add_argument("path", nargs="?", default="data/calibration_measurements.csv")
    live_parser = subparsers.add_parser("collect-live", help="collect Sharp/ToF values from a connected EP")
    live_parser.add_argument("sensor", choices=("sharp_left", "sharp_right", "tof"))
    live_parser.add_argument("--output", default="data/calibration_measurements.csv")
    live_parser.add_argument("--board-id", type=int, help="sensor-adapter board ID; defaults to 1/2 from REQ")
    live_parser.add_argument("--port", type=int, help="sensor-adapter port; defaults to 1/2 from REQ")
    live_parser.add_argument("--tof-index", type=int, default=0, help="ToF array index, 0-based")
    live_parser.add_argument("--samples", type=int, default=10)
    live_parser.add_argument("--conn-type", choices=("ap", "sta"), default="ap")
    fit_parser = subparsers.add_parser("fit", help="fit calibration curves and save plots")
    fit_parser.add_argument("input", help="measurement CSV")
    fit_parser.add_argument("--output-dir", default="calibration_output")
    args = parser.parse_args()
    try:
        if args.command == "init-csv":
            init_csv(args.path)
        elif args.command == "collect-live":
            collect_live(args.sensor, args.output, args.board_id, args.port, args.tof_index, args.samples, args.conn_type)
        else:
            fit_command(args.input, args.output_dir)
    except (OSError, RuntimeError, ValueError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
