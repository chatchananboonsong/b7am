#!/usr/bin/env python3
"""Find a shortest grid route from a map JSON using A*.

This is a standalone command. It does not connect to the robot or run as part
of explore.py/main.py. Example:
    python shortest_path.py --goal 3,1
    python shortest_path.py --map data/robot_map_plan.json --goal 3,1 --output data/shortest_path.json
    python shortest_path.py --start 1,4 --goal 3,4 --heading East --run
Coordinates are (column,row), zero-based. The path minimizes grid steps.
"""

import argparse
import heapq
import json
from itertools import count
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


Position = Tuple[int, int]
DIRECTIONS = (
    (0, -1, "top", "bottom", "North"),
    (1, 0, "right", "left", "East"),
    (0, 1, "bottom", "top", "South"),
    (-1, 0, "left", "right", "West"),
)
TURN_COMMAND = {
    ("North", "East"): "Turn Right (90 deg)",
    ("East", "South"): "Turn Right (90 deg)",
    ("South", "West"): "Turn Right (90 deg)",
    ("West", "North"): "Turn Right (90 deg)",
    ("North", "West"): "Turn Left (90 deg)",
    ("West", "South"): "Turn Left (90 deg)",
    ("South", "East"): "Turn Left (90 deg)",
    ("East", "North"): "Turn Left (90 deg)",
    ("North", "South"): "Turn Around (180 deg)",
    ("South", "North"): "Turn Around (180 deg)",
    ("East", "West"): "Turn Around (180 deg)",
    ("West", "East"): "Turn Around (180 deg)",
}


def parse_position(value: str) -> Position:
    try:
        col, row = value.replace(" ", "").split(",", 1)
        return int(col), int(row)
    except (ValueError, AttributeError) as exc:
        raise argparse.ArgumentTypeError("Position must be COL,ROW, for example 3,1") from exc


def load_grid(path: Path):
    with path.open("r", encoding="utf-8") as source:
        data = json.load(source)
    info = data.get("grid_info", {})
    cols, rows = int(info.get("cols", 0)), int(info.get("rows", 0))
    if cols <= 0 or rows <= 0:
        raise ValueError("Map JSON must include positive grid_info.cols and grid_info.rows")

    walls: Dict[Position, Dict[str, Optional[bool]]] = {
        (col, row): {"top": None, "right": None, "bottom": None, "left": None}
        for row in range(rows) for col in range(cols)
    }
    for entry in data.get("walls", []):
        col, row = map(int, entry["pos"])
        if (col, row) not in walls:
            continue
        for key in ("top", "right", "bottom", "left"):
            if key in entry.get("walls", {}):
                walls[(col, row)][key] = bool(entry["walls"][key])

    has_scan_metadata = "scanned_cells" in data
    scanned: Set[Position] = {tuple(map(int, pos)) for pos in data.get("scanned_cells", [])}
    return data, cols, rows, walls, scanned, has_scan_metadata


def a_star(start: Position, goal: Position, cols: int, rows: int,
           walls: Dict[Position, Dict[str, Optional[bool]]],
           scanned: Set[Position], has_scan_metadata: bool) -> Optional[List[Position]]:
    def in_bounds(pos: Position) -> bool:
        return 0 <= pos[0] < cols and 0 <= pos[1] < rows

    if not in_bounds(start) or not in_bounds(goal):
        raise ValueError(f"Start/goal must be inside the {cols}x{rows} map")
    if start == goal:
        return [start]

    serial = count()
    frontier = [(abs(start[0] - goal[0]) + abs(start[1] - goal[1]), next(serial), start)]
    came_from: Dict[Position, Position] = {}
    cost_so_far = {start: 0}

    while frontier:
        _, _, current = heapq.heappop(frontier)
        if current == goal:
            route = [current]
            while current in came_from:
                current = came_from[current]
                route.append(current)
            return list(reversed(route))

        for dc, dr, wall_key, opposite_key, _ in DIRECTIONS:
            neighbor = current[0] + dc, current[1] + dr
            if not in_bounds(neighbor):
                continue
            # Both sides must explicitly say open; unknown walls are blocked.
            if walls[current][wall_key] is not False or walls[neighbor][opposite_key] is not False:
                continue
            # Explored maps contain unknown edges serialized as false. Only use
            # passages observed from at least one fully scanned adjacent cell.
            if has_scan_metadata and current not in scanned and neighbor not in scanned:
                continue

            new_cost = cost_so_far[current] + 1
            if new_cost >= cost_so_far.get(neighbor, float("inf")):
                continue
            came_from[neighbor] = current
            cost_so_far[neighbor] = new_cost
            heuristic = abs(neighbor[0] - goal[0]) + abs(neighbor[1] - goal[1])
            heapq.heappush(frontier, (new_cost + heuristic, next(serial), neighbor))
    return None


def shortest_route_to_found_target(
    start: Position,
    data: dict,
    cols: int,
    rows: int,
    walls: Dict[Position, Dict[str, Optional[bool]]],
    scanned: Set[Position],
    has_scan_metadata: bool,
):
    """Return the shortest reachable route to any target recorded in the map."""
    candidates = []
    for index, target in enumerate(data.get("targets", []), start=1):
        target_pos = target.get("target_pos")
        location_basis = "estimated target cell"
        if target_pos is None:
            # With bearing-only detections, route to the cell where it was seen.
            target_pos = target.get("observed_from")
            location_basis = "cell where target was observed (range unknown)"
        if not isinstance(target_pos, (list, tuple)) or len(target_pos) != 2:
            continue
        try:
            goal = int(target_pos[0]), int(target_pos[1])
        except (TypeError, ValueError):
            continue
        if not (0 <= goal[0] < cols and 0 <= goal[1] < rows):
            continue
        route = a_star(start, goal, cols, rows, walls, scanned, has_scan_metadata)
        if route is not None:
            candidates.append((len(route) - 1, index, goal, route, location_basis, target))

    if not candidates:
        return None
    return min(candidates, key=lambda item: (item[0], item[1]))


def make_commands(path: List[Position], initial_heading: str) -> List[str]:
    if len(path) < 2:
        return []
    heading = initial_heading
    pending_forward = 0
    commands: List[str] = []
    direction_by_delta = {(dc, dr): name for dc, dr, _, _, name in DIRECTIONS}
    for current, following in zip(path, path[1:]):
        target_heading = direction_by_delta[(following[0] - current[0], following[1] - current[1])]
        if target_heading != heading:
            if pending_forward:
                commands.append(f"Move Forward: {pending_forward} cells")
                pending_forward = 0
            commands.append(TURN_COMMAND[(heading, target_heading)])
            heading = target_heading
        pending_forward += 1
    if pending_forward:
        commands.append(f"Move Forward: {pending_forward} cells")
    return commands


def print_map(cols: int, rows: int, path: List[Position], start: Position, goal: Position):
    path_set = set(path)
    for row in range(rows):
        line = []
        for col in range(cols):
            pos = (col, row)
            symbol = "B" if pos == start == goal else "S" if pos == start else "T" if pos == goal else "*" if pos in path_set else "."
            line.append(symbol)
        print(" ".join(line))
    print("Legend: S=start, T=target, B=start and target, *=path, .=other cell")


def run_on_robot(path: List[Position], data: dict, map_path: Path, args) -> int:
    """Follow an A* route one grid at a time using the explorer's scan/PID logic."""
    if len(path) < 2:
        print("Start and goal are the same cell; no movement needed.")
        return 0
    if not args.yes:
        try:
            answer = input("Robot will execute the route above. Start moving? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = ""
        if answer not in ("y", "yes"):
            print("Movement cancelled.")
            return 1

    from src.robot_system import RobotSystem
    from src.explorer import AutonomousExplorer

    grid_info = data.get("grid_info", {})
    grid_size_m = float(grid_info.get("grid_size_m", 0.60))
    heading_index = {"North": 0, "East": 1, "South": 2, "West": 3}[args.heading]
    output_path = Path(args.run_map_output)
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parent / output_path

    system = RobotSystem(
        calibration_file=args.calibration,
        sensor_rate_hz=20.0,
        mock_mode=args.mock,
        conn_type=args.conn_type,
    )
    if not system.connect_robot():
        print("Could not connect to the robot; route was not started.")
        return 1

    try:
        system.setup_threads()
        controller = system.thread_2_controller
        controller.grid_size_m = grid_size_m
        controller.base_speed = args.speed
        # Reuse the same scan-before-step and closed-loop single-cell movement
        # implementation as explore.py. Keep the live run map separate from
        # the input map that A* used to plan this route.
        navigator = AutonomousExplorer(
            system,
            cols=int(grid_info["cols"]),
            rows=int(grid_info["rows"]),
            grid_size_m=grid_size_m,
            start_col=path[0][0],
            start_row=path[0][1],
            initial_direction=heading_index,
            sim_map_file=str(map_path),
            output_file=str(output_path),
            step_delay=0,
        )
        if not args.mock:
            print(f"Following the A* route on robot at {args.speed:.2f} m/s. Press Ctrl+C to stop.")
        else:
            print(f"Following the A* route in mock mode at {args.speed:.2f} m/s.")

        system.start()
        navigator._running = True
        for step_number, next_pos in enumerate(path[1:], start=1):
            dc, dr = next_pos[0] - navigator.current_col, next_pos[1] - navigator.current_row
            direction_by_delta = {(0, -1): 0, (1, 0): 1, (0, 1): 2, (-1, 0): 3}
            target_direction = direction_by_delta.get((dc, dr))
            if target_direction is None:
                print(f"Invalid non-adjacent route step to {next_pos}; stopping.")
                return 1
            print(f"\n[Route] Step {step_number}/{len(path) - 1}: {navigator.current_col,navigator.current_row} -> {next_pos}")
            navigator.turn_to_direction(target_direction)
            if not navigator.step_forward_one_cell():
                print(f"Route stopped safely at ({navigator.current_col},{navigator.current_row}).")
                return 1

        navigator._running = False
        print(f"Route completed at ({navigator.current_col},{navigator.current_row}).")
        print(f"Live scan map saved to: {output_path}")
        return 0
    except KeyboardInterrupt:
        print("\nRoute interrupted; stopping the robot.")
        return 130
    finally:
        system.shutdown()


def main(argv=None) -> int:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Find a shortest grid path with A* (no robot connection).")
    parser.add_argument("--map", dest="map_file", default="data/explored_map.json",
                        help="Input map JSON (default: data/explored_map.json)")
    parser.add_argument("--start", type=parse_position, help="Override start as COL,ROW")
    parser.add_argument("--goal", type=parse_position,
                        help="Optional explicit goal as COL,ROW; otherwise route to the nearest reachable found target")
    parser.add_argument("--heading", choices=("North", "East", "South", "West"), default=None,
                        help="Initial robot heading for generated commands (defaults to North for planning)")
    parser.add_argument("--output", help="Optional JSON path for the route and executable commands")
    parser.add_argument("--run", action="store_true",
                        help="Execute the route with the robot controller after confirmation")
    parser.add_argument("--yes", action="store_true",
                        help="Skip the movement confirmation prompt when using --run")
    parser.add_argument("--mock", action="store_true", help="Execute against the mock robot instead of hardware")
    parser.add_argument("--conn-type", choices=("ap", "sta"), default="ap", help="Robot connection mode")
    parser.add_argument("--speed", type=float, default=0.15, help="Robot speed in m/s when using --run")
    parser.add_argument("--calibration", default="calibration_output/calibration.json",
                        help="Sensor calibration file used by the robot controller")
    parser.add_argument("--run-map-output", default="data/shortest_path_run_map.json",
                        help="Separate map output updated during --run")
    args = parser.parse_args(argv)

    if args.run and args.start is None:
        parser.error("--run requires --start COL,ROW to match the robot's current map position")
    if args.run and not args.mock and args.heading is None:
        parser.error("--run requires --heading to match the robot's current orientation")

    map_path = Path(args.map_file)
    if not map_path.is_absolute():
        map_path = project_dir / map_path
    if not map_path.exists():
        parser.error(f"Map file not found: {map_path}")

    try:
        data, cols, rows, walls, scanned, has_scan_metadata = load_grid(map_path)
        start = args.start or tuple(map(int, data.get("start", [0, rows - 1])))
        if args.goal is not None:
            goal = args.goal
            path = a_star(start, goal, cols, rows, walls, scanned, has_scan_metadata)
            target_description = "explicit goal"
        else:
            target_route = shortest_route_to_found_target(
                start, data, cols, rows, walls, scanned, has_scan_metadata
            )
            if target_route is not None:
                _steps, target_index, goal, path, location_basis, target = target_route
                target_description = (
                    f"target {target_index} ({target.get('color', 'unknown')} "
                    f"{target.get('shape', '')}), {location_basis}"
                )
            elif data.get("goal") is not None:
                goal = tuple(map(int, data["goal"]))
                path = a_star(start, goal, cols, rows, walls, scanned, has_scan_metadata)
                target_description = "goal saved in map"
            elif data.get("targets"):
                parser.error("No recorded target has a reachable location in the known map.")
            else:
                parser.error("Map has no found targets; use --goal COL,ROW or explore and record a target first.")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))

    if path is None:
        print(f"No known open path from {start} to {goal}.")
        return 2

    heading = args.heading or data.get("robot_heading", "North")
    commands = make_commands(path, heading)
    print(f"Destination: {target_description} at {goal}")
    print(f"A* shortest path: {len(path) - 1} grid steps")
    print("Coordinates are (column,row), starting at zero.")
    print("Path:", " -> ".join(f"({col},{row})" for col, row in path))
    print_map(cols, rows, path, start, goal)
    print("Commands:")
    for command in commands:
        print(f"  {command}")

    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = project_dir / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result = {
            "grid_info": data["grid_info"],
            "start": list(start),
            "goal": list(goal),
            "path": [list(pos) for pos in path],
            "path_length_cells": len(path) - 1,
            "commands": commands,
        }
        with output_path.open("w", encoding="utf-8") as target:
            json.dump(result, target, indent=2, ensure_ascii=False)
        print(f"Saved route to: {output_path}")
    if args.run:
        if args.speed <= 0:
            parser.error("--speed must be greater than zero")
        return run_on_robot(path, data, map_path, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
