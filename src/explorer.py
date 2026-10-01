#!/usr/bin/env python3
"""Autonomous Grid Maze Exploration and Mapping System with Wall Avoidance for RoboMaster EP.

Key Features:
- Systematically explores an unknown grid maze (default 6x5 cells, 60x60 cm each).
- Scans walls at each cell using ToF (front) and calibrated Sharp IR (left/right) sensors.
- Strict multi-layer wall collision avoidance ("ไม่ชนกำแพง"):
    1. Pre-move live ToF verification: Never steps into a blocked cell.
    2. Step 3 closed-loop PID lateral centering: Dynamically keeps robot in corridor center.
    3. Emergency deceleration & stop if front distance drops below safe threshold.
- Exploration algorithms:
    - Depth-First Search (DFS) with Backtracking (default, maps all reachable cells).
    - Wall-Following (Right-Hand / Left-Hand rule).
- Real-time ASCII terminal visualization showing robot position, heading, visited cells, and walls.
- Exports the fully discovered map to JSON compatible with data/robot_map_plan.json and main.py.
- Built-in Mock Simulator for full dry-run testing without physical robot hardware.
"""

import argparse
import copy
import json
import math
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from .gimbal_controller import GimbalScanController
    from .pid_controller import WallCenteringPID
    from .robot_controller import RobotControllerThread
    from .robot_system import RobotSystem
    from .sensor_pipeline import RobotSensorSnapshot, SensorHub
except (ImportError, ValueError):
    from gimbal_controller import GimbalScanController
    from pid_controller import WallCenteringPID
    from robot_controller import RobotControllerThread
    from robot_system import RobotSystem
    from sensor_pipeline import RobotSensorSnapshot, SensorHub


if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Cardinal Directions & Geometry Definitions
# ---------------------------------------------------------------------------
# Heading index: 0 = North (dy=-1, dx=0), 1 = East (dy=0, dx=+1),
#                2 = South (dy=+1, dx=0), 3 = West (dy=0, dx=-1)
DIRECTION_NAMES = ["North (Up)", "East (Right)", "South (Down)", "West (Left)"]
DIRECTION_ARROWS = ["^", ">", "v", "<"]
DIRECTION_OFFSETS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # (d_col, d_row)
WORLD_WALL_KEYS = ["top", "right", "bottom", "left"]
OPPOSITE_WALL_KEYS = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}


def parse_grid_dimensions(
    grid_str: Optional[str] = None,
    cols: Optional[int] = None,
    rows: Optional[int] = None,
    default_cols: int = 6,
    default_rows: int = 6,
) -> Tuple[int, int]:
    """Parses grid dimensions from a string like '5x6', '6x5', '4x4', '5,6', '5 6' or individual cols/rows integers."""
    if grid_str:
        s = grid_str.strip().lower()
        for sep in ("x", "*", ",", " "):
            if sep in s:
                parts = [p.strip() for p in s.split(sep) if p.strip()]
                if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                    return int(parts[0]), int(parts[1])
        if s.isdigit():
            val = int(s)
            return val, val
    final_cols = cols if cols is not None and cols > 0 else default_cols
    final_rows = rows if rows is not None and rows > 0 else default_rows
    return final_cols, final_rows


def parse_grid_position(value: str) -> Tuple[int, int]:
    """Parse a zero-based grid coordinate written as COL,ROW."""
    try:
        parts = [part.strip() for part in value.split(",")]
        if len(parts) != 2:
            raise ValueError
        return int(parts[0]), int(parts[1])
    except (AttributeError, TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "Start position must be COL,ROW (zero-based), for example 2,3"
        ) from exc


def format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


# ---------------------------------------------------------------------------
# Explored Grid Map Representation
# ---------------------------------------------------------------------------
class ExploredGridMap:
    """Maintains the internal representation of the grid maze and walls discovered."""

    def __init__(self, cols: int = 6, rows: int = 6, grid_size_m: float = 0.60):
        self.cols = cols
        self.rows = rows
        self.grid_size_m = grid_size_m

        # 2D grid: grid[row][col] = dict of cell info
        self.cells: List[List[Dict[str, Any]]] = []
        for r in range(rows):
            row_cells = []
            for c in range(cols):
                cell_dict = {
                    "row": r,
                    "col": c,
                    "visited": False,
                    "scanned": False,
                    "targets_scanned": False,
                    # walls: True = wall exists, False = passage open, None = unknown
                    "walls": {"top": None, "bottom": None, "left": None, "right": None},
                }
                # Outer boundaries are known solid walls
                if r == 0:
                    cell_dict["walls"]["top"] = True
                if r == rows - 1:
                    cell_dict["walls"]["bottom"] = True
                if c == 0:
                    cell_dict["walls"]["left"] = True
                if c == cols - 1:
                    cell_dict["walls"]["right"] = True
                row_cells.append(cell_dict)
            self.cells.append(row_cells)

        self.visited_count = 0
        self.targets: List[Dict[str, Any]] = []
        self.exploration_time_seconds = 0.0

    def is_valid_coord(self, col: int, row: int) -> bool:
        return 0 <= col < self.cols and 0 <= row < self.rows

    def expand_edge(self, direction: int, current_col: int, current_row: int):
        """Add one cell beyond an open map edge and connect it to the current cell."""
        if direction not in range(4):
            return

        if direction in (1, 3):
            insert_at = 0 if direction == 3 else self.cols
            for row_idx, row_cells in enumerate(self.cells):
                new_cell = {
                    "row": row_idx,
                    "col": insert_at,
                    "visited": False,
                    "scanned": False,
                    "targets_scanned": False,
                    "walls": {"top": None, "bottom": None, "left": None, "right": None},
                }
                if row_idx == 0:
                    new_cell["walls"]["top"] = True
                if row_idx == self.rows - 1:
                    new_cell["walls"]["bottom"] = True
                if direction == 3:
                    new_cell["walls"]["left"] = True
                    new_cell["walls"]["right"] = None
                    row_cells.insert(0, new_cell)
                    row_cells[1]["walls"]["left"] = None
                else:
                    new_cell["walls"]["right"] = True
                    new_cell["walls"]["left"] = None
                    row_cells.append(new_cell)
            self.cols += 1
            if direction == 3:
                current_col += 1
                self.cells[current_row][0]["walls"]["right"] = False
                self.cells[current_row][1]["walls"]["left"] = False
                for row_cells in self.cells:
                    for col_idx, cell in enumerate(row_cells):
                        cell["col"] = col_idx
            else:
                self.cells[current_row][-2]["walls"]["right"] = False
                self.cells[current_row][-1]["walls"]["left"] = False
        else:
            insert_at = 0 if direction == 0 else self.rows
            new_row = []
            for col_idx in range(self.cols):
                new_cell = {
                    "row": insert_at,
                    "col": col_idx,
                    "visited": False,
                    "scanned": False,
                    "targets_scanned": False,
                    "walls": {"top": None, "bottom": None, "left": None, "right": None},
                }
                if col_idx == 0:
                    new_cell["walls"]["left"] = True
                if col_idx == self.cols - 1:
                    new_cell["walls"]["right"] = True
                if direction == 0:
                    new_cell["walls"]["top"] = True
                    new_cell["walls"]["bottom"] = None
                else:
                    new_cell["walls"]["bottom"] = True
                    new_cell["walls"]["top"] = None
                new_row.append(new_cell)
            if direction == 0:
                self.cells.insert(0, new_row)
                for col_idx in range(self.cols):
                    self.cells[1][col_idx]["walls"]["top"] = None
                current_row += 1
                self.cells[0][current_col]["walls"]["bottom"] = False
                self.cells[1][current_col]["walls"]["top"] = False
            else:
                self.cells.append(new_row)
                for col_idx in range(self.cols):
                    self.cells[-2][col_idx]["walls"]["bottom"] = None
                self.cells[-1][current_col]["walls"]["top"] = False
                self.cells[-2][current_col]["walls"]["bottom"] = False
            self.rows += 1
            for row_idx, row_cells in enumerate(self.cells):
                for col_idx, cell in enumerate(row_cells):
                    cell["row"] = row_idx
                    cell["col"] = col_idx

        return current_col, current_row

    def mark_visited(self, col: int, row: int):
        if self.is_valid_coord(col, row):
            if not self.cells[row][col]["visited"]:
                self.cells[row][col]["visited"] = True
                self.visited_count += 1

    def is_visited(self, col: int, row: int) -> bool:
        if self.is_valid_coord(col, row):
            return bool(self.cells[row][col]["visited"])
        return True

    def set_wall(self, col: int, row: int, wall_key: str, has_wall: bool, force: bool = False):
        """Sets wall status for cell (col, row) and mirrors it to the adjacent cell."""
        if not self.is_valid_coord(col, row):
            return

        # Keep the first complete scan as the cached observation. A confirmed
        # collision/safety reading may override it with force=True.
        if self.cells[row][col]["scanned"] and not force:
            return

        self.cells[row][col]["walls"][wall_key] = bool(has_wall)

        # Mirror wall to neighbor
        dir_idx = WORLD_WALL_KEYS.index(wall_key)
        dc, dr = DIRECTION_OFFSETS[dir_idx]
        nc, nr = col + dc, row + dr
        if self.is_valid_coord(nc, nr):
            opposite = OPPOSITE_WALL_KEYS[wall_key]
            self.cells[nr][nc]["walls"][opposite] = bool(has_wall)

    def has_wall(self, col: int, row: int, wall_key: str) -> Optional[bool]:
        """Returns True if wall exists, False if open, None if unknown."""
        if not self.is_valid_coord(col, row):
            return True
        return self.cells[row][col]["walls"][wall_key]

    def render_ascii(self, cur_col: int, cur_row: int, cur_dir: int) -> str:
        """Returns an ASCII art string representing the current state of the map."""
        lines = []
        header = f"=== MAZE MAP ({self.cols}x{self.rows}) | Visited: {self.visited_count}/{self.cols * self.rows} ==="
        lines.append(header)

        # Top border
        top_str = "+"
        for c in range(self.cols):
            w = self.cells[0][c]["walls"]["top"]
            top_str += "---+" if w is True else ("   +" if w is False else " ? +")
        lines.append(top_str)

        for r in range(self.rows):
            # Cells row
            row_str = ""
            for c in range(self.cols):
                # Left wall
                lw = self.cells[r][c]["walls"]["left"]
                left_char = "|" if lw is True else (" " if lw is False else ":")
                row_str += left_char

                # Cell content
                has_target = any(
                    (t.get("target_pos") or t.get("observed_from")) == [c, r]
                    for t in self.targets
                )
                if c == cur_col and r == cur_row:
                    row_str += f" {DIRECTION_ARROWS[cur_dir]} "
                elif has_target:
                    row_str += " T "
                elif self.cells[r][c]["visited"]:
                    row_str += " · "
                else:
                    row_str += " ? "

            # Rightmost border
            rw = self.cells[r][self.cols - 1]["walls"]["right"]
            row_str += "|" if rw is True else (" " if rw is False else ":")
            lines.append(row_str)

            # Bottom wall row
            bot_str = "+"
            for c in range(self.cols):
                bw = self.cells[r][c]["walls"]["bottom"]
                bot_str += "---+" if bw is True else ("   +" if bw is False else " ? +")
            lines.append(bot_str)

        lines.append(f"Legend: T=Target estimate/observation cell | Robot arrow marks current position")
        for index, target in enumerate(self.targets, start=1):
            pos = target.get("target_pos")
            direction = target.get("direction", "unknown")
            if pos is not None:
                location = f"estimated cell ({pos[0]}, {pos[1]})"
            else:
                source = target.get("observed_from", [cur_col, cur_row])
                location = f"from ({source[0]}, {source[1]}) toward {direction} (range unknown)"
            lines.append(f"Target {index}: {target.get('color', 'unknown')} {target.get('shape', '')} | {location}")
        return "\n".join(lines)

    def to_json_dict(self, start_pos: Tuple[int, int], history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Converts explored map into a dictionary format compatible with robot_map_plan.json."""
        wall_list = []
        for r in range(self.rows):
            for c in range(self.cols):
                cell_walls = {}
                for k in ("top", "bottom", "left", "right"):
                    w_val = self.cells[r][c]["walls"][k]
                    # In exported map, unknown walls default to false unless outer border
                    cell_walls[k] = bool(w_val) if w_val is not None else False
                wall_list.append({"pos": [c, r], "walls": cell_walls})

        visited_cells = [[c, r] for r in range(self.rows) for c in range(self.cols) if self.cells[r][c]["visited"]]

        return {
            "grid_info": {
                "rows": self.rows,
                "cols": self.cols,
                "grid_size_px": 100,
                "grid_size_m": self.grid_size_m,
            },
            "start": [start_pos[0], start_pos[1]],
            "visited_cells": visited_cells,
            "scanned_cells": [
                [c, r]
                for r in range(self.rows)
                for c in range(self.cols)
                if self.cells[r][c]["scanned"]
            ],
            "walls": wall_list,
            "targets": copy.deepcopy(self.targets),
            "exploration_time_seconds": self.exploration_time_seconds,
            "exploration_history": history or [],
        }

    def save_json(
        self,
        file_path: str,
        start_pos: Tuple[int, int],
        history: Optional[List[Dict[str, Any]]] = None,
        announce: bool = True,
    ):
        """Saves explored map to a JSON file."""
        data = self.to_json_dict(start_pos, history)
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        if announce:
            print(f"[ExploredGridMap] Successfully saved explored map to: {p.resolve()}")

    def save_realtime(self, file_path: str, start_pos: Tuple[int, int], history: List[Dict[str, Any]]):
        """Persist the current map after each scan or move for live progress recovery."""
        self.save_json(file_path, start_pos, history, announce=False)


class LiveExplorerDashboard:
    """OpenCV dashboard with the live camera beside the currently explored map."""

    def __init__(self, explorer):
        self.explorer = explorer
        self.window_name = "RoboMaster Explorer | Map + Live Camera"
        self.status = "CONNECTING"

    def render(self, camera_frame, status_text=""):
        import cv2
        import numpy as np

        if status_text:
            self.status = status_text
        explorer = self.explorer
        map_width, view_height = 520, 820
        camera_width = 920
        canvas = np.full((view_height, map_width + camera_width, 3), (24, 27, 34), dtype=np.uint8)

        cv2.putText(canvas, "EXPLORED MAP", (18, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (90, 230, 180), 2)
        cv2.putText(canvas, f"Visited {explorer.map.visited_count} | Grid {explorer.cols}x{explorer.rows}",
                    (18, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (230, 230, 230), 1)
        cv2.putText(canvas, f"Robot ({explorer.current_col}, {explorer.current_row})  {DIRECTION_NAMES[explorer.current_direction]}",
                    (18, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 210, 90), 1)
        cv2.putText(canvas, f"Elapsed: {format_duration(explorer.elapsed_exploration_seconds())}",
                    (18, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (190, 220, 255), 1)

        map_top, map_bottom = 120, view_height - 35
        cell_size = max(12, min((map_width - 36) // max(1, explorer.cols),
                                (map_bottom - map_top) // max(1, explorer.rows)))
        map_draw_width = cell_size * explorer.cols
        map_draw_height = cell_size * explorer.rows
        origin_x = (map_width - map_draw_width) // 2
        origin_y = map_top + (map_bottom - map_top - map_draw_height) // 2

        for row in range(explorer.rows):
            for col in range(explorer.cols):
                cell = explorer.map.cells[row][col]
                x1, y1 = origin_x + col * cell_size, origin_y + row * cell_size
                x2, y2 = x1 + cell_size, y1 + cell_size
                if col == explorer.current_col and row == explorer.current_row:
                    fill = (60, 165, 235)
                elif cell["scanned"]:
                    fill = (75, 105, 75)
                elif cell["visited"]:
                    fill = (65, 80, 65)
                else:
                    fill = (42, 47, 57)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), fill, cv2.FILLED)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (95, 102, 112), 1)
                walls = cell["walls"]
                wall_color = (245, 245, 245)
                thickness = max(2, cell_size // 12)
                if walls["top"] is True:
                    cv2.line(canvas, (x1, y1), (x2, y1), wall_color, thickness)
                if walls["right"] is True:
                    cv2.line(canvas, (x2, y1), (x2, y2), wall_color, thickness)
                if walls["bottom"] is True:
                    cv2.line(canvas, (x1, y2), (x2, y2), wall_color, thickness)
                if walls["left"] is True:
                    cv2.line(canvas, (x1, y1), (x1, y2), wall_color, thickness)
                if col == explorer.current_col and row == explorer.current_row:
                    arrow = DIRECTION_ARROWS[explorer.current_direction]
                    cv2.putText(canvas, arrow, (x1 + cell_size // 3, y1 + (cell_size * 2) // 3),
                                cv2.FONT_HERSHEY_SIMPLEX, max(0.45, cell_size / 48), (255, 255, 255), 2)
                elif cell["visited"] and cell_size >= 28:
                    cv2.putText(canvas, "V", (x1 + 5, y1 + cell_size - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (225, 235, 225), 1)
                target_here = any(
                    (t.get("target_pos") or t.get("observed_from")) == [col, row]
                    for t in explorer.map.targets
                )
                if target_here:
                    cv2.putText(canvas, "T", (x2 - max(12, cell_size // 3), y1 + max(14, cell_size // 3)),
                                cv2.FONT_HERSHEY_SIMPLEX, max(0.4, cell_size / 52), (0, 220, 255), 2)

        camera_x = map_width
        cv2.putText(canvas, "LIVE CAMERA / ROBOT STATUS", (camera_x + 18, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (90, 210, 255), 2)
        status = self.status.encode("ascii", "replace").decode("ascii")[:100]
        cv2.putText(canvas, status or "RUNNING", (camera_x + 18, 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 230, 160), 1)
        if camera_frame is not None and getattr(camera_frame, "size", 0):
            frame_h, frame_w = camera_frame.shape[:2]
            scale = min((camera_width - 28) / frame_w, (view_height - 105) / frame_h)
            resized = cv2.resize(camera_frame, (max(1, int(frame_w * scale)), max(1, int(frame_h * scale))))
            cam_y = 82 + max(0, (view_height - 96 - resized.shape[0]) // 2)
            cam_x = camera_x + (camera_width - resized.shape[1]) // 2
            canvas[cam_y:cam_y + resized.shape[0], cam_x:cam_x + resized.shape[1]] = resized
        cv2.imshow(self.window_name, canvas)
        cv2.waitKey(1)


# ---------------------------------------------------------------------------
# Ground Truth Simulator (for Mock Exploration Testing)
# ---------------------------------------------------------------------------
class SimulatedMaze:
    """Provides ground-truth wall queries to simulate realistic sensor responses in Mock Mode."""

    def __init__(self, cols: int = 6, rows: int = 6, plan_file: Optional[str] = None):
        self.cols = cols
        self.rows = rows
        self.walls: Dict[Tuple[int, int], Dict[str, bool]] = {}

        loaded_ok = False
        if plan_file and Path(plan_file).exists():
            loaded_ok = self._load_from_json(plan_file)
        if not loaded_ok:
            self._build_default_maze()

    def _load_from_json(self, plan_file: str) -> bool:
        try:
            with open(plan_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            grid_info = data.get("grid_info", {})
            file_rows = grid_info.get("rows", 6)
            file_cols = grid_info.get("cols", 5)

            if file_cols != self.cols or file_rows != self.rows:
                # User requested a different grid size than the plan file
                print(f"[SimulatedMaze] Ground truth {plan_file} ({file_cols}x{file_rows}) does not match requested ({self.cols}x{self.rows}). Building simulated maze for {self.cols}x{self.rows}.")
                return False

            for item in data.get("walls", []):
                pos = tuple(item["pos"])
                self.walls[pos] = dict(item["walls"])
            print(f"[SimulatedMaze] Loaded {len(self.walls)} cells from ground-truth {plan_file} ({self.cols}x{self.rows})")
            return True
        except Exception as e:
            print(f"[SimulatedMaze] Failed to load {plan_file}: {e}. Falling back to default maze.")
            return False

    def _build_default_maze(self):
        """Generates a test maze with corridors and dead-ends for any arbitrary (cols, rows)."""
        for r in range(self.rows):
            for c in range(self.cols):
                self.walls[(c, r)] = {"top": False, "bottom": False, "left": False, "right": False}
                if r == 0:
                    self.walls[(c, r)]["top"] = True
                if r == self.rows - 1:
                    self.walls[(c, r)]["bottom"] = True
                if c == 0:
                    self.walls[(c, r)]["left"] = True
                if c == self.cols - 1:
                    self.walls[(c, r)]["right"] = True

        if self.cols == 5 and self.rows == 6:
            internal_walls = [
                (0, 5, "top"), (1, 5, "top"), (3, 5, "top"),
                (1, 4, "right"), (2, 4, "top"),
                (0, 3, "right"), (1, 3, "top"), (3, 3, "right"),
                (2, 2, "left"), (3, 2, "top"), (4, 2, "top"),
                (1, 1, "top"), (1, 1, "right"), (3, 1, "top"),
                (2, 0, "bottom"),
            ]
        else:
            # Procedural internal walls for any NxM grid
            internal_walls = []
            for r in range(self.rows):
                for c in range(self.cols):
                    if (r % 2 == 1 and c % 2 == 0 and r < self.rows - 1) or (r % 2 == 0 and c % 2 == 1 and c < self.cols - 1):
                        if (c, r) != (0, self.rows - 1) and (c, r) != (self.cols - 1, 0):
                            w_dir = "bottom" if (r % 2 == 1) else "right"
                            internal_walls.append((c, r, w_dir))

        for c, r, w_key in internal_walls:
            if 0 <= c < self.cols and 0 <= r < self.rows:
                self.walls[(c, r)][w_key] = True
                dir_idx = WORLD_WALL_KEYS.index(w_key)
                dc, dr = DIRECTION_OFFSETS[dir_idx]
                nc, nr = c + dc, r + dr
                if 0 <= nc < self.cols and 0 <= nr < self.rows:
                    self.walls[(nc, nr)][OPPOSITE_WALL_KEYS[w_key]] = True
        print(f"[SimulatedMaze] Built procedural test maze for {self.cols}x{self.rows} grid.")

    def query_walls_relative(self, col: int, row: int, heading_idx: int) -> Tuple[bool, bool, bool]:
        """Returns (has_front, has_left, has_right) relative to robot heading."""
        cell_walls = self.walls.get((col, row), {"top": True, "bottom": True, "left": True, "right": True})
        dir_front = heading_idx
        dir_right = (heading_idx + 1) % 4
        dir_left = (heading_idx + 3) % 4

        has_front = bool(cell_walls.get(WORLD_WALL_KEYS[dir_front], True))
        has_right = bool(cell_walls.get(WORLD_WALL_KEYS[dir_right], True))
        has_left = bool(cell_walls.get(WORLD_WALL_KEYS[dir_left], True))
        return has_front, has_left, has_right

    def query_wall_by_direction(self, col: int, row: int, dir_idx: int) -> bool:
        """Returns True if wall exists in world direction (0:N, 1:E, 2:S, 3:W)."""
        cell_walls = self.walls.get((col, row), {"top": True, "bottom": True, "left": True, "right": True})
        return bool(cell_walls.get(WORLD_WALL_KEYS[dir_idx % 4], True))


# ---------------------------------------------------------------------------
# Autonomous Explorer Controller ("ไม่ชนกำแพง")
# ---------------------------------------------------------------------------
class AutonomousExplorer:
    """Orchestrates autonomous exploration of the maze with strict collision avoidance."""

    def __init__(
        self,
        robot_system: RobotSystem,
        cols: int = 6,
        rows: int = 6,
        grid_size_m: float = 0.60,
        start_col: int = 0,
        start_row: int = 5,
        initial_direction: int = 0,  # 0 = North
        strategy: str = "dfs",       # "dfs" or "wall-follow"
        sim_map_file: Optional[str] = "data/robot_map_plan.json",
        output_file: str = "data/explored_map.json",
        step_delay: float = 0.5,
        target_scan_config: Optional[Dict[str, Any]] = None,
    ):
        self.sys_runner = robot_system
        self.mock_mode = robot_system.mock_mode
        self.cols = cols
        self.rows = rows
        self.grid_size_m = grid_size_m
        self.current_col = start_col
        self.current_row = start_row
        self.start_pos = (start_col, start_row)
        self.current_direction = initial_direction
        self.strategy = strategy.lower()
        self.output_file = output_file
        self.step_delay = step_delay
        self.target_scan_config = target_scan_config
        self.target_scan_module = None
        self.target_memory = None
        if target_scan_config and not self.mock_mode:
            import gimbal_scan_controller as target_scan_module
            self.target_scan_module = target_scan_module
            self.target_memory = target_scan_module.TargetMemory()
        self.gimbal_scanner = GimbalScanController(
            ep_robot=robot_system.robot,
            sensor_hub=robot_system.sensor_hub,
            mock_mode=self.mock_mode,
            settle_delay_sec=0.10,
        )

        self.map = ExploredGridMap(cols=cols, rows=rows, grid_size_m=grid_size_m)
        self.map.mark_visited(self.current_col, self.current_row)

        if self.controller:
            self.controller.grid_size_m = grid_size_m

        self.history: List[Dict[str, Any]] = []
        self._running = False
        self._finished = False
        self._exploration_started_at: Optional[float] = None
        self.exploration_duration_seconds = 0.0
        self.step_count = 0
        self.completed_grid_steps = 0
        # Simulator ground-truth for mock testing
        self.sim_maze: Optional[SimulatedMaze] = None
        if self.mock_mode:
            self.sim_maze = SimulatedMaze(cols=cols, rows=rows, plan_file=sim_map_file)
        self._backtrack_stack = None
        self._visited_states = None
        self.dashboard = LiveExplorerDashboard(self) if self.target_scan_module else None
        if self.dashboard:
            self.target_scan_module.set_live_dashboard(self.dashboard.render)
            if self.controller:
                self.controller.ui_tick = self._tick_dashboard
        self._save_progress()

    def _save_progress(self):
        """Write a live snapshot so visited cells and discovered walls are retained."""
        self.map.exploration_time_seconds = self.elapsed_exploration_seconds()
        self.map.save_realtime(self.output_file, self.start_pos, self.history)

    def elapsed_exploration_seconds(self) -> float:
        if self._exploration_started_at is None or self._finished:
            return self.exploration_duration_seconds
        return max(0.0, time.monotonic() - self._exploration_started_at)

    def _record_detected_target(self, direction_id, target, range_m=None, target_pos=None):
        """Store a target marker and its viewing direction in the explored map."""
        direction_offset = {"front": 0, "right": 1, "back": 2, "left": 3}.get(direction_id)
        if direction_offset is None:
            return
        world_direction = (self.current_direction + direction_offset) % 4
        direction_name = ("North", "East", "South", "West")[world_direction]
        estimated_pos = None
        if target_pos is not None:
            col, row = int(target_pos[0]), int(target_pos[1])
            if self.map.is_valid_coord(col, row):
                estimated_pos = [col, row]
        image_cell = target.get("grid_cell")
        record = {
            "observed_from": [self.current_col, self.current_row],
            "direction": direction_name,
            "target_pos": estimated_pos,
            "image_cell": image_cell,
            "range_m": float(range_m) if range_m is not None else None,
            "color": target.get("color"),
            "shape": target.get("shape"),
            "location_basis": "range_estimate" if estimated_pos is not None else "bearing_only",
        }
        key = (record["observed_from"], direction_name, image_cell,
               record["color"], record["shape"])
        existing = next((item for item in self.map.targets
                         if (item.get("observed_from"), item.get("direction"),
                             item.get("image_cell"), item.get("color"), item.get("shape")) == key), None)
        if existing is None:
            self.map.targets.append(record)
            if estimated_pos is not None:
                print(f"  [MAP TARGET] {record['color']} {record['shape']} -> estimated cell {tuple(estimated_pos)}")
            else:
                print(f"  [MAP TARGET] {record['color']} {record['shape']} seen from "
                      f"cell {tuple(record['observed_from'])}, toward {direction_name}; range unknown")
        else:
            existing.update(record)
        self._save_progress()
        self._tick_dashboard(f"TARGET {record.get('color')} FROM {tuple(record['observed_from'])} {direction_name}")

    def _tick_dashboard(self, status_text: str = ""):
        if not self.dashboard or not self.sys_runner.robot:
            return
        status = status_text or self.controller.current_action
        self.target_scan_module.pump_camera_stream(
            self.sys_runner.robot.camera,
            "RoboMaster Explorer | Map + Live Camera",
            status_text=status,
            read_timeout=0.01,
        )

    def _scan_and_fire_targets(self, wall_directions=None):
        """Run the attached gimbal camera workflow once before a grid move."""
        if not self.target_scan_module or not self.target_scan_config:
            return True
        config = self.target_scan_config
        # Only lower the camera toward directions where the wall scan found a
        # wall, and preserve the user's direction selection as an allowlist.
        selected_directions = set(config["directions"])
        if wall_directions is not None:
            selected_directions.intersection_update(wall_directions)
        if not selected_directions:
            print("  [Gimbal Target Scan] ไม่มีด้านที่พบกำแพงและเลือกให้สแกน; ข้ามการตรวจ")
            return True
        try:
            shots = self.target_scan_module.scan_and_fire_directions_once(
                ep_robot=self.sys_runner.robot,
                color_name=config["color"],
                shape_name=config["shape"],
                tilt_pitch=config["pitch"],
                scan_sec=config["scan_sec"],
                auto_fire=config["auto_fire"],
                fire_on_detection=config["fire_on_detection"],
                shots_per_target=config["shots"],
                fire_mode=config["fire_mode"],
                enabled_directions=[d for d in ("front", "right", "back", "left")
                                    if d in selected_directions],
                target_memory=self.target_memory,
                target_physical_size_m=config.get("target_size_m", 0.0),
                sensor_hub=self.sensor_hub,
                robot_grid=(self.current_col, self.current_row),
                robot_heading=self.current_direction,
                grid_size_m=self.grid_size_m,
                on_target=lambda direction_id, target, range_m, target_pos:
                    self._record_detected_target(direction_id, target, range_m, target_pos),
            )
            if shots is None:
                self.stop()
                self.controller.emergency_stop()
                print("  [Gimbal Target Scan] ยกเลิกการสำรวจตามคำสั่งจากหน้ากล้อง")
                return False
            print(f"  [Gimbal Target Scan] ยิงไป {shots} นัดในช่องนี้")
            return True
        except Exception as exc:
            self.controller.emergency_stop()
            print(f"  [Gimbal Target Scan Error] หยุดเพื่อความปลอดภัย: {exc}")
            self.stop()
            return False

    def _expand_open_edge(self, direction: int):
        """Treat a sensor-confirmed opening at the current map edge as another grid."""
        if self.mock_mode:
            return
        at_edge = {
            0: self.current_row == 0,
            1: self.current_col == self.cols - 1,
            2: self.current_row == self.rows - 1,
            3: self.current_col == 0,
        }[direction]
        if not at_edge:
            return

        old_col, old_row = self.current_col, self.current_row
        self.current_col, self.current_row = self.map.expand_edge(
            direction, self.current_col, self.current_row
        )
        dc = self.current_col - old_col
        dr = self.current_row - old_row
        self.cols, self.rows = self.map.cols, self.map.rows
        if dc or dr:
            self.start_pos = (self.start_pos[0] + dc, self.start_pos[1] + dr)
            for entry in self.history:
                entry["pos"][0] += dc
                entry["pos"][1] += dr
            if self._backtrack_stack is not None:
                for index, (col, row) in enumerate(self._backtrack_stack):
                    self._backtrack_stack[index] = (col + dc, row + dr)
            if self._visited_states is not None:
                shifted_states = {
                    (col + dc, row + dr, heading)
                    for col, row, heading in self._visited_states
                }
                self._visited_states.clear()
                self._visited_states.update(shifted_states)

    @property
    def controller(self) -> RobotControllerThread:
        return self.sys_runner.thread_2_controller

    @property
    def sensor_hub(self) -> SensorHub:
        return self.sys_runner.sensor_hub

    # -----------------------------------------------------------------------
    # Gimbal 4-Direction Wall Scanning & Classification
    # -----------------------------------------------------------------------
    def scan_walls(self) -> Tuple[bool, bool, bool, bool]:
        """Scans walls in all 4 directions (Front, Right, Back, Left) using active Gimbal rotation.

        Rotates gimbal:
          1. Front (yaw = 0°)
          2. Right (yaw = -90°)
          3. Back  (yaw = 180°)
          4. Left  (yaw = +90°)
        Then recenters to 0° (Front).

        Returns: (has_front, has_right, has_back, has_left)
        """
        print(f"  [Gimbal 360° Scan @ ({self.current_col}, {self.current_row})] Heading: {DIRECTION_NAMES[self.current_direction]}")
        relative_dirs = {
            "front": self.current_direction,
            "right": (self.current_direction + 1) % 4,
            "back": (self.current_direction + 2) % 4,
            "left": (self.current_direction + 3) % 4,
        }

        # Always use a fresh scan for movement and target-direction decisions.
        # A previously scanned cell can have stale OPEN readings (for example,
        # the ToF may later stop the robot at a wall).
        self._tick_dashboard("SCANNING WALLS / TOF")

        def mock_distance(direction_name: str) -> float:
            world_dir = relative_dirs[direction_name]
            has_wall = self.sim_maze.query_wall_by_direction(
                self.current_col, self.current_row, world_dir
            )
            return 145.0 if has_wall else 750.0

        # Use the shared GimbalScanController so the ToF reading is taken only
        # after the gimbal reaches each direction and settles.
        scan = self.gimbal_scanner.scan_4_directions(
            mock_distance_provider=mock_distance if self.mock_mode and self.sim_maze else None
        )
        wall_results = {name: result[0] for name, result in scan.items()}
        dist_results = {name: result[1] for name, result in scan.items()}
        scan_validity = self.gimbal_scanner.last_scan_validity

        # Keep the requested grid dimensions fixed; an open reading at the
        # boundary must not silently grow a 6x6 map into 6x7.
        for name, world_dir in relative_dirs.items():
            if not scan_validity.get(name, False):
                continue
            self.map.set_wall(
                self.current_col,
                self.current_row,
                WORLD_WALL_KEYS[world_dir],
                wall_results[name],
            )

        # Only mark the cell scanned when all four directional readings are
        # valid, so a partial sensor failure does not become permanent cache.
        self.map.cells[self.current_row][self.current_col]["scanned"] = all(
            scan_validity.get(name, False) for name in relative_dirs
        )
        self._save_progress()
        self._tick_dashboard("WALL SCAN COMPLETE")

        if self.mock_mode and self.sim_maze:
            has_front_wall = self.sim_maze.query_wall_by_direction(self.current_col, self.current_row, self.current_direction)
            sim_front_dist = 145.0 if has_front_wall else 750.0
            if self.sys_runner.thread_1_sensor:
                self.sys_runner.thread_1_sensor.inject_mock_data(
                    tof_dist=sim_front_dist,
                    gimbal_yaw=0.0,
                )

        hf = wall_results["front"]
        hr = wall_results["right"]
        hb = wall_results["back"]
        hl = wall_results["left"]

        print(
            f"  [Gimbal Wall Results] "
            f"Front: {dist_results['front']:.0f}mm ({'WALL' if hf else 'OPEN' if scan_validity.get('front') else 'UNKNOWN'}) | "
            f"Right: {dist_results['right']:.0f}mm ({'WALL' if hr else 'OPEN' if scan_validity.get('right') else 'UNKNOWN'}) | "
            f"Back: {dist_results['back']:.0f}mm ({'WALL' if hb else 'OPEN' if scan_validity.get('back') else 'UNKNOWN'}) | "
            f"Left: {dist_results['left']:.0f}mm ({'WALL' if hl else 'OPEN' if scan_validity.get('left') else 'UNKNOWN'})"
        )
        return hf, hr, hb, hl

    # -----------------------------------------------------------------------
    # Motion Execution with Collision Avoidance
    # -----------------------------------------------------------------------
    def turn_to_direction(self, target_direction: int):
        """Rotates the robot to face target_direction (0: North, 1: East, 2: South, 3: West)."""
        if target_direction == self.current_direction:
            return

        diff = (target_direction - self.current_direction) % 4
        if diff == 1:
            print(f"  [Action] Turn Right -> {DIRECTION_NAMES[target_direction]}")
            self.controller.turn_right(90.0)
        elif diff == 2:
            print(f"  [Action] Turn Around (180°) -> {DIRECTION_NAMES[target_direction]}")
            self.controller.turn_around()
        elif diff == 3:
            print(f"  [Action] Turn Left -> {DIRECTION_NAMES[target_direction]}")
            self.controller.turn_left(90.0)

        self.current_direction = target_direction

        if self.mock_mode and self.sim_maze:
            has_front_wall = self.sim_maze.query_wall_by_direction(self.current_col, self.current_row, self.current_direction)
            sim_front_dist = 145.0 if has_front_wall else 750.0
            if self.sys_runner.thread_1_sensor:
                self.sys_runner.thread_1_sensor.inject_mock_data(
                    tof_dist=sim_front_dist,
                    yaw=float(self.current_direction * 90.0),
                    gimbal_yaw=0.0,
                )

        time.sleep(0.05 if self.mock_mode else 0.1)

    def step_forward_one_cell(self) -> bool:
        """Moves forward exactly 1 grid cell (60 cm) with Step 3 PID centering and collision checks.

        Returns True if movement succeeded, False if aborted due to obstacle.
        """
        # Take a fresh Gimbal/ToF reading in the intended travel direction
        # immediately before entering the next grid cell.
        has_front, has_right, has_back, has_left = self.scan_walls()
        # The camera workflow may track a target while streaming frames. Keep
        # the chassis explicitly stopped until gimbal scan/aim/fire returns.
        if self.controller:
            self.controller.stop_chassis()
        self._tick_dashboard("BASE STOPPED | WALL SCAN COMPLETE")
        current_cell = self.map.cells[self.current_row][self.current_col]
        if self.target_scan_config and not current_cell.get("targets_scanned", False):
            relative_wall_directions = {
                name for name, is_wall in (
                    ("front", has_front), ("right", has_right),
                    ("back", has_back), ("left", has_left),
                ) if is_wall
            }
            selected_wall_directions = relative_wall_directions.intersection(
                self.target_scan_config["directions"]
            )
            if selected_wall_directions:
                # Create enough front clearance before the camera starts
                # checking, aiming, or firing at a target. Keep at least
                # 250 mm of measured front clearance.
                state = self.sensor_hub.get_latest_state()
                if not self.mock_mode:
                    if not state.tof_valid or state.tof_filtered_mm is None:
                        print("  [TARGET SCAN BLOCKED] ToF is invalid; refusing to aim/fire without front clearance.")
                        return False
                    if state.tof_filtered_mm < 250.0:
                        print(
                            f"  [TARGET SCAN] Front clearance is {state.tof_filtered_mm:.1f} mm; "
                            "backing away to 250 mm before checking the target."
                        )
                        if not self.controller.back_away_from_front_wall(
                            target_mm=250.0, tolerance_mm=0.0
                        ):
                            print("  [TARGET SCAN BLOCKED] Could not create safe front clearance; skipping aim/fire.")
                            return False
                        state = self.sensor_hub.get_latest_state()
                        if (not state.tof_valid or state.tof_filtered_mm is None
                                or state.tof_filtered_mm < 250.0):
                            print("  [TARGET SCAN BLOCKED] Front distance is still below 250 mm; skipping aim/fire.")
                            return False

                if not self._scan_and_fire_targets(relative_wall_directions):
                    return False
                current_cell["targets_scanned"] = True
            else:
                print("  [Gimbal Target Scan] ไม่มีด้านที่พบกำแพงในทิศที่เลือก; จะตรวจใหม่เมื่อสแกนพบกำแพง")
        if self.map.has_wall(
            self.current_col, self.current_row, WORLD_WALL_KEYS[self.current_direction]
        ) is not False:
            print("  [MOVE BLOCKED] No valid open-path reading for the next grid.")
            return False
        if has_front:
            print("  [MOVE BLOCKED] Gimbal scan detected a wall ahead; staying in current grid.")
            self.map.set_wall(
                self.current_col,
                self.current_row,
                WORLD_WALL_KEYS[self.current_direction],
                True,
                force=True,
            )
            return False

        dc, dr = DIRECTION_OFFSETS[self.current_direction]
        target_c = self.current_col + dc
        target_r = self.current_row + dr
        if not self.map.is_valid_coord(target_c, target_r):
            print(f"  [COLLISION PREVENTED] Target cell ({target_c}, {target_r}) is out of the discovered map.")
            return False

        # Live safety check: read ToF before commanding movement
        state = self.sensor_hub.get_latest_state()
        if (not self.mock_mode and
                (not state.tof_valid or state.tof_filtered_mm is None)):
            self.controller.stop_chassis()
            print("  [MOVE BLOCKED] ToF reading is invalid; refusing to move without a front-distance check.")
            return False
        has_front, _, _ = self.controller.wall_pid.classify_wall_state(state)
        if has_front and state.tof_filtered_mm is not None and state.tof_filtered_mm < 250.0:
            print(
                f"  [COLLISION PREVENTED] ToF front sensor is closer than 250 mm ({state.tof_filtered_mm:.1f} mm)! "
                f"Backing away before stopping this route step."
            )
            # Match the in-step safety behavior: retreat to a safe clearance
            # before aborting, instead of merely refusing to issue a command.
            self.controller.back_away_from_front_wall(target_mm=250.0, tolerance_mm=0.0)
            # Update map with confirmed wall
            self.map.set_wall(
                self.current_col, self.current_row,
                WORLD_WALL_KEYS[self.current_direction], True, force=True,
            )
            self.map.cells[self.current_row][self.current_col]["targets_scanned"] = False
            return False

        print(f"  [Action] Moving forward 1 cell into ({target_c}, {target_r})...")

        # Execute 1 cell move with Step 3 closed-loop PID centering
        if not self.controller.navigate_single_grid_step(step_idx=1, total_steps=1):
            stopped_state = self.sensor_hub.get_latest_state()
            if (stopped_state.tof_valid and stopped_state.tof_filtered_mm is not None
                    and stopped_state.tof_filtered_mm < 250.0):
                self.map.set_wall(
                    self.current_col, self.current_row,
                    WORLD_WALL_KEYS[self.current_direction], True, force=True,
                )
                self.map.cells[self.current_row][self.current_col]["targets_scanned"] = False
            self._save_progress()
            print("  [MOVE ABORTED] Grid step did not reach the next cell; map position unchanged.")
            return False

        self.step_count += 1

        # Update current robot coordinates
        self.current_col = target_c
        self.current_row = target_r
        self.map.mark_visited(self.current_col, self.current_row)
        self.completed_grid_steps += 1
        if self.completed_grid_steps % 2 == 0:
            self.controller.trim_heading(max_adjust_deg=2.0)
        self._tick_dashboard("GRID MOVE COMPLETE")

        if self.mock_mode and self.sim_maze:
            has_front_wall = self.sim_maze.query_wall_by_direction(self.current_col, self.current_row, self.current_direction)
            sim_front_dist = 145.0 if has_front_wall else 750.0
            if self.sys_runner.thread_1_sensor:
                self.sys_runner.thread_1_sensor.inject_mock_data(
                    tof_dist=sim_front_dist,
                    yaw=float(self.current_direction * 90.0),
                    gimbal_yaw=0.0,
                )

        # Know for certain that passage behind us is open
        opposite_wall = OPPOSITE_WALL_KEYS[WORLD_WALL_KEYS[self.current_direction]]
        self.map.set_wall(self.current_col, self.current_row, opposite_wall, False)

        # Record step history
        self.history.append({
            "step": self.step_count,
            "pos": [self.current_col, self.current_row],
            "facing": DIRECTION_NAMES[self.current_direction],
        })
        self._save_progress()

        if self.step_delay > 0:
            time.sleep(self.step_delay)

        return True

    # -----------------------------------------------------------------------
    # Exploration Algorithm 1: Depth-First Search with Backtracking (DFS)
    # -----------------------------------------------------------------------
    def explore_dfs(self, max_steps: int = 150):
        """Explores all reachable cells in the maze using DFS and backtracks to unvisited branches."""
        print("\n=======================================================")
        print("[EXPLORER] STARTING AUTONOMOUS MAZE EXPLORATION (DFS BACKTRACKING)")
        print(f"   Start: ({self.current_col}, {self.current_row}) | Strategy: Depth-First Search")
        print("   Wall Avoidance: Multi-Layer PID Centering & ToF Lock")
        print("=======================================================\n")

        backtrack_stack: List[Tuple[int, int]] = []
        self._backtrack_stack = backtrack_stack
        self._exploration_started_at = time.monotonic()
        self._running = True

        while self._running and self.step_count < max_steps:
            # 1. Scan and classify walls in 4 directions using active Gimbal rotation
            has_front, has_right, has_back, has_left = self.scan_walls()

            # Render live ASCII map in terminal
            print(f"Elapsed: {format_duration(self.elapsed_exploration_seconds())}")
            print("\n" + self.map.render_ascii(self.current_col, self.current_row, self.current_direction) + "\n")
            self._save_progress()

            # 2. Look for adjacent accessible unvisited neighbors
            # Preference order: Front, Left, Right, Back
            preferred_dir_order = [
                self.current_direction,
                (self.current_direction + 3) % 4,
                (self.current_direction + 1) % 4,
                (self.current_direction + 2) % 4,
            ]

            chosen_dir = None
            for d in preferred_dir_order:
                w_key = WORLD_WALL_KEYS[d]
                # Is there a wall blocking this direction?
                if self.map.has_wall(self.current_col, self.current_row, w_key) is not False:
                    continue

                dc, dr = DIRECTION_OFFSETS[d]
                nc, nr = self.current_col + dc, self.current_row + dr
                if self.map.is_valid_coord(nc, nr) and not self.map.is_visited(nc, nr):
                    chosen_dir = d
                    break

            if chosen_dir is not None:
                # Found an unvisited branch! Move forward along it
                backtrack_stack.append((self.current_col, self.current_row))
                self.turn_to_direction(chosen_dir)
                success = self.step_forward_one_cell()
                if not success:
                    # If blocked by unexpected wall, pop backtrack stack
                    if backtrack_stack:
                        backtrack_stack.pop()
            else:
                # No unvisited neighbor from current cell -> Dead end or fully explored branch!
                if not backtrack_stack:
                    print("\n[EXPLORATION COMPLETE] All reachable cells have been fully explored and mapped!")
                    break

                # Backtrack to the nearest cell with unvisited branches
                prev_c, prev_r = backtrack_stack.pop()
                print(f"  [Backtracking] Dead-end / branch complete. Returning to ({prev_c}, {prev_r})...")

                # Find direction from current cell to backtrack cell
                back_dir = None
                for d in range(4):
                    dc, dr = DIRECTION_OFFSETS[d]
                    if self.current_col + dc == prev_c and self.current_row + dr == prev_r:
                        back_dir = d
                        break

                if back_dir is not None:
                    self.turn_to_direction(back_dir)
                    if not self.step_forward_one_cell():
                        print("  [Backtracking] Move was blocked; keeping current position and recalculating route.")
                else:
                    print(f"  [Warning] Backtrack target ({prev_c}, {prev_r}) is not adjacent to ({self.current_col}, {self.current_row}); discarding stale path entry and continuing search.")

        self.finish_exploration()

    # -----------------------------------------------------------------------
    # Exploration Algorithm 2: Wall-Following (Right-Hand / Left-Hand Rule)
    # -----------------------------------------------------------------------
    def explore_wall_follow(self, side: str = "right", max_steps: int = 80):
        """Explores maze following the selected perimeter wall (Right-hand or Left-hand rule)."""
        print(f"\n[EXPLORER] STARTING WALL-FOLLOWING EXPLORATION ({side.upper()}-HAND RULE)...")
        self._exploration_started_at = time.monotonic()
        self._running = True

        visited_states: Set[Tuple[int, int, int]] = set()
        self._visited_states = visited_states

        while self._running and self.step_count < max_steps:
            has_front, has_right, has_back, has_left = self.scan_walls()
            print(f"Elapsed: {format_duration(self.elapsed_exploration_seconds())}")
            print("\n" + self.map.render_ascii(self.current_col, self.current_row, self.current_direction) + "\n")
            self._save_progress()

            state_key = (self.current_col, self.current_row, self.current_direction)
            if state_key in visited_states and self.step_count > 4:
                print("  [Loop Detected] Completed circuit along wall boundary.")
                break
            visited_states.add(state_key)

            if side == "right":
                # Right-hand rule: Try Right -> Straight -> Left -> Turn Around
                dir_candidates = [
                    (self.current_direction + 1) % 4,  # Right
                    self.current_direction,            # Straight
                    (self.current_direction + 3) % 4,  # Left
                    (self.current_direction + 2) % 4,  # Around
                ]
            else:
                # Left-hand rule: Try Left -> Straight -> Right -> Turn Around
                dir_candidates = [
                    (self.current_direction + 3) % 4,  # Left
                    self.current_direction,            # Straight
                    (self.current_direction + 1) % 4,  # Right
                    (self.current_direction + 2) % 4,  # Around
                ]

            moved = False
            for d in dir_candidates:
                w_key = WORLD_WALL_KEYS[d]
                if self.map.has_wall(self.current_col, self.current_row, w_key) is not False:
                    continue
                dc, dr = DIRECTION_OFFSETS[d]
                nc, nr = self.current_col + dc, self.current_row + dr
                if not self.map.is_valid_coord(nc, nr):
                    continue

                self.turn_to_direction(d)
                if self.step_forward_one_cell():
                    moved = True
                    break

            if not moved:
                print("  [Trapped] Robot is completely enclosed by walls!")
                break

        self.finish_exploration()

    def finish_exploration(self):
        """Finalizes exploration, stops chassis, and exports the discovered map."""
        if self._finished:
            return
        if self._exploration_started_at is not None:
            self.exploration_duration_seconds = max(
                0.0, time.monotonic() - self._exploration_started_at
            )
        self.map.exploration_time_seconds = self.exploration_duration_seconds
        self._finished = True
        self._running = False
        if self.controller:
            self.controller.stop_chassis()
        if self.target_scan_module and self.sys_runner.robot:
            self._tick_dashboard("EXPLORATION COMPLETE")
            try:
                self.sys_runner.robot.camera.stop_video_stream()
            except Exception:
                pass
            self.target_scan_module.set_live_dashboard(None)

        print("\n" + "=" * 65)
        print("[EXPLORATION SUMMARY]")
        print("=" * 65)
        print(f"Total Steps Taken : {self.step_count}")
        print(f"Exploration Time  : {format_duration(self.exploration_duration_seconds)} "
              f"({self.exploration_duration_seconds:.1f} seconds)")
        print(f"Cells Visited     : {self.map.visited_count} / {self.cols * self.rows} ({self.map.visited_count / (self.cols * self.rows) * 100:.1f}%)")
        print(f"Final Robot Pos   : ({self.current_col}, {self.current_row}) Facing {DIRECTION_NAMES[self.current_direction]}")
        print("\n" + self.map.render_ascii(self.current_col, self.current_row, self.current_direction) + "\n")

        # Save map to JSON
        self.map.save_json(self.output_file, self.start_pos, self.history)
        print("=" * 65)

    def stop(self):
        self._running = False


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
def main(cli_args: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Autonomous Grid Maze Exploration with Wall Avoidance for RoboMaster EP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mock", action="store_true", help="Run in Mock Simulation mode (no physical robot needed)")
    parser.add_argument("--conn-type", choices=("ap", "sta"), default="ap", help="Connection mode to RoboMaster EP")
    parser.add_argument("--strategy", choices=("dfs", "wall-follow"), default="dfs", help="Exploration strategy algorithm")
    parser.add_argument("--grid", "--size", dest="grid", default=None, help="Grid size formatted as COLSxROWS (e.g. 4x4, 5x6, 6x6)")
    parser.add_argument("--rows", type=int, default=None, help="Grid rows count (default: 6)")
    parser.add_argument("--cols", type=int, default=None, help="Grid columns count (default: 6)")
    parser.add_argument("--cell-size", type=float, default=0.60, help="Grid cell size in meters (default: 0.60)")
    parser.add_argument("--cell-size-cm", type=float, default=None, help="Grid cell size in centimeters (e.g. 60 or 50)")
    parser.add_argument("--start", type=parse_grid_position, default=None,
                        help="Initial grid cell as COL,ROW (zero-based; e.g. 2,3)")
    parser.add_argument("--start-col", type=int, default=None, help="Initial column index (default: 0)")
    parser.add_argument("--start-row", type=int, default=None, help="Initial row index (default: rows - 1)")
    parser.add_argument("--sim-map", default="data/robot_map_plan.json", help="Ground truth map for simulation mode")
    parser.add_argument("--output", default="data/explored_map.json", help="Output file path to save discovered map")
    parser.add_argument("--speed", type=float, default=0.22, help="Base forward speed (m/s)")
    parser.add_argument("--step-delay", type=float, default=0.4, help="Pause delay between steps (s)")
    parser.add_argument("--max-steps", type=int, default=120, help="Maximum number of steps before stopping")

    args = parser.parse_args(cli_args)

    # Resolve and validate the selected start before showing camera setup or
    # connecting to the physical robot.
    cols, rows = parse_grid_dimensions(args.grid, args.cols, args.rows, default_cols=6, default_rows=6)
    if args.start is not None and (args.start_col is not None or args.start_row is not None):
        parser.error("Use either --start COL,ROW or --start-col/--start-row, not both")
    if args.start is not None:
        start_col, start_row = args.start
    else:
        start_col = args.start_col if args.start_col is not None else 0
        start_row = args.start_row if args.start_row is not None else (rows - 1)
    if not (0 <= start_col < cols and 0 <= start_row < rows):
        parser.error(
            f"Start position ({start_col},{start_row}) is outside the {cols}x{rows} grid; "
            f"valid columns are 0-{cols - 1} and rows are 0-{rows - 1}"
        )

    target_scan_config = None
    if not args.mock:
        try:
            from gimbal_scan_main import GimbalScanConfigDialog
            (
                target_color,
                target_shape,
                target_pitch,
                target_scan_sec,
                target_auto_fire,
                target_fire_on_detection,
                target_shots,
                target_size_cm,
                target_fire_mode,
                target_directions,
                confirmed,
            ) = GimbalScanConfigDialog().run()
        except Exception as exc:
            print(f"[Gimbal Scan Setup Error] เปิดหน้าตั้งค่าสแกนไม่ได้: {exc}")
            return 1
        if not confirmed:
            print("[Explorer] ยกเลิกการสำรวจจากหน้าตั้งค่าสแกน")
            return 0
        target_scan_config = {
            "color": target_color,
            "shape": target_shape,
            "pitch": target_pitch,
            "scan_sec": target_scan_sec,
            "auto_fire": target_auto_fire,
            "fire_on_detection": target_fire_on_detection,
            "shots": target_shots,
            "target_size_m": target_size_cm / 100.0,
            "fire_mode": target_fire_mode,
            "directions": target_directions,
        }
        print(
            "[Target Scan Selection] "
            f"color={target_color} | shape={target_shape} | pitch={target_pitch}° | "
            f"directions={','.join(target_directions)} | fire={'ON' if target_auto_fire else 'OFF'} | "
            f"timing={'on-detection' if target_fire_on_detection else 'center-lock'} | "
            f"shots={target_shots} | target_size={target_size_cm:g}cm | mode={target_fire_mode}"
        )

    # Resolve cell size after grid dimensions and start cell are validated.
    cell_size_m = (args.cell_size_cm / 100.0) if args.cell_size_cm is not None else args.cell_size

    print("=" * 70)
    print("[ROBOMASTER EP] AUTONOMOUS GRID EXPLORATION & MAPPING SYSTEM")
    print(f"Mode: {'SIMULATION / MOCK' if args.mock else 'LIVE ROBOT (' + args.conn_type.upper() + ')'}")
    print(f"Strategy: {args.strategy.upper()} | Grid: {cols}x{rows} ({cell_size_m*100:.0f}cm/cell) | Start: ({start_col}, {start_row})")
    print("=" * 70)

    # 1. Initialize Robot System
    sys_runner = RobotSystem(
        mock_mode=args.mock,
        conn_type=args.conn_type,
        sensor_rate_hz=20.0,
    )

    if not sys_runner.connect_robot():
        if not args.mock:
            print("\n" + "=" * 70)
            print("[ERROR] Cannot connect to physical RoboMaster EP robot!")
            print(f"  • Connection Mode: {args.conn_type.upper()}")
            print("  • Troubleshooting steps:")
            print("    1. Make sure your PC is connected to the robot Wi-Fi (e.g. 'RM_XXXXXX').")
            print("    2. Check that the robot Wi-Fi switch is set to Direct Connection (AP).")
            print("    3. If you want to test in simulation mode without the robot, use '--mock'.")
            print("=" * 70 + "\n")
            return 1

    # 2. Setup threads (Thread 1: Sensor Collection, Thread 2: Robot Controller)
    sys_runner.setup_threads()
    if sys_runner.thread_2_controller:
        sys_runner.thread_2_controller.base_speed = args.speed
        sys_runner.thread_2_controller.grid_size_m = cell_size_m
        if sys_runner.mock_mode:
            sys_runner.thread_2_controller.step_pause_sec = 0.01
            if sys_runner.thread_2_controller.mock_actuator:
                sys_runner.thread_2_controller.mock_actuator.speed_mult = 15.0
        else:
            sys_runner.thread_2_controller.step_pause_sec = 0.5

    # 3. Create Autonomous Explorer
    explorer = AutonomousExplorer(
        robot_system=sys_runner,
        cols=cols,
        rows=rows,
        grid_size_m=cell_size_m,
        start_col=start_col,
        start_row=start_row,
        strategy=args.strategy,
        sim_map_file=args.sim_map,
        output_file=args.output,
        step_delay=args.step_delay,
        target_scan_config=target_scan_config,
    )

    # Graceful exit handler
    def handle_sigint(sig, frame):
        print("\n[Interrupt] Caught signal, stopping exploration safely...")
        explorer.stop()
        if sys_runner.thread_2_controller:
            sys_runner.thread_2_controller.emergency_stop()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_sigint)

    # 4. Start Thread 1 (Sensors)
    print("\n[Explorer] Starting Thread 1 (Sensor Pipeline)...")
    if target_scan_config and sys_runner.robot:
        try:
            if not sys_runner.robot.camera.start_video_stream(display=False):
                raise RuntimeError("SDK did not start the video stream")
            sys_runner.robot.gimbal.sub_angle(
                freq=20, callback=explorer.target_scan_module.on_gimbal_angle_cb
            )
            print("[Explorer] Gimbal target camera is ready; scan/shoot runs before each move.")
        except Exception as exc:
            print(f"[Explorer] Cannot start target scan camera/gimbal: {exc}")
            try:
                sys_runner.robot.camera.stop_video_stream()
            except Exception:
                pass
            sys_runner.shutdown(save_telemetry=False, run_analysis=False)
            return 1
    sys_runner.thread_1_sensor.start_collecting()
    time.sleep(0.3)

    try:
        if args.strategy == "dfs":
            explorer.explore_dfs(max_steps=args.max_steps)
        elif args.strategy == "wall-follow":
            explorer.explore_wall_follow(side="right", max_steps=args.max_steps)
    except KeyboardInterrupt:
        print("\n[Explorer] Interrupted by user.")
    finally:
        explorer.finish_exploration()
        sys_runner.shutdown(save_telemetry=True, run_analysis=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
