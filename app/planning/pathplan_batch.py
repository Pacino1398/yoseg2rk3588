from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import DEFAULT_CONFIG
from app.inference.segmentation import (
    get_default_data_yaml,
    get_default_project_dir,
    get_default_source_dir,
    get_default_weights,
    get_next_run_name,
    run_prediction,
)
from app.mapping.grid_map import CLASS_HEIGHTS, GridMapHandler, TRAVERSABLE_CLASSES, load_grouped_mask_entries
from app.paths import resolve_path
from app.planning.dstar_lite import DStarLite

IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp", ".pfm"}
VIDEO_SUFFIXES = {".asf", ".avi", ".gif", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".ts", ".wmv"}

PLANNER_GRID_COLOR = "gray"
PLANNER_PATH_COLOR = "#00FF00"
PLANNER_START_COLOR = "blue"
PLANNER_GOAL_COLOR = "red"
TRAVERSABLE_OVERLAY_COLOR = "#58FF58"
INFO_PANEL_EDGE_COLOR = "gray"
LABEL_OFFSETS = [
    (0.0, 0.0),
    (0.35, 0.0),
    (-0.35, 0.0),
    (0.0, 0.35),
    (0.0, -0.35),
    (0.35, 0.35),
    (-0.35, 0.35),
    (0.35, -0.35),
    (-0.35, -0.35),
]
MIN_LABEL_DISTANCE = 0.75


def get_default_pathplan_project_dir() -> Path:
    return DEFAULT_CONFIG.runs_dir / "pathplan"


def get_latest_segmentation_run_dir(project_dir: str | Path) -> Path:
    project_path = Path(project_dir)
    if not project_path.exists():
        raise FileNotFoundError(f"未找到分割输出目录: {project_path}")

    run_dirs = [
        path
        for path in project_path.iterdir()
        if path.is_dir() and path.name.startswith("exp") and (path.name == "exp" or path.name[3:].isdigit())
    ]
    if not run_dirs:
        raise FileNotFoundError(f"未找到分割实验目录: {project_path}")

    def sort_key(path: Path) -> int:
        suffix = path.name[3:]
        return 0 if suffix == "" else int(suffix)

    return sorted(run_dirs, key=sort_key)[-1]


def create_pathplan_run_dir(project_dir: str | Path) -> Path:
    project_path = Path(project_dir)
    project_path.mkdir(parents=True, exist_ok=True)

    first_run = project_path / "exp"
    if not first_run.exists():
        first_run.mkdir(parents=True, exist_ok=True)
        return first_run

    index = 1
    while True:
        run_dir = project_path / f"exp{index}"
        if not run_dir.exists():
            run_dir.mkdir(parents=True, exist_ok=True)
            return run_dir
        index += 1


def get_next_segmentation_run_dir(project_dir: str | Path) -> Path:
    project_path = Path(project_dir)
    return project_path / get_next_run_name(project_path, "exp")


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


def is_video_file(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_SUFFIXES


def iter_source_media(source_path: Path) -> list[Path]:
    if source_path.is_file():
        if is_image_file(source_path) or is_video_file(source_path):
            return [source_path]
        raise FileNotFoundError(f"不支持的输入文件: {source_path}")

    if source_path.is_dir():
        media_files = [
            item
            for item in sorted(source_path.iterdir())
            if item.is_file() and (is_image_file(item) or is_video_file(item))
        ]
        if media_files:
            return media_files
        raise FileNotFoundError(f"目录内未找到图片或视频: {source_path}")

    raise FileNotFoundError(f"输入不存在: {source_path}")


def get_frame_stem(media_path: Path, frame_index: int) -> str:
    return f"{media_path.stem}_frame{frame_index:06d}"


def cell_to_pixel(cell: tuple[int, int], grid_scale: int) -> tuple[int, int]:
    return (cell[0] * grid_scale + grid_scale // 2, cell[1] * grid_scale + grid_scale // 2)


def get_mask_canvas_shape(mask_entries: list[list]) -> tuple[int, int] | None:
    for entry in mask_entries:
        if len(entry) < 4:
            continue
        mask = entry[3]
        if isinstance(mask, np.ndarray) and mask.ndim >= 2:
            return int(mask.shape[0]), int(mask.shape[1])
    return None


def get_video_canvas_shape(
    video_path: Path,
    mask_groups: dict[str, list[list]],
    fallback_shape: tuple[int, int],
) -> tuple[int, int]:
    frame_prefix = f"{video_path.stem}_frame"
    for stem in sorted(mask_groups):
        if not stem.startswith(frame_prefix):
            continue
        canvas_shape = get_mask_canvas_shape(mask_groups[stem])
        if canvas_shape is not None:
            return canvas_shape
    return fallback_shape


def load_class_names(data_yaml: str | Path | None = None) -> dict[int, str]:
    yaml_path = resolve_path(data_yaml, DEFAULT_CONFIG.data_yaml) if data_yaml is not None else DEFAULT_CONFIG.data_yaml
    try:
        with Path(yaml_path).open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
    except Exception:
        return {-1: "manual"}

    names = data.get("names", {})
    class_names = {-1: "manual"}
    if isinstance(names, dict):
        for key, value in names.items():
            try:
                class_id = int(key)
            except (TypeError, ValueError):
                continue
            class_names[class_id] = str(value)
    elif isinstance(names, list):
        for class_id, value in enumerate(names):
            class_names[class_id] = str(value)
    return class_names


def get_obstacle_facecolor(cell: tuple[int, int], obstacle_heights: dict[tuple[int, int], int]) -> tuple[float, float, float]:
    height = obstacle_heights.get(cell)
    if height is None:
        return (0.0, 0.0, 0.0)

    height_levels = sorted(set(CLASS_HEIGHTS.values()) | set(obstacle_heights.values()))
    if not height_levels:
        return (0.0, 0.0, 0.0)
    if len(height_levels) == 1:
        gray = 0.2
        return (gray, gray, gray)

    level_index = height_levels.index(height)
    ratio = level_index / (len(height_levels) - 1)
    gray = 0.92 - 0.8 * ratio
    return (gray, gray, gray)


def get_obstacle_color(cell: tuple[int, int], obstacle_heights: dict[tuple[int, int], int]) -> tuple[int, int, int]:
    gray = get_obstacle_facecolor(cell, obstacle_heights)[0]
    gray_u8 = int(round(gray * 255))
    gray_u8 = max(0, min(255, gray_u8))
    return (gray_u8, gray_u8, gray_u8)


def get_annotation_text_color(cell: tuple[int, int], obstacle_heights: dict[tuple[int, int], int]) -> str:
    gray = get_obstacle_facecolor(cell, obstacle_heights)[0]
    return "white" if gray < 0.45 else "black"


def get_annotation_color(cell: tuple[int, int], obstacle_heights: dict[tuple[int, int], int]) -> tuple[int, int, int]:
    return (255, 255, 255) if get_annotation_text_color(cell, obstacle_heights) == "white" else (0, 0, 0)


def format_mask_instance_label(instance: dict[str, object], class_names: dict[int, str]) -> str:
    class_id = int(instance.get("class_id", -1))
    class_name = class_names.get(class_id, f"class_{class_id}")
    mask_index = instance.get("mask_index")
    if isinstance(mask_index, int):
        return f"{class_name}_{mask_index}"
    return class_name


def select_display_instances(mask_instances: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: list[dict[str, object]] = []

    for instance in mask_instances:
        cells = instance.get("cells")
        if not isinstance(cells, tuple) or not cells:
            continue
        cell_set = set(cells)
        if not cell_set:
            continue

        matched_group: dict[str, object] | None = None
        for group in groups:
            group_cells = group["cells"]
            if cell_set & group_cells:
                matched_group = group
                break

        if matched_group is None:
            groups.append({"cells": set(cell_set), "instances": [instance]})
            continue

        matched_group["cells"].update(cell_set)
        matched_group["instances"].append(instance)

        merged = True
        while merged:
            merged = False
            for other_group in list(groups):
                if other_group is matched_group:
                    continue
                if matched_group["cells"] & other_group["cells"]:
                    matched_group["cells"].update(other_group["cells"])
                    matched_group["instances"].extend(other_group["instances"])
                    groups.remove(other_group)
                    merged = True
                    break

    selected_instances: list[dict[str, object]] = []
    for group in groups:
        instances = group["instances"]
        best_instance = max(
            instances,
            key=lambda item: (
                float(item.get("confidence", 0.0)),
                len(item.get("cells", ())),
                -int(item.get("class_id", 0)),
            ),
        )
        selected_instances.append(best_instance)

    return selected_instances


def build_class_annotations(
    mask_instances: list[dict[str, object]],
    obstacle_heights: dict[tuple[int, int], int],
    class_names: dict[int, str],
) -> list[tuple[tuple[float, float], str, str]]:
    annotations: list[tuple[tuple[float, float], str, str]] = []
    used_positions: list[tuple[float, float]] = []

    for instance in select_display_instances(mask_instances):
        center_cell = instance.get("center_cell")
        if not isinstance(center_cell, tuple) or len(center_cell) != 2:
            continue

        x, y = center_cell
        base_position = (x + 0.5, y + 0.5)
        label = format_mask_instance_label(instance, class_names)
        text_color = get_annotation_text_color(center_cell, obstacle_heights)

        chosen_position = base_position
        best_position = base_position
        best_distance = -1.0
        for dx, dy in LABEL_OFFSETS:
            candidate = (base_position[0] + dx, base_position[1] + dy)
            if not used_positions:
                chosen_position = candidate
                break

            nearest_distance = min(
                ((candidate[0] - px) ** 2 + (candidate[1] - py) ** 2) ** 0.5
                for px, py in used_positions
            )
            if nearest_distance > best_distance:
                best_distance = nearest_distance
                best_position = candidate
            if nearest_distance >= MIN_LABEL_DISTANCE:
                chosen_position = candidate
                break
        else:
            chosen_position = best_position

        used_positions.append(chosen_position)
        annotations.append((chosen_position, label, text_color))

    return annotations


def build_class_info_lines(
    obstacle_class_ids: dict[tuple[int, int], int],
    obstacle_heights: dict[tuple[int, int], int],
    class_names: dict[int, str],
) -> str:
    present_class_ids = sorted(set(obstacle_class_ids.values()), key=lambda class_id: (class_id == -1, class_id))
    if not present_class_ids:
        return "calss: none"

    lines = ["obstacle classes:"]
    manual_height = max(CLASS_HEIGHTS.values()) if CLASS_HEIGHTS else 10
    for class_id in present_class_ids:
        class_name = class_names.get(class_id, f"class_{class_id}")
        if class_id == -1:
            lines.append(f"manual | height {manual_height}")
            continue

        height = CLASS_HEIGHTS.get(class_id)
        if height is None:
            matching_heights = [
                obstacle_heights[cell]
                for cell, current_class_id in obstacle_class_ids.items()
                if current_class_id == class_id and cell in obstacle_heights
            ]
            height = matching_heights[0] if matching_heights else 0

        if class_id in TRAVERSABLE_CLASSES:
            lines.append(f"{class_id}: {class_name} | h={height} | passable")
        else:
            lines.append(f"{class_id}: {class_name} | h={height}")
    return "\n".join(lines)


def render_plan_view(
    canvas_shape: tuple[int, int],
    grid_handler: GridMapHandler,
    path: list[tuple[int, int]],
    start: tuple[int, int],
    goal: tuple[int, int],
    grid_scale: int,
    class_names: dict[int, str] | None = None,
    show_labels: bool = True,
) -> np.ndarray:
    canvas_h, canvas_w = canvas_shape
    grid_w = max(1, canvas_w // grid_scale)
    grid_h = max(1, canvas_h // grid_scale)
    dpi = 100
    figure_w = max(canvas_w / dpi, 6.0)
    figure_h = max(canvas_h / dpi, 4.0)

    fig = Figure(figsize=(figure_w, figure_h), dpi=dpi, facecolor="white")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    fig.subplots_adjust(right=0.8)

    ax.set_xlim(0, grid_w)
    ax.set_ylim(0, grid_h)
    ax.invert_yaxis()
    ax.grid(True, color=PLANNER_GRID_COLOR, linewidth=0.2)
    ax.set_aspect("equal")
    ax.set_title(
        "D* Lite Dynamic Planner\nLeft:Add Obstacle | Middle:Set Start | Right:Set Goal",
        fontsize=12,
    )

    display_obs = grid_handler.display_obs if hasattr(grid_handler, "display_obs") else grid_handler.obstacles
    for x, y in display_obs:
        ax.add_patch(
            Rectangle(
                (x, y),
                1,
                1,
                facecolor=get_obstacle_facecolor((x, y), grid_handler.obstacle_heights),
                edgecolor="none",
            )
        )

    for x, y in grid_handler.traversable_obstacles:
        ax.add_patch(Rectangle((x, y), 1, 1, facecolor=TRAVERSABLE_OVERLAY_COLOR, edgecolor="none", alpha=0.60))

    if show_labels and class_names is not None:
        for (x, y), label, text_color in build_class_annotations(
            grid_handler.mask_instances,
            grid_handler.obstacle_heights,
            class_names,
        ):
            ax.text(
                x,
                y,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.35, "pad": 0.8},
            )

    if path:
        xs = [point[0] + 0.5 for point in path]
        ys = [point[1] + 0.5 for point in path]
        ax.plot(xs, ys, linewidth=3, color=PLANNER_PATH_COLOR, label="D*Lite Path")
    else:
        ax.plot([], [], linewidth=3, color=PLANNER_PATH_COLOR, label="D*Lite Path")

    if 0 <= start[0] < grid_w and 0 <= start[1] < grid_h:
        ax.plot([start[0] + 0.5], [start[1] + 0.5], "o", markersize=8, color=PLANNER_START_COLOR, label="Start")
    else:
        ax.plot([], [], "o", markersize=8, color=PLANNER_START_COLOR, label="Start")

    if 0 <= goal[0] < grid_w and 0 <= goal[1] < grid_h:
        ax.plot([goal[0] + 0.5], [goal[1] + 0.5], "o", markersize=8, color=PLANNER_GOAL_COLOR, label="Goal")
    else:
        ax.plot([], [], "o", markersize=8, color=PLANNER_GOAL_COLOR, label="Goal")
    ax.legend(loc="lower left")

    info_text = build_class_info_lines(
        grid_handler.obstacle_class_ids,
        grid_handler.obstacle_heights,
        class_names or {-1: "manual"},
    )
    ax.text(
        1.02,
        0.98,
        info_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": INFO_PANEL_EDGE_COLOR, "alpha": 0.9},
    )

    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape((height, width, 4))
    result = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    if result.shape[0] != canvas_h or result.shape[1] != canvas_w:
        result = cv2.resize(result, (canvas_w, canvas_h), interpolation=cv2.INTER_AREA)
    fig.clear()
    return result


def draw_obstacles(
    canvas_shape: tuple[int, int],
    grid_handler: GridMapHandler,
    grid_scale: int,
    class_names: dict[int, str] | None = None,
    show_labels: bool = True,
) -> np.ndarray:
    empty_path: list[tuple[int, int]] = []
    hidden_marker = (-1, -1)
    return render_plan_view(
        canvas_shape,
        grid_handler,
        empty_path,
        hidden_marker,
        hidden_marker,
        grid_scale,
        class_names=class_names,
        show_labels=show_labels,
    )


def draw_plan_overlay(
    canvas_shape: tuple[int, int],
    grid_handler: GridMapHandler,
    path: list[tuple[int, int]],
    start: tuple[int, int],
    goal: tuple[int, int],
    grid_scale: int,
    class_names: dict[int, str] | None = None,
    show_labels: bool = True,
) -> np.ndarray:
    return render_plan_view(
        canvas_shape,
        grid_handler,
        path,
        start,
        goal,
        grid_scale,
        class_names=class_names,
        show_labels=show_labels,
    )


def build_plan_result(frame_shape: tuple[int, int], mask_entries: list[list], grid_scale: int) -> dict[str, object]:
    frame_h, frame_w = frame_shape
    grid_w = max(1, frame_w // grid_scale)
    grid_h = max(1, frame_h // grid_scale)

    grid_handler = GridMapHandler(grid_w=grid_w, grid_h=grid_h, grid_scale=grid_scale)
    obs, target_point = grid_handler.batch_masks_to_obs(mask_entries)

    start = (grid_w // 2, grid_h // 2)
    goal = target_point if target_point is not None else (max(0, grid_w - 5), max(0, grid_h - 5))

    planner = DStarLite(
        start,
        goal,
        obs,
        grid_w,
        grid_h,
        passable_obs=grid_handler.traversable_obstacles,
        terrain_penalties=grid_handler.terrain_penalties,
    )

    try:
        path = planner.plan()
    except Exception:
        path = []

    return {
        "grid_handler": grid_handler,
        "path": path,
        "start": start,
        "goal": goal,
        "canvas_shape": frame_shape,
    }


def plan_frame(
    frame_shape: tuple[int, int],
    mask_entries: list[list],
    grid_scale: int,
    class_names: dict[int, str] | None = None,
) -> np.ndarray:
    plan_result = build_plan_result(frame_shape, mask_entries, grid_scale)
    return draw_plan_overlay(
        frame_shape,
        plan_result["grid_handler"],
        plan_result["path"],
        plan_result["start"],
        plan_result["goal"],
        grid_scale,
        class_names=class_names,
        show_labels=True,
    )


def process_image_file(
    image_path: Path,
    mask_groups: dict[str, list[list]],
    output_dir: Path,
    grid_scale: int,
    class_names: dict[int, str],
) -> None:
    mask_entries = mask_groups.get(image_path.stem, [])
    canvas_shape = get_mask_canvas_shape(mask_entries)
    if canvas_shape is None:
        print(f"跳过未找到有效 masks 的图片: {image_path}")
        return

    result = plan_frame(canvas_shape, mask_entries, grid_scale, class_names=class_names)
    save_path = output_dir / f"{image_path.stem}_planned.png"
    cv2.imwrite(str(save_path), result)
    print(f"已保存图片规划结果: {save_path}")


def process_video_file(
    video_path: Path,
    mask_groups: dict[str, list[list]],
    output_dir: Path,
    grid_scale: int,
    class_names: dict[int, str],
) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"跳过无法读取的视频: {video_path}")
        return

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    fallback_shape = (
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
    )
    canvas_shape = get_video_canvas_shape(video_path, mask_groups, fallback_shape)
    save_path = output_dir / f"{video_path.stem}_planned.mp4"
    writer = cv2.VideoWriter(
        str(save_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (canvas_shape[1], canvas_shape[0]),
    )

    frame_index = 0
    try:
        while True:
            ok, _frame = capture.read()
            if not ok:
                break

            frame_index += 1
            frame_stem = get_frame_stem(video_path, frame_index)
            result = plan_frame(canvas_shape, mask_groups.get(frame_stem, []), grid_scale, class_names=class_names)
            writer.write(result)
    finally:
        capture.release()
        writer.release()

    print(f"已保存视频规划结果: {save_path}")


def run_batch_pathplan(
    source: str | Path | None = None,
    weights: str | Path | None = None,
    data_yaml: str | Path | None = None,
    project: str | Path | None = None,
    device: str | None = None,
    conf_thres: float | None = None,
    grid_scale: int | None = None,
) -> Path:
    source_path = resolve_path(source, get_default_source_dir())
    weights_path = resolve_path(weights, get_default_weights())
    data_yaml_path = resolve_path(data_yaml, get_default_data_yaml())
    project_path = resolve_path(project, get_default_pathplan_project_dir())
    segmentation_project_path = get_default_project_dir()
    media_files = iter_source_media(source_path)
    run_dir = create_pathplan_run_dir(project_path)

    print(f"路径规划输出目录: {run_dir}")
    segmentation_run_dir = get_next_segmentation_run_dir(segmentation_project_path)
    print(f"分割输出目录: {segmentation_run_dir}")
    result = run_prediction(
        weights=weights_path,
        data_yaml=data_yaml_path,
        source=source_path,
        project=segmentation_project_path,
        name="exp",
        device=device,
        conf_thres=conf_thres,
    )
    if result.returncode != 0:
        raise SystemExit(result.returncode)

    mask_dir = segmentation_run_dir / "masks"
    mask_groups = load_grouped_mask_entries(mask_dir)
    current_grid_scale = grid_scale if grid_scale is not None else DEFAULT_CONFIG.default_grid_scale
    class_names = load_class_names(data_yaml_path)

    for media_path in media_files:
        if is_image_file(media_path):
            process_image_file(media_path, mask_groups, run_dir, current_grid_scale, class_names)
        else:
            process_video_file(media_path, mask_groups, run_dir, current_grid_scale, class_names)

    return run_dir


def parse_args():
    parser = argparse.ArgumentParser(description="对图片/视频批量执行分割后路径规划，并保存结果。")
    parser.add_argument("--source", type=Path, default=get_default_source_dir(), help="输入图片、视频或目录")
    parser.add_argument("--weights", type=Path, default=get_default_weights(), help="模型权重")
    parser.add_argument("--data", type=Path, default=get_default_data_yaml(), help="数据配置 yaml")
    parser.add_argument("--project", type=Path, default=get_default_pathplan_project_dir(), help="路径规划输出根目录")
    parser.add_argument("--device", default=DEFAULT_CONFIG.default_device, help="推理设备")
    parser.add_argument("--conf-thres", type=float, default=DEFAULT_CONFIG.default_conf_thres, help="置信度阈值")
    parser.add_argument("--grid-scale", type=int, default=DEFAULT_CONFIG.default_grid_scale, help="栅格缩放")
    return parser.parse_args()


def main():
    args = parse_args()
    run_batch_pathplan(
        source=args.source,
        weights=args.weights,
        data_yaml=args.data,
        project=args.project,
        device=args.device,
        conf_thres=args.conf_thres,
        grid_scale=args.grid_scale,
    )


if __name__ == "__main__":
    main()
