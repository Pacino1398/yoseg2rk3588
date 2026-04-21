from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import DEFAULT_CONFIG
from app.inference.onnx_realtime import (
    OnnxRealtimeSegmenter,
    get_default_data_yaml,
    get_default_realtime_weights,
)
from app.inference.rknn_realtime import RknnRealtimeSegmenter
from app.inference.segmentation import get_default_source_dir
from app.paths import resolve_path
from app.planning.pathplan_batch import (
    build_plan_result,
    create_pathplan_run_dir,
    get_default_pathplan_project_dir,
    get_frame_stem,
    is_image_file,
    is_video_file,
    iter_source_media,
    load_class_names,
    render_plan_view,
)

STREAM_SCHEMES = ("rtsp://", "rtmp://", "http://", "https://")
WINDOW_NAME = "realtime_pathplan"


def is_stream_source(source: str) -> bool:
    return source.isdigit() or source.lower().startswith(STREAM_SCHEMES)


def resolve_source(source: str | Path | None) -> str | Path:
    if source is None:
        return get_default_source_dir()
    if isinstance(source, Path):
        return resolve_path(source, get_default_source_dir())
    if is_stream_source(source):
        return source
    return resolve_path(source, get_default_source_dir())


def get_source_stem(source: str | Path) -> str:
    if isinstance(source, Path):
        return source.stem

    if source.isdigit():
        return f"camera{source}"

    if source.lower().startswith(STREAM_SCHEMES):
        parsed = urlparse(source)
        stream_stem = Path(parsed.path).stem
        return stream_stem or "stream"

    return Path(source).stem


def resolve_backend(backend: str, weights: str | Path | None) -> str:
    if backend != "auto":
        return backend
    if weights is None:
        default_weights = get_default_realtime_weights()
        return "rknn" if default_weights.suffix.lower() == ".rknn" else "onnx"
    suffix = Path(weights).suffix.lower()
    if suffix == ".rknn":
        return "rknn"
    if suffix == ".onnx":
        return "onnx"
    raise ValueError(f"无法根据权重后缀自动判断 realtime 后端: {weights}")


def create_segmenter(
    backend: str,
    weights: str | Path | None,
    data_yaml: str | Path | None,
    device: str | None,
    imgsz: int | tuple[int, int],
    conf_thres: float | None,
    iou_thres: float,
    dnn: bool,
    half: bool,
):
    if backend == "rknn":
        return RknnRealtimeSegmenter(
            weights=weights,
            data_yaml=data_yaml,
            device=device,
            imgsz=imgsz,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            dnn=dnn,
            half=half,
        )
    return OnnxRealtimeSegmenter(
        weights=weights,
        data_yaml=data_yaml,
        device=device,
        imgsz=imgsz,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        dnn=dnn,
        half=half,
    )


def render_planned_frame(
    frame: np.ndarray,
    segmenter,
    class_names: dict[int, str],
    grid_scale: int,
    frame_stem: str,
) -> np.ndarray:
    mask_entries = segmenter.predict_frame(frame, frame_stem)
    plan_result = build_plan_result(frame.shape[:2], mask_entries, grid_scale)
    return render_plan_view(
        plan_result["canvas_shape"],
        plan_result["grid_handler"],
        plan_result["path"],
        plan_result["start"],
        plan_result["goal"],
        grid_scale,
        class_names=class_names,
        show_labels=True,
    )


def maybe_show_frame(frame: np.ndarray, enabled: bool) -> bool:
    if not enabled:
        return False

    cv2.imshow(WINDOW_NAME, frame)
    key = cv2.waitKey(1) & 0xFF
    return key in {27, ord("q")}


def process_image_source(
    image_path: Path,
    segmenter,
    class_names: dict[int, str],
    grid_scale: int,
    run_dir: Path | None,
    view: bool,
) -> Path | None:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")

    planned = render_planned_frame(frame, segmenter, class_names, grid_scale, image_path.stem)
    output_path = None
    if run_dir is not None:
        output_path = run_dir / f"{image_path.stem}_planned.png"
        cv2.imwrite(str(output_path), planned)
    if maybe_show_frame(planned, view):
        return output_path
    return output_path


def process_video_capture(
    capture: cv2.VideoCapture,
    source_name: str,
    segmenter,
    class_names: dict[int, str],
    grid_scale: int,
    run_dir: Path | None,
    view: bool,
    fps: float | None = None,
) -> Path | None:
    ok, frame = capture.read()
    if not ok or frame is None:
        raise RuntimeError(f"无法读取视频流: {source_name}")

    frame_h, frame_w = frame.shape[:2]
    current_fps = fps if fps is not None else capture.get(cv2.CAP_PROP_FPS)
    if not current_fps or current_fps <= 0:
        current_fps = 30.0

    output_path = run_dir / f"{source_name}_planned.mp4" if run_dir is not None else None
    writer = None
    if output_path is not None:
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            current_fps,
            (frame_w, frame_h),
        )

    frame_index = 1
    try:
        while True:
            frame_stem = get_frame_stem(Path(f"{source_name}.mp4"), frame_index)
            planned = render_planned_frame(frame, segmenter, class_names, grid_scale, frame_stem)
            if writer is not None:
                writer.write(planned)
            if maybe_show_frame(planned, view):
                break

            ok, frame = capture.read()
            if not ok or frame is None:
                break
            frame_index += 1
    finally:
        if writer is not None:
            writer.release()

    return output_path


def process_video_source(
    video_path: Path,
    segmenter,
    class_names: dict[int, str],
    grid_scale: int,
    run_dir: Path | None,
    view: bool,
) -> Path | None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    try:
        return process_video_capture(
            capture,
            video_path.stem,
            segmenter,
            class_names,
            grid_scale,
            run_dir,
            view,
        )
    finally:
        capture.release()


def process_stream_source(
    stream_source: str,
    segmenter,
    class_names: dict[int, str],
    grid_scale: int,
    run_dir: Path | None,
    view: bool,
) -> Path | None:
    capture_target: int | str = int(stream_source) if stream_source.isdigit() else stream_source
    capture = cv2.VideoCapture(capture_target)
    if not capture.isOpened():
        raise RuntimeError(f"无法打开流: {stream_source}")

    try:
        return process_video_capture(
            capture,
            get_source_stem(stream_source),
            segmenter,
            class_names,
            grid_scale,
            run_dir,
            view,
        )
    finally:
        capture.release()


def run_realtime_pathplan(
    source: str | Path | None = None,
    weights: str | Path | None = None,
    data_yaml: str | Path | None = None,
    project: str | Path | None = None,
    device: str | None = None,
    conf_thres: float | None = None,
    iou_thres: float = 0.45,
    imgsz: int | tuple[int, int] = 640,
    grid_scale: int | None = None,
    view: bool = False,
    save: bool = True,
    dnn: bool = False,
    half: bool = False,
    backend: str = "auto",
) -> Path | None:
    source_value = resolve_source(source)
    current_grid_scale = grid_scale if grid_scale is not None else DEFAULT_CONFIG.default_grid_scale
    selected_backend = resolve_backend(backend, weights)
    selected_weights = weights if weights is not None else get_default_realtime_weights()
    run_dir = None
    if save:
        project_path = resolve_path(project, get_default_pathplan_project_dir())
        run_dir = create_pathplan_run_dir(project_path)
        print(f"路径规划输出目录: {run_dir}")

    segmenter = create_segmenter(
        selected_backend,
        selected_weights,
        data_yaml,
        device,
        imgsz,
        conf_thres,
        iou_thres,
        dnn,
        half,
    )
    print(f"实时推理后端: {selected_backend} | 权重: {selected_weights}")
    class_names = load_class_names(resolve_path(data_yaml, get_default_data_yaml()))

    try:
        if isinstance(source_value, Path):
            if source_value.is_dir():
                for media_path in iter_source_media(source_value):
                    if is_image_file(media_path):
                        output_path = process_image_source(media_path, segmenter, class_names, current_grid_scale, run_dir, view)
                    elif is_video_file(media_path):
                        output_path = process_video_source(media_path, segmenter, class_names, current_grid_scale, run_dir, view)
                    else:
                        continue
                    if output_path is not None:
                        print(f"已保存规划结果: {output_path}")
                return run_dir

            if is_image_file(source_value):
                output_path = process_image_source(source_value, segmenter, class_names, current_grid_scale, run_dir, view)
                if output_path is not None:
                    print(f"已保存规划结果: {output_path}")
                return run_dir if run_dir is not None else output_path

            output_path = process_video_source(source_value, segmenter, class_names, current_grid_scale, run_dir, view)
            if output_path is not None:
                print(f"已保存规划结果: {output_path}")
            return run_dir if run_dir is not None else output_path

        output_path = process_stream_source(source_value, segmenter, class_names, current_grid_scale, run_dir, view)
        if output_path is not None:
            print(f"已保存规划结果: {output_path}")
        return run_dir if run_dir is not None else output_path
    finally:
        close = getattr(segmenter, "close", None)
        if callable(close):
            close()
        if view:
            cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="使用 ONNX 或 RKNN 分割结果直接做实时路径规划，不经过 mask 落盘中转。")
    parser.add_argument("--source", default=str(get_default_source_dir()), help="输入图片、视频、目录、摄像头索引或流地址")
    parser.add_argument("--weights", default=str(get_default_realtime_weights()), help="实时分割权重路径，支持 .onnx 或 .rknn")
    parser.add_argument("--backend", choices=("auto", "onnx", "rknn"), default="auto", help="实时推理后端")
    parser.add_argument("--data", default=str(get_default_data_yaml()), help="数据配置 yaml")
    parser.add_argument("--project", type=Path, default=get_default_pathplan_project_dir(), help="路径规划输出根目录")
    parser.add_argument("--device", default=DEFAULT_CONFIG.default_device, help="推理设备")
    parser.add_argument("--conf-thres", type=float, default=DEFAULT_CONFIG.default_conf_thres, help="置信度阈值")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU 阈值")
    parser.add_argument("--imgsz", nargs="+", type=int, default=[640], help="推理尺寸，支持 --imgsz 640 或 --imgsz 640 640")
    parser.add_argument("--grid-scale", type=int, default=DEFAULT_CONFIG.default_grid_scale, help="栅格缩放")
    parser.add_argument("--view", action="store_true", help="实时显示规划画面，按 q 或 Esc 退出")
    parser.add_argument("--nosave", action="store_true", help="只显示不保存输出")
    parser.add_argument("--dnn", action="store_true", help="使用 OpenCV DNN 加载 ONNX")
    parser.add_argument("--half", action="store_true", help="启用 FP16")
    return parser.parse_args()


def normalize_imgsz(values: list[int]) -> int | tuple[int, int]:
    if len(values) == 1:
        return values[0]
    return values[0], values[1]


def main():
    args = parse_args()
    run_realtime_pathplan(
        source=args.source,
        weights=args.weights,
        data_yaml=args.data,
        project=args.project,
        device=args.device,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        imgsz=normalize_imgsz(args.imgsz),
        grid_scale=args.grid_scale,
        view=args.view,
        save=not args.nosave,
        dnn=args.dnn,
        half=args.half,
        backend=args.backend,
    )


if __name__ == "__main__":
    main()
