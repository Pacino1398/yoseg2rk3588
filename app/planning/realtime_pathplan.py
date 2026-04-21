from __future__ import annotations

import argparse
import sys
import threading
import time
from http import server
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np

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
    render_plan_on_frame,
)

STREAM_SCHEMES = ("rtsp://", "rtmp://", "http://", "https://")
WINDOW_NAME = "realtime_pathplan"
DEFAULT_REMOTE_PATH = "/stream.mjpg"


class MjpegFrameStore:
    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.payload: bytes | None = None
        self.sequence = 0

    def update(self, frame: np.ndarray) -> None:
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            return
        payload = encoded.tobytes()
        with self.condition:
            self.payload = payload
            self.sequence += 1
            self.condition.notify_all()

    def wait_for_frame(self, last_sequence: int, timeout: float = 1.0) -> tuple[int, bytes | None]:
        with self.condition:
            if self.sequence == last_sequence:
                self.condition.wait(timeout=timeout)
            return self.sequence, self.payload


class ThreadingMjpegServer(server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address: tuple[str, int], handler_cls, frame_store: MjpegFrameStore, stream_path: str):
        super().__init__(server_address, handler_cls)
        self.frame_store = frame_store
        self.stream_path = stream_path


class MjpegRequestHandler(server.BaseHTTPRequestHandler):
    server: ThreadingMjpegServer

    def do_GET(self) -> None:
        if self.path not in {self.server.stream_path, "/"}:
            self.send_error(404)
            return

        if self.path == "/":
            body = (
                f"<html><body><img src=\"{self.server.stream_path}\" "
                f"style=\"max-width:100%;height:auto;\"></body></html>"
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(200)
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        sequence = 0
        try:
            while True:
                sequence, payload = self.server.frame_store.wait_for_frame(sequence)
                if payload is None:
                    continue
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii"))
                self.wfile.write(payload)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            return

    def log_message(self, format: str, *args) -> None:
        return


class MjpegStreamServer:
    def __init__(self, host: str, port: int, stream_path: str):
        self.frame_store = MjpegFrameStore()
        self.server = ThreadingMjpegServer((host, port), MjpegRequestHandler, self.frame_store, stream_path)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.host = host
        self.port = port
        self.stream_path = stream_path

    def start(self) -> None:
        self.thread.start()

    def update_frame(self, frame: np.ndarray) -> None:
        self.frame_store.update(frame)

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=1.0)


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


def normalize_display_mode(display: str | None, view: bool) -> str:
    if display is not None:
        return display
    return "local" if view else "none"


def should_show_local(display: str) -> bool:
    return display in {"local", "both"}


def should_stream_remote(display: str) -> bool:
    return display in {"remote", "both"}


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
    return render_plan_on_frame(
        frame,
        plan_result["grid_handler"],
        plan_result["path"],
        plan_result["start"],
        plan_result["goal"],
        grid_scale,
        class_names=class_names,
        show_labels=True,
    )


def maybe_show_frame(frame: np.ndarray, enabled: bool, imshow_state: dict[str, bool] | None = None) -> bool:
    if not enabled:
        return False

    try:
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        return key in {27, ord("q")}
    except cv2.error as exc:
        if imshow_state is not None and not imshow_state.get("warned", False):
            print(f"本机显示不可用，已自动关闭 local 显示: {exc}")
            imshow_state["warned"] = True
        if imshow_state is not None:
            imshow_state["enabled"] = False
        return False


def process_image_source(
    image_path: Path,
    segmenter,
    class_names: dict[int, str],
    grid_scale: int,
    run_dir: Path | None,
    show_local: bool,
    imshow_state: dict[str, bool],
    remote_server: MjpegStreamServer | None,
) -> Path | None:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")

    planned = render_planned_frame(frame, segmenter, class_names, grid_scale, image_path.stem)
    if remote_server is not None:
        remote_server.update_frame(planned)
    output_path = None
    if run_dir is not None:
        output_path = run_dir / f"{image_path.stem}_planned.png"
        cv2.imwrite(str(output_path), planned)
    if maybe_show_frame(planned, show_local and imshow_state["enabled"], imshow_state):
        return output_path
    return output_path


def process_video_capture(
    capture: cv2.VideoCapture,
    source_name: str,
    segmenter,
    class_names: dict[int, str],
    grid_scale: int,
    run_dir: Path | None,
    show_local: bool,
    imshow_state: dict[str, bool],
    remote_server: MjpegStreamServer | None,
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
            if remote_server is not None:
                remote_server.update_frame(planned)
            if writer is not None:
                writer.write(planned)
            if maybe_show_frame(planned, show_local and imshow_state["enabled"], imshow_state):
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
    show_local: bool,
    imshow_state: dict[str, bool],
    remote_server: MjpegStreamServer | None,
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
            show_local,
            imshow_state,
            remote_server,
        )
    finally:
        capture.release()


def process_stream_source(
    stream_source: str,
    segmenter,
    class_names: dict[int, str],
    grid_scale: int,
    run_dir: Path | None,
    show_local: bool,
    imshow_state: dict[str, bool],
    remote_server: MjpegStreamServer | None,
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
            show_local,
            imshow_state,
            remote_server,
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
    display: str | None = None,
    remote_host: str = "0.0.0.0",
    remote_port: int = 8080,
    remote_path: str = DEFAULT_REMOTE_PATH,
) -> Path | None:
    source_value = resolve_source(source)
    current_grid_scale = grid_scale if grid_scale is not None else DEFAULT_CONFIG.default_grid_scale
    selected_backend = resolve_backend(backend, weights)
    selected_weights = weights if weights is not None else get_default_realtime_weights()
    display_mode = normalize_display_mode(display, view)
    show_local = should_show_local(display_mode)
    enable_remote = should_stream_remote(display_mode)
    normalized_remote_path = remote_path if remote_path.startswith("/") else f"/{remote_path}"
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
    imshow_state = {"enabled": show_local, "warned": False}
    remote_server = MjpegStreamServer(remote_host, remote_port, normalized_remote_path) if enable_remote else None
    if remote_server is not None:
        remote_server.start()
        print(f"MJPEG 预览地址: http://{remote_host if remote_host != '0.0.0.0' else '127.0.0.1'}:{remote_port}{normalized_remote_path}")

    try:
        if isinstance(source_value, Path):
            if source_value.is_dir():
                for media_path in iter_source_media(source_value):
                    if is_image_file(media_path):
                        output_path = process_image_source(
                            media_path,
                            segmenter,
                            class_names,
                            current_grid_scale,
                            run_dir,
                            show_local,
                            imshow_state,
                            remote_server,
                        )
                    elif is_video_file(media_path):
                        output_path = process_video_source(
                            media_path,
                            segmenter,
                            class_names,
                            current_grid_scale,
                            run_dir,
                            show_local,
                            imshow_state,
                            remote_server,
                        )
                    else:
                        continue
                    if output_path is not None:
                        print(f"已保存规划结果: {output_path}")
                return run_dir

            if is_image_file(source_value):
                output_path = process_image_source(
                    source_value,
                    segmenter,
                    class_names,
                    current_grid_scale,
                    run_dir,
                    show_local,
                    imshow_state,
                    remote_server,
                )
                if output_path is not None:
                    print(f"已保存规划结果: {output_path}")
                return run_dir if run_dir is not None else output_path

            output_path = process_video_source(
                source_value,
                segmenter,
                class_names,
                current_grid_scale,
                run_dir,
                show_local,
                imshow_state,
                remote_server,
            )
            if output_path is not None:
                print(f"已保存规划结果: {output_path}")
            return run_dir if run_dir is not None else output_path

        output_path = process_stream_source(
            source_value,
            segmenter,
            class_names,
            current_grid_scale,
            run_dir,
            show_local,
            imshow_state,
            remote_server,
        )
        if output_path is not None:
            print(f"已保存规划结果: {output_path}")
        return run_dir if run_dir is not None else output_path
    finally:
        close = getattr(segmenter, "close", None)
        if callable(close):
            close()
        if remote_server is not None:
            remote_server.close()
        if imshow_state["enabled"]:
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
    parser.add_argument("--display", choices=("local", "remote", "both", "none"), default=None, help="显示目标：本机窗口、上位机 MJPEG、同时显示或都不显示")
    parser.add_argument("--remote-host", default="0.0.0.0", help="MJPEG 服务绑定地址")
    parser.add_argument("--remote-port", type=int, default=8080, help="MJPEG 服务端口")
    parser.add_argument("--remote-path", default=DEFAULT_REMOTE_PATH, help="MJPEG 预览路径")
    parser.add_argument("--view", action="store_true", help="兼容旧参数，等价于 --display local")
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
        display=args.display,
        remote_host=args.remote_host,
        remote_port=args.remote_port,
        remote_path=args.remote_path,
    )


if __name__ == "__main__":
    main()
