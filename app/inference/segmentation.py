from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

import cv2

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import DEFAULT_CONFIG
from app.paths import resolve_path

# RK3588 部署建议保持为 None，运行时通过命令行参数传入，或使用项目默认路径。
MANUAL_SOURCE: str | Path | None = None
MANUAL_WEIGHTS: str | Path | None = None
MANUAL_DATA_YAML: str | Path | None = None
MANUAL_PROJECT: str | Path | None = None
MANUAL_NAME: str | None = None


def _resolve_manual_path(value: str | Path | None, default: Path) -> Path:
    if value is None:
        return default

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = DEFAULT_CONFIG.repo_root / path
    return path.resolve()


def get_default_source_dir() -> Path:
    return _resolve_manual_path(MANUAL_SOURCE, DEFAULT_CONFIG.default_source)


def get_default_weights() -> Path:
    return _resolve_manual_path(MANUAL_WEIGHTS, DEFAULT_CONFIG.default_weights)


def get_default_data_yaml() -> Path:
    return _resolve_manual_path(MANUAL_DATA_YAML, DEFAULT_CONFIG.data_yaml)


def get_default_project_dir() -> Path:
    return _resolve_manual_path(MANUAL_PROJECT, DEFAULT_CONFIG.runs_dir)


def get_default_run_name() -> str:
    return MANUAL_NAME or DEFAULT_CONFIG.default_run_name


def get_next_run_name(project_dir: str | Path, base_name: str) -> str:
    project_path = Path(project_dir)
    first_run = project_path / base_name
    if not first_run.exists():
        return base_name

    index = 1
    while True:
        run_name = f"{base_name}{index}"
        if not (project_path / run_name).exists():
            return run_name
        index += 1


def resolve_run_name(project: str | Path | None = None, name: str | None = None) -> str:
    project_path = resolve_path(project, get_default_project_dir())
    base_run_name = name or get_default_run_name()
    return get_next_run_name(project_path, base_run_name)


def build_predict_command(
    weights: str | Path | None = None,
    data_yaml: str | Path | None = None,
    source: str | Path | None = None,
    project: str | Path | None = None,
    name: str | None = None,
    device: str | None = None,
    conf_thres: float | None = None,
) -> list[str]:
    weights_path = resolve_path(weights, get_default_weights())
    data_yaml_path = resolve_path(data_yaml, get_default_data_yaml())
    source_path = resolve_path(source, get_default_source_dir())
    project_path = resolve_path(project, get_default_project_dir())
    run_name = resolve_run_name(project=project_path, name=name)
    device_name = device or DEFAULT_CONFIG.default_device
    conf = conf_thres if conf_thres is not None else DEFAULT_CONFIG.default_conf_thres

    return [
        sys.executable,
        str(DEFAULT_CONFIG.repo_root / "yolo" / "segment" / "predict.py"),
        "--weights",
        str(weights_path),
        "--data",
        str(data_yaml_path),
        "--source",
        str(source_path),
        "--project",
        str(project_path),
        "--name",
        run_name,
        "--exist-ok",
        "--save-txt",
        "--save-conf",
        "--device",
        device_name,
        "--conf-thres",
        str(conf),
    ]


def run_prediction(
    weights: str | Path | None = None,
    data_yaml: str | Path | None = None,
    source: str | Path | None = None,
    project: str | Path | None = None,
    name: str | None = None,
    device: str | None = None,
    conf_thres: float | None = None,
) -> subprocess.CompletedProcess:
    command = build_predict_command(weights, data_yaml, source, project, name, device, conf_thres)
    print("--- 开始推理 ---")
    print("执行指令:", " ".join(command))
    result = subprocess.run(command, cwd=DEFAULT_CONFIG.repo_root, check=False)
    if result.returncode == 0:
        print("\n成功")
    else:
        print(f"\n推理失败，错误码: {result.returncode}")
    return result


def load_generated_masks(project: str | Path | None = None, name: str | None = None) -> List[list]:
    project_path = resolve_path(project, get_default_project_dir())
    run_name = name or get_default_run_name()
    mask_dir = project_path / run_name / "masks"
    if not mask_dir.exists():
        print("未找到 masks 文件夹")
        return []

    mask_list: List[list] = []
    for mask_file in sorted(mask_dir.glob("*.png")):
        mask = cv2.imread(str(mask_file), 0)
        if mask is not None:
            mask_list.append([None, None, None, mask])

    print(f"读取到 {len(mask_list)} 个掩码")
    return mask_list


def prediction_seg(source_path: str | Path):
    run_name = resolve_run_name()
    result = run_prediction(source=source_path, name=run_name)
    if result.returncode != 0:
        return []
    return load_generated_masks(name=run_name)


def parse_args():
    parser = argparse.ArgumentParser(description="运行 YOLOv5 分割推理并收集生成的 masks。")
    parser.add_argument("--weights", type=Path, default=get_default_weights(), help="模型权重")
    parser.add_argument("--data", type=Path, default=get_default_data_yaml(), help="数据配置 yaml")
    parser.add_argument("--source", type=Path, default=get_default_source_dir(), help="输入图片/目录")
    parser.add_argument("--project", type=Path, default=get_default_project_dir(), help="输出目录")
    parser.add_argument("--name", default=get_default_run_name(), help="实验名")
    parser.add_argument("--device", default=DEFAULT_CONFIG.default_device, help="推理设备")
    parser.add_argument("--conf-thres", type=float, default=DEFAULT_CONFIG.default_conf_thres, help="置信度阈值")
    parser.add_argument("--load-masks", action="store_true", help="推理结束后回读生成的 masks")
    return parser.parse_args()


def main():
    args = parse_args()
    run_name = resolve_run_name(project=args.project, name=args.name)
    result = run_prediction(
        weights=args.weights,
        data_yaml=args.data,
        source=args.source,
        project=args.project,
        name=run_name,
        device=args.device,
        conf_thres=args.conf_thres,
    )
    if args.load_masks and result.returncode == 0:
        load_generated_masks(project=args.project, name=run_name)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
