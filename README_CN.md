# Yosegment2Rk3588

基于 YOLOv5 分割结果的栅格化与 D* Lite 路径规划工程，支持三条工作流：

- 标准链路：`PT 分割 -> masks 落盘 -> 路径规划`
- 开发链路：`ONNX 分割 -> 内存 mask_entries -> 路径规划`
- 部署链路：`RKNN 分割 -> 内存 mask_entries -> 路径规划`

当前核心业务代码都在 `app/` 目录。

## 1. 目录结构

```text
.
├─app/
│  ├─inference/         # 分割入口（PT/ONNX/RKNN）
│  ├─mapping/           # mask -> 栅格障碍物
│  └─planning/          # D* Lite 规划与渲染
├─data/
│  └─my.yaml            # 类别配置
├─tools/
│  └─export_rknn.py     # ONNX -> RKNN 导出脚本
├─weights/
│  ├─0414_qy++.pt
│  ├─0414_qy++.onnx
│  └─0414_qy++.rknn     # 导出后生成
├─yolo/                 # vendored YOLOv5 代码
├─requirements.txt
├─requirements-rk3588.txt
└─README_CN.md
```

## 2. 三条主链路

### 2.1 标准落盘链路（PT）

```text
source
  -> app/inference/segmentation.py
  -> runs/segment/exp*/masks
  -> app/mapping/grid_map.py
  -> app/planning/path_planner.py 或 pathplan_batch.py
  -> runs/pathplan/exp*
```

适合离线处理、批处理、复盘调试。

### 2.2 ONNX 实时链路（开发机验证）

```text
frame/video/stream
  -> app/inference/onnx_realtime.py
  -> 内存 mask_entries
  -> app/planning/pathplan_batch.py::build_plan_result/render_plan_view
  -> app/planning/realtime_pathplan.py
```

适合在 x86 开发机上快速验证分割、路径规划和可视化是否一致。

### 2.3 RKNN 实时链路（RK3588 部署）

```text
camera/video/rtsp
  -> app/inference/rknn_realtime.py
  -> 内存 mask_entries
  -> app/planning/pathplan_batch.py::build_plan_result/render_plan_view
  -> app/planning/realtime_pathplan.py
```

这是当前仓库最接近最终落地的部署形态：只导出一个 `.rknn` 分割模型，后处理、栅格化、路径规划和可视化仍在 CPU 侧 Python 中执行。

## 3. 环境准备

### 3.1 通用开发环境（x86）

```bash
python -m venv .venv
source .venv/bin/activate  # Windows 请用 .venv\Scripts\activate
python -m pip install -U pip
pip install -r requirements.txt
```

### 3.2 RK3588 运行环境（板端）

`requirements-rk3588.txt` 用于板端运行时依赖。

建议：

```bash
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3-pip ffmpeg libgl1 libglib2.0-0

python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
pip install -r requirements-rk3588.txt
```

然后额外安装板端 RKNN 运行时：

```bash
pip install rknn-toolkit-lite2
```

如果需要本地窗口显示（`--view`），把 `opencv-python-headless` 换成 `opencv-python`。

### 3.3 RKNN 导出环境（通常不是板端）

`.onnx -> .rknn` 一般在装有 `rknn-toolkit2` 的 Linux x86 环境完成，不建议直接在 RK3588 板端导出。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt
pip install rknn-toolkit2
```

## 4. 快速开始

说明：为了避免默认路径不匹配，建议始终显式传 `--source`、`--weights`、`--data`。

### 4.1 仅做分割（PT）

```bash
python app/inference/segmentation.py \
  --source ./test_input \
  --weights ./weights/0414_qy++.pt \
  --data ./data/my.yaml \
  --project ./runs/segment \
  --device cpu \
  --load-masks
```

输出目录：`runs/segment/exp*`

### 4.2 交互式路径规划（读取已有 masks）

```bash
python app/planning/path_planner.py \
  --mask-dir ./runs/segment/exp/masks \
  --project ./runs/pathplan \
  --data ./data/my.yaml \
  --grid-scale 10
```

输出目录：`runs/pathplan/exp*`

### 4.3 批处理（先分割再规划）

```bash
python app/planning/pathplan_batch.py \
  --source ./test_input \
  --weights ./weights/0414_qy++.pt \
  --data ./data/my.yaml \
  --project ./runs/pathplan \
  --device cpu \
  --grid-scale 10
```

### 4.4 ONNX 实时路径规划（开发机回归）

单图/目录/视频：

```bash
python app/planning/realtime_pathplan.py \
  --source ./test_input/demo.mp4 \
  --weights ./weights/0414_qy++.onnx \
  --backend onnx \
  --data ./data/my.yaml \
  --device cpu \
  --grid-scale 10
```

摄像头：

```bash
python app/planning/realtime_pathplan.py \
  --source 0 \
  --weights ./weights/0414_qy++.onnx \
  --backend onnx \
  --data ./data/my.yaml \
  --device cpu \
  --view
```

### 4.5 RKNN 实时路径规划（RK3588 部署）

先准备 `weights/0414_qy++.rknn`，然后运行：

```bash
python app/planning/realtime_pathplan.py \
  --source 0 \
  --weights ./weights/0414_qy++.rknn \
  --backend rknn \
  --data ./data/my.yaml \
  --view
```

RTSP 流：

```bash
python app/planning/realtime_pathplan.py \
  --source rtsp://xxx \
  --weights ./weights/0414_qy++.rknn \
  --backend rknn \
  --data ./data/my.yaml \
  --view
```

也可以用 `--backend auto`，按权重后缀自动选择 ONNX 或 RKNN。

只显示不保存：追加 `--nosave`

## 5. 模型导出

### 5.1 PT 转 ONNX

```bash
python yolo/export.py --weights ./weights/0414_qy++.pt --include onnx --imgsz 640 640
```

生成后可直接给 `realtime_pathplan.py --weights xxx.onnx` 使用。

### 5.2 ONNX 转 RKNN

不量化导出：

```bash
python tools/export_rknn.py \
  --onnx ./weights/0414_qy++.onnx \
  --output ./weights/0414_qy++.rknn \
  --target rk3588 \
  --opset-check
```

INT8 量化导出：

```bash
python tools/export_rknn.py \
  --onnx ./weights/0414_qy++.onnx \
  --output ./weights/0414_qy++.rknn \
  --target rk3588 \
  --quantize \
  --dataset ./data/rknn_dataset.txt
```

其中 `rknn_dataset.txt` 每行一个图片路径，建议使用接近摄像头实际场景的样本。

## 6. 关键配置约定

### 6.1 类别语义（`app/mapping/grid_map.py`）

- 目标点类别：`0`（`TARGET_CLASS`）
- 可通行但有代价类别：`4, 6`（`TRAVERSABLE_CLASSES`）
- 障碍物高度：`CLASS_HEIGHTS` 字典

### 6.2 默认路径覆盖（`app/paths.py`）

可通过环境变量覆盖默认路径：

- `YOSEGMENT_YOLO_DIR`
- `YOSEGMENT_DATA_YAML`
- `YOSEGMENT_RUNS_DIR`
- `YOSEGMENT_TEST_INPUT`
- `YOSEGMENT_WEIGHTS_DIR`

## 7. RK3588 部署注意事项

1. `realtime_pathplan.py` 现在支持 `--backend auto|onnx|rknn`。
2. 真正需要导出成 `.rknn` 的只有分割模型，不是 `realtime_pathplan.py` 本身。
3. 推荐最终交付物是一个 `.rknn` 文件加现有 `app/` Python 代码。
4. 若仅后台运行（无图形界面），不要加 `--view`，并使用 `opencv-python-headless`。
5. 若要调试精度，先在开发机上用同一份 `.onnx` 跑通 ONNX 链路，再切到 `.rknn` 对比。
6. 板端运行需要 `rknn-toolkit-lite2`；导出 `.rknn` 需要 `rknn-toolkit2`。

## 8. 常见问题

### 8.1 `ModuleNotFoundError: No module named 'cv2'`

未安装 OpenCV：

```bash
pip install opencv-python-headless
```

### 8.2 `ONNX realtime 推理需要先安装 onnxruntime`

```bash
pip install onnxruntime
```

### 8.3 `RKNN realtime 推理需要先安装 rknn-toolkit2 或板端 rknn-toolkit-lite2`

板端运行：

```bash
pip install rknn-toolkit-lite2
```

导出模型：

```bash
pip install rknn-toolkit2
```

### 8.4 导出时报量化数据集错误

确认 `--dataset` 指向一个存在的 txt 文件，且文件内每行都是导出环境可访问的图片绝对路径或相对路径。

---

建议先走一遍：`PT -> ONNX -> RKNN -> realtime_pathplan.py --backend rknn`。这样最容易定位是模型导出问题，还是板端运行时问题。
