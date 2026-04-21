# Yosegment2Rk3588

基于 YOLOv5 分割结果的栅格化与 D* Lite 路径规划工程，支持 PT、ONNX、RKNN 三条链路。

当前实时链路已经改为：**直接在原始相机/视频画面上叠加障碍物、目标点和规划路径**，不再使用 Matplotlib 风格规划图做实时显示。

后续在修改统一在gitee：
https://gitee.com/Songqy1398/yoseg2rk3588.git

## 功能概览

- PT 离线分割与落盘
- ONNX 实时分割 + 路径规划
- RKNN 实时分割 + 路径规划（RK3588 / 香橙派）
- 原始视频帧上的实时规划叠加显示
- 上位机 HTTP/MJPEG 实时预览
- 本机窗口 / 上位机 / 双显示 / 关闭显示 四种模式

## 目录结构

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
│  └─0414_qy++.rknn
├─yolo/                 # vendored YOLOv5 代码
├─requirements.txt
├─requirements-rk3588.txt
├─README.md
└─README_CN.md
```

## 三条主链路

### 1. PT 标准落盘链路

```text
source
  -> app/inference/segmentation.py
  -> runs/segment/exp*/masks
  -> app/mapping/grid_map.py
  -> app/planning/path_planner.py / pathplan_batch.py
  -> runs/pathplan/exp*
```

适合离线处理、批处理、复盘调试。

### 2. ONNX 实时链路

```text
frame/video/stream
  -> app/inference/onnx_realtime.py
  -> 内存 mask_entries
  -> app/planning/pathplan_batch.py::build_plan_result/render_plan_on_frame
  -> app/planning/realtime_pathplan.py
```

### 3. RKNN 实时链路

```text
camera/video/rtsp
  -> app/inference/rknn_realtime.py
  -> 内存 mask_entries
  -> app/planning/pathplan_batch.py::build_plan_result/render_plan_on_frame
  -> app/planning/realtime_pathplan.py
```

这是 RK3588 / 香橙派上的主要部署形态：`.rknn` 负责分割推理，后处理、栅格化、D* Lite 路径规划和叠加渲染仍由 Python 在 CPU 侧完成。

## 环境准备

### x86 开发环境

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

### RK3588 / 香橙派环境

```bash
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3-pip ffmpeg libgl1 libglib2.0-0

python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
pip install -r requirements-rk3588.txt
pip install rknn-toolkit-lite2
```

如果需要本机窗口显示：

```bash
pip uninstall -y opencv-python-headless
pip install opencv-python
```

## 快速开始

说明：建议始终显式传 `--source`、`--weights`、`--data`。

### 1. PT 分割

```bash
python app/inference/segmentation.py \
  --source ./test_input \
  --weights ./weights/0414_qy++.pt \
  --data ./data/my.yaml \
  --project ./runs/segment \
  --device cpu \
  --load-masks
```

### 2. ONNX 实时路径规划

本机窗口：

```bash
python app/planning/realtime_pathplan.py \
  --source 0 \
  --weights ./weights/0414_qy++.onnx \
  --backend onnx \
  --data ./data/my.yaml \
  --device cpu \
  --display local
```

上位机预览：

```bash
python app/planning/realtime_pathplan.py \
  --source 0 \
  --weights ./weights/0414_qy++.onnx \
  --backend onnx \
  --data ./data/my.yaml \
  --device cpu \
  --display remote \
  --remote-port 8080
```

浏览器打开：

```text
http://<board-ip>:8080/stream.mjpg
```

### 3. RKNN 实时路径规划

本机窗口：

```bash
python app/planning/realtime_pathplan.py \
  --source 0 \
  --weights ./weights/0414_qy++.rknn \
  --backend rknn \
  --data ./data/my.yaml \
  --display local
```

上位机 HTTP/MJPEG：

```bash
python app/planning/realtime_pathplan.py \
  --source 0 \
  --weights ./weights/0414_qy++.rknn \
  --backend rknn \
  --data ./data/my.yaml \
  --display remote \
  --remote-port 8080
```

本机 + 上位机：

```bash
python app/planning/realtime_pathplan.py \
  --source 0 \
  --weights ./weights/0414_qy++.rknn \
  --backend rknn \
  --data ./data/my.yaml \
  --display both \
  --remote-port 8080
```

RTSP 流：

```bash
python app/planning/realtime_pathplan.py \
  --source rtsp://xxx \
  --weights ./weights/0414_qy++.rknn \
  --backend rknn \
  --data ./data/my.yaml \
  --display local
```

## 显示控制

`realtime_pathplan.py` 支持：

- `--display local`：仅本机窗口
- `--display remote`：仅上位机 MJPEG
- `--display both`：本机窗口 + 上位机 MJPEG
- `--display none`：不显示，只保存或后台运行

兼容旧参数：

- `--view` 等价于 `--display local`

MJPEG 相关参数：

- `--remote-host`：绑定地址，默认 `0.0.0.0`
- `--remote-port`：端口，默认 `8080`
- `--remote-path`：路径，默认 `/stream.mjpg`

## 模型导出

### PT -> ONNX

```bash
python yolo/export.py --weights ./weights/0414_qy++.pt --include onnx --imgsz 640 640
```

### ONNX -> RKNN

```bash
python tools/export_rknn.py \
  --onnx ./weights/0414_qy++.onnx \
  --output ./weights/0414_qy++.rknn \
  --target rk3588 \
  --opset-check
```

INT8 量化：

```bash
python tools/export_rknn.py \
  --onnx ./weights/0414_qy++.onnx \
  --output ./weights/0414_qy++.rknn \
  --target rk3588 \
  --quantize \
  --dataset ./data/rknn_dataset.txt
```

## 类别语义

见 `data/my.yaml` 与 `app/mapping/grid_map.py`。

当前关键类别：

- `0`: deliver_point
- `1`: car
- `3`: road_sign
- `4`: tree
- `5`: person
- `6`: forest
- `9`: house

其中：

- 目标点类别：`0`
- 可通行但有代价类别：`4, 6`

## 常见问题

### 1. 本机窗口不显示

如果是无桌面环境，不要用 `--display local` 或 `--display both`。

如果需要本机窗口，请安装：

```bash
pip install opencv-python
```

### 2. 上位机看不到流

确认：

- 板端使用了 `--display remote` 或 `--display both`
- 端口未被防火墙阻止
- 上位机和板端网络互通
- 浏览器访问的是：

```text
http://<board-ip>:8080/stream.mjpg
```

### 3. RKNN 运行时报模块缺失

板端运行需要：

```bash
pip install rknn-toolkit-lite2
```

导出 `.rknn` 需要：

```bash
pip install rknn-toolkit2
```

## 备注

更完整的中文说明见：`README_CN.md`
