# 闭环式多模态视觉系统代码库

本项目实现了一个基于 Qwen3-VL、SAM2 和 CLIP 的闭环式多模态视觉问答系统，包含完整的模型推理、闭环控制、可靠性评测及可视化分析模块。

## 1. 文件结构说明

代码库位于 `code/` 目录下，采用模块化设计：

```text
code/
├── core/                   # 核心基础模块
│   ├── config.py           # 全局配置（路径、模型参数、API Key）
│   ├── main_loop.py        # 闭环系统主控制器（推理-定位-分割-验证）
│   └── utils.py            # 通用工具函数（图像加载、JSON读写等）
│
├── models/                 # 模型封装模块
│   ├── model_qwen.py       # Qwen-VL 模型接口（DashScope API）
│   ├── model_sam2.py       # SAM2 分割模型封装
│   └── model_clip.py       # CLIP 验证模型封装
│
├── experiments/            # 实验与评测脚本
│   ├── evaluation.py       # 基础 Baseline 评测 (One-shot)
│   ├── ablation.py         # 消融实验 (Baseline vs Loop)
│   ├── reliability.py      # 可靠性评测 (OOD, Size Robustness)
│   └── efficiency.py       # 效率与资源分析
│
├── analysis/               # 结果分析与可视化
│   ├── error_analysis.py   # 失败案例归因分析
│   ├── report.py           # 自动生成实验报告 (Markdown)
│   └── visualization_advanced.py # 高级图表与Excel报表生成
│
└── run.py                  # 项目统一运行入口
```

## 2. 代码功能描述

### 核心模块 (Core)
*   **`config.py`**: 管理项目路径、API 密钥、模型名称及超参数。
*   **`main_loop.py`**: 实现 `VQAClosedLoopSystem` 类，协调 Qwen、SAM2 和 CLIP 的交互，执行闭环推理流程。
*   **`utils.py`**: 提供图像预处理、BBox 计算、结果保存等辅助功能。

### 模型模块 (Models)
*   **`model_qwen.py`**: 封装 DashScope 的 MultiModalConversation 接口，支持多轮对话和证据定位 Prompt 生成。
*   **`model_sam2.py`**: 封装 SAM2 (或 SAM1 作为 Fallback)，支持基于 Box 提示的分割与裁剪。
*   **`model_clip.py`**: 封装 CLIP 模型，计算图像-文本余弦相似度，用于验证分割结果的有效性。

### 实验模块 (Experiments)
*   **`evaluation.py`**: 运行基线 (One-shot) 评测，计算 VQA Accuracy。
*   **`ablation.py`**: 运行消融实验，对比不同配置（如 Loop w/o CLIP）下的性能。
*   **`reliability.py`**: 执行 OOD 测试（图像扰动）和目标大小鲁棒性分析。
*   **`efficiency.py`**: 统计工具调用次数、推理耗时及显存占用。

### 分析模块 (Analysis)
*   **`error_analysis.py`**: 自动分析失败案例，归类为定位失败、验证拒绝或推理错误。
*   **`visualization_advanced.py`**: 基于 Seaborn 生成高质量分析图表和汇总 Excel。

## 3. 依赖关系说明

*   **内部依赖**: `core` 模块被其他所有模块引用；`models` 模块被 `main_loop.py` 引用；`experiments` 模块调用 `core` 和 `models`。
*   **外部依赖**:
    *   `dashscope`: Qwen-VL API 调用
    *   `torch`, `torchvision`: 深度学习基础
    *   `transformers`: CLIP 模型加载
    *   `segment-anything` / `sam2`: 分割模型
    *   `Pillow`, `numpy`, `opencv-python`: 图像处理
    *   `pandas`, `seaborn`, `matplotlib`, `openpyxl`: 数据分析与可视化
    *   `tqdm`: 进度条显示

## 4. 实验执行流程详解

本项目的实验流程设计严谨，分为以下四个有序步骤。推荐使用 `run.py` 统一入口进行调度。

### Step 1: 核心消融实验 (Ablation Study)
**执行脚本**: `code/experiments/ablation.py`
**执行命令**: `python code/run.py --ablation`

*   **任务目标**: 验证闭环系统各组件的有效性。
*   **执行内容**:
    1.  **Baseline (One-shot)**: 仅使用 Qwen-VL 直接回答问题，作为性能基准。
    2.  **Loop w/o CLIP**: 启用 Qwen+SAM2 闭环，但移除 CLIP 验证环节（即盲目接受所有分割结果）。
    3.  **Full Loop**: 启用完整的 Qwen+SAM2+CLIP 闭环，包含相关性验证和拒绝机制。
*   **输出产物**:
    *   `results/ablation/baseline_metrics.json`
    *   `results/ablation/full_loop_predictions.json` (包含每个样本的详细推理路径、BBox、CLIP分数)

### Step 2: 可靠性评测 (Reliability Assessment)
**执行脚本**: `code/experiments/reliability.py`
**执行命令**: `python code/run.py --reliability`

*   **任务目标**: 评估系统在非理想条件下的鲁棒性。
*   **执行内容**:
    1.  **OOD (Out-of-Distribution) 测试**: 对原始图像施加随机扰动（高斯噪声、运动模糊、亮度对比度变化），重新运行闭环推理，计算准确率下降幅度。
    2.  **目标大小分析**: 基于 Full Loop 的推理结果，根据目标 BBox 面积将样本划分为 Small (<32px), Medium (32-96px), Large (>96px) 三类，统计各组准确率。
*   **输出产物**:
    *   `results/reliability/ood_metrics.json`

### Step 3: 效率基准测试 (Efficiency Benchmark)
**执行脚本**: `code/experiments/efficiency.py`
**执行命令**: `python code/run.py --efficiency`

*   **任务目标**: 量化系统的计算成本。
*   **执行内容**:
    1.  **离线分析**: 读取 `full_loop_predictions.json`，统计平均每个样本调用 SAM2、CLIP 和 Qwen 的次数。
    2.  **在线测试**: 运行少量样本的实时推理，记录单样本平均耗时 (Latency) 和显存峰值占用。
*   **输出产物**:
    *   `results/efficiency/analysis.json`
    *   `results/efficiency/benchmark.json`

### Step 4: 结果分析与可视化 (Analysis & Visualization)
**执行脚本**: `code/analysis/*.py`
**执行命令**: `python code/run.py --analysis --report`

*   **任务目标**: 将原始数据转化为直观的图表和报告。
*   **执行内容**:
    1.  **错误归因 (Error Analysis)**: 分析失败样本，将其归类为：
        *   *Location Failure*: 未能检测到目标 BBox。
        *   *Verification Rejection*: CLIP 拒绝了不相关的分割结果。
        *   *Reasoning Failure*: 视觉证据正确，但 VLM 回答错误。
    2.  **图表绘制**: 生成消融对比柱状图、OOD 鲁棒性图、效率分析图、错误归因饼图。
    3.  **Excel 汇总**: 将所有关键指标汇总到 `summary_report.xlsx`。
    4.  **报告生成**: 自动生成 Markdown 格式的实验报告 `experiment_report.md`。
*   **输出产物**:
    *   `results/*.png` (多张分析图表)
    *   `results/summary_report_202x.xlsx`
    *   `results/experiment_report.md`

## 5. 快速开始

### 一键运行
运行所有实验并生成最终报告：
```bash
python code/run.py --all
```