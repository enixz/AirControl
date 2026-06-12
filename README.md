# 🎯 AirControl - 隔空手势+语音控制系统

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Windows%2010%2F11-0078d4?logo=windows&logoColor=white)
![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)
![Stars](https://img.shields.io/github/stars/enixz/AirControl?style=social)

**🚀 解放双手，隔空操控你的电脑**

*手势控制 · 语音助手 · 多模态交互*

[English](README_EN.md) | [中文](#快速开始)

</div>

---

## ✨ 一句话介绍

> **AirControl** 是一款基于 MediaPipe + 语音识别的 Windows 空中控制器，让你无需触碰键盘鼠标，仅通过**手势**和**语音**即可控制 PPT 演示、操控鼠标、屏幕板书，甚至启动语音助手。

---

## 🎬 功能亮点

<table>
<tr>
<td width="50%">

### 🖐️ 手势控制
- **演示模式**：挥手翻页、开始/结束播放
- **鼠标模式**：空中鼠标、捏合点击、剪刀滚动
- **板书模式**：单指书写、握拳清屏、形状校正

</td>
<td width="50%">

### 🎤 语音助手
- **离线关键词识别**：Sherpa-ONNX 直接识别命令短语（无需唤醒词）
- **离线听写**：SenseVoice-Small 把语音直接写到屏幕（板书模式说"开始板书"）
- **模式感知**：不同模式自动激活不同指令集，避免误触

</td>
</tr>
<tr>
<td>

### 🧠 智能识别
- **主控手自动选择**：抬哪只手用哪只手，配置零关心
- **卡尔曼滤波 + 幽灵手恢复**：21 关键点双重平滑，短暂遮挡自动补帧
- **形状校正**：自动识别并修正手绘图形
- **笔触距离自适应**：远距离自动变细，近距离自动变粗

</td>
<td>

### ⚡ 高性能
- **自动摄像头分辨率探测**：跨设备零配置，自动选最高 ≥20fps 模式
- **MJPEG 强制编码**：720p 也能跑满 30fps（HD-3000 等老摄像头友好）
- **WPS / PPT 自动定位**：扫注册表 + 通配搜索，换电脑无需改路径
- **断线自动重连**：USB 摄像头被拔出会自动恢复

</td>
</tr>
</table>

---

## 🚀 快速开始

### 1️⃣ 克隆项目

```bash
git clone https://github.com/enixz/AirControl.git
cd AirControl
```

### 2️⃣ 安装依赖

```bash
pip install -r requirements.txt
```

### 3️⃣ 运行程序

```bash
# GUI 版本（推荐）
python -m app.main_ui

# 命令行版本
python -m app.main
```

> 💡 **提示**：首次运行会自动下载 MediaPipe 模型文件（约 16MB），请确保网络连接正常。

---

## 📖 详细功能

### 🎭 三种交互模式

通过 **单手 🤟 手势保持约 1 秒**（拇指+食指+小指伸出）循环切换模式：

#### 1. 演示模式 📊

用于控制 PowerPoint 或 WPS 演示文稿。

| 手势 | 动作 | 快捷键 |
|------|------|--------|
| 👋 向右挥手 | 下一页 | → |
| 👋 向左挥手 | 上一页 | ← |
| 👋 向上挥手 | 开始播放 | F5 |
| 👋 向下挥手 | 结束播放 | Esc |
| 👍 拇指竖起 | 切换到目标软件 | - |
| ✊ 握拳 | 可自定义映射 | - |

> 🛡️ **防误触设计**：五指张开随意移动不会触发任何操作；手刚进入画面的前 0.3 秒内也不会触发动作。

#### 2. 鼠标模式 🖱️

将你的手变成空中鼠标，屏幕上会显示一个大的圆形光标。

| 手势 | 动作 | 视觉反馈 |
|------|------|----------|
| ☝️ 中指尖移动 | 控制光标位置 | 白色圆形光标 |
| 🤏 拇指+食指捏合 | 左键点击 | 蓝色扩散动画 |
| 🤏 拇指+中指捏合 | 右键点击 | 绿色扩散动画 |
| ✌️ 剪刀手移动 | 滚轮滚动 | 黄色箭头指示 |
| 🤏 捏合保持 | 左键拖拽 | 脉冲动画 |

**边缘加速**：当光标靠近屏幕边缘时，移动速度自动提升 2-3 倍，轻松访问任务栏和屏幕角落。

#### 3. 板书模式 ✏️

在屏幕上进行手写标注，适合教学或演示讲解。

| 手势 | 动作 |
|------|------|
| ☝️ 仅伸食指 + 拇指并拢 | 落笔书写 |
| ☝️ 仅伸食指 + 拇指分开 | 抬笔悬停（需手正对相机，侧面时拇指被遮挡不可信，笔状态冻结） |
| ✌️ 食指+中指伸出（无需张开，贴紧也算） | 抬笔悬停（侧面也可靠，随时可用） |
| ✊ 握拳 | 清空画布 |
| ✊…✊ 单手 1 秒内连续两次握拳 | 切换形状校正开/关 |
| 🤟 单手保持约 1 秒 | 切换交互模式（演示↔鼠标↔板书） |

**智能形状校正**：开启后，手绘的线条、三角形、矩形、椭圆会自动修正为标准几何图形。

> 💡 **🤟 保持切模式** 是全局通用的，不只板书模式。任何模式下单手摆出 🤟（拇指+食指+小指伸出）保持约 1 秒即触发。识别基于 MediaPipe 手势标签而非逐指几何判定，远距离同样可靠；触发后需放下手势才能再次切换。保持时长和判定占比可通过 `mode_switch_hold_sec` / `mode_switch_vote_ratio` 调整。

---

### 🎤 语音助手

AirControl 集成双引擎语音系统：

#### 离线关键词检测（KWS）
- **引擎**：Sherpa-ONNX（完全离线，保护隐私）
- **模型**：`kws-zh-wenetspeech`（中文，约 18 MB）
- **机制**：**无需唤醒词**，直接监听命令短语，命中即执行
- **指令集随模式切换**：演示模式只听演示相关词，板书模式只听板书相关词，互不干扰
- **延迟**：典型 ~100 ms 检出
- **冷却**：1.0 s 防抖，避免一句话连触发两次

#### 离线语音听写（ASR）
- **引擎**：SenseVoice-Small（阿里达摩院开源，sherpa-onnx 加载）
- **场景**：板书模式说 **"开始板书"** 开始录音，说 **"结束板书"** 停止并将语音转为文字写到画布
- **能力**：自由文本输入；支持中、英、日、韩、粤语自动检测
- **特性**：完全离线、无 API 配置、ITN 标点自动还原
- **磁盘占用**：模型约 234 MB（int8 量化）

##### 模型下载

```bash
# 从 sherpa-onnx 官方发布页下载 SenseVoice-Small
# https://github.com/k2-fsa/sherpa-onnx/releases (tag: asr-models)
# 文件名：sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2

# 解压后将整个目录重命名/移动到 AirControl/models/sense-voice/
# 期望结构：
#   models/sense-voice/
#     ├── model.int8.onnx
#     └── tokens.txt
```

未放置模型时听写功能自动停用，KWS 关键词不受影响。

#### 完整语音指令清单

各模式专属指令（仅在该模式下被激活）：

| 模式 | 指令 | 动作 |
|------|------|------|
| 演示 | `开始播放` `结束播放` `下一页` `上一页` | F5 / Esc / → / ← |
| 鼠标 | `点一下` `双击` `右键` | 左单击 / 左双击 / 右键 |
| 板书 | `清屏` `图形修正` | 清空画布 / 切换形状校正 |
| 板书 | `开始板书` `结束板书` | 启动 / 结束语音听写 |

任意模式下都可用：

| 指令 | 动作 |
|------|------|
| `演示模式` `鼠标模式` `板书模式` | 直接跳到指定模式（无需 🤟 手势切换） |
| `最小化助手` `显示助手` | 最小化 / 还原 AirControl 浮窗 |
| `召唤豆包` | 启动配置的语音助手应用 |

> 💡 浮窗上点 🎤 标签可弹出**完整语音指令面板**，当前模式的指令会高亮标注"← 当前"。再点 🎤 关闭面板，也可点面板右上角 ✕ 或直接拖动面板到任意位置。

---

## 🏗️ 技术架构

```
┌─────────────────────────────────────────────────────────┐
│                    AirControl 架构                       │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │  摄像头采集  │  │  语音输入   │  │  用户配置   │      │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘      │
│         │                │                │              │
│         ▼                ▼                ▼              │
│  ┌─────────────────────────────────────────────────┐    │
│  │              InferenceWorker (QThread)           │    │
│  │  ┌─────────────┐  ┌─────────────┐               │    │
│  │  │ MediaPipe   │  │ 卡尔曼滤波  │               │    │
│  │  │ HandLandmark│  │ + EMA 平滑  │               │    │
│  │  └─────────────┘  └─────────────┘               │    │
│  └─────────────────────────────────────────────────┘    │
│         │                                                │
│         ▼                                                │
│  ┌─────────────────────────────────────────────────┐    │
│  │              ModeManager (模式管理器)            │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐      │    │
│  │  │ 演示模式 │  │ 鼠标模式 │  │ 板书模式 │      │    │
│  │  └──────────┘  └──────────┘  └──────────┘      │    │
│  └─────────────────────────────────────────────────┘    │
│         │                                                │
│         ▼                                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │  PPT 控制   │  │  鼠标控制   │  │  屏幕绘制   │      │
│  └─────────────┘  └─────────────┘  └─────────────┘      │
└─────────────────────────────────────────────────────────┘
```

### 核心技术栈

| 组件 | 技术 | 用途 |
|------|------|------|
| 手部检测 | MediaPipe HandLandmarker | 21 关键点实时检测 |
| 手势识别 | ML 模型 + 规则回退 | 手势分类（拳头、张开、剪刀等） |
| 位置平滑 | 卡尔曼滤波 + EMA | 消除抖动，丢失时预测 |
| 形状校正 | OpenCV 几何分析 | 自动修正手绘图形 |
| GUI | PyQt6 | 悬浮窗、设置面板、覆盖层 |
| 鼠标控制 | Win32 API | SetCursorPos、mouse_event |
| 语音 KWS | Sherpa-ONNX | 离线关键词检测 |
| 语音 ASR | SenseVoice-Small（ONNX） | 离线语音听写 |
| 音频采集 | sounddevice | 实时音频流 |

---

## ⚙️ 配置选项

点击悬浮窗的 ⚙️ 设置按钮，可调整以下参数：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| **摄像头** | 后台异步枚举系统可用摄像头索引，下拉选完保存即**运行时切换**（笔记本+外接 USB 摄像头随时切） | 当前 |
| 控制目标软件 | PowerPoint 或 WPS | WPS |
| 手势模型精度 | Lite（更快）/ Heavy（更准） | Heavy |
| 交互模式 | presentation / mouse / draw | presentation |
| 手势防抖（冷却） | 连续手势间的最小间隔 | 1000 ms |
| 鼠标灵敏度 | 鼠标模式下的跟踪灵敏度 | 40% |
| 画笔粗细 | 板书模式下的笔触宽度 | 20 px |
| 边缘加速 | 鼠标靠近边缘时自动加速 | 开启 |
| 语音助手 | 选择语音助手应用 | 豆包 |
| 动作映射 | 各手势对应的具体操作 | 见 `config.json` |

### 进阶配置（直接编辑 `config.json`）

| 字段 | 说明 | 默认 |
|------|------|------|
| `camera_width` / `camera_height` | 摄像头分辨率，`null` 时自动探测最高 ≥ min_fps 的模式 | `null` |
| `camera_min_fps` | 自动探测时帧率下限，达不到的分辨率会被跳过 | 20 |
| `camera_force_mjpeg` | 强制 MJPEG 编码（老摄像头 720p 上 30fps 必需） | true |
| `dominant_hand` | 惯用手偏好：`Auto` / `Left` / `Right`，Auto 时纯靠运动+高度+近远自动选 | `Auto` |
| `hand_detection_confidence` | 手部检测阈值，远距离调低（0.4-0.6） | 0.6 |
| `hand_presence_confidence` | 手在画面中的判定阈值 | 0.5 |
| `hand_tracking_confidence` | 帧间跟踪阈值 | 0.5 |
| `pen_width_auto_scale` | 笔触粗细随手距自动缩放（关闭则始终同一粗细，光标灵敏度仍随手大小自适应） | false |
| `mode_switch_hold_sec` | 🤟 切模式手势需保持的时长（秒） | 1.0 |
| `mode_switch_vote_ratio` | 保持窗口内 🤟 标签帧占比阈值，远距离误检多可适当调低 | 0.6 |
| `draw_frontality_gate` | 板书拇指可观测性门限（掌宽/食指长）。低于此值视为手侧对相机、拇指不可信，书写状态冻结；横扫时笔画总断可调低，抬笔不灵敏可调高 | 0.55 |
| `draw_record_trace` | 板书时逐帧录制关键点到 `draw_trace.jsonl`，供 `simulate_draw.py --replay` 离线回放排查断触 | true |
| `dictation_enabled` | 启用 SenseVoice 离线语音听写（draw 模式说"开始板书"） | true |
| `dictation_language` | 听写语种：`auto`/`zh`/`en`/`ja`/`ko`/`yue` | `auto` |
| `wps_exe_path` | 手动覆盖 WPS 路径，自动定位失败时用 | （无） |
| `debug_overlay` | 启动即显示 FPS/手数/handedness 等调试信息 | false |

> 💡 调试覆盖层也可以**运行时按 F1 切换**——不用改 config 重启。

错误的配置值会被 schema 校验拦截并回退默认值（日志里会打 warning），不会让程序黑屏。

配置会自动保存到 `config.json` 文件中。

---

## 📁 项目结构

```
AirControl/
├── app/
│   ├── main.py                    # 命令行版本入口（OpenCV 窗口）
│   ├── main_ui.py                 # GUI 版本入口（PyQt6 悬浮窗）
│   ├── config_manager.py          # 配置文件读写管理
│   ├── mode_manager.py            # 模式管理器（🤟 保持切换）
│   ├── drawing_overlay.py         # 板书模式全屏画布
│   ├── draw_toolbar.py            # 画板工具栏
│   ├── mouse_cursor_overlay.py    # 鼠标光标叠加层
│   ├── modes/
│   │   ├── base.py                # 模式基类（策略模式）
│   │   ├── presentation.py        # 演示模式
│   │   ├── mouse_mode.py          # 鼠标模式
│   │   └── draw_mode.py           # 板书模式
│   ├── services/
│   │   ├── camera.py              # 摄像头服务
│   │   ├── hand_tracker.py        # 手部关键点追踪（MediaPipe + 卡尔曼滤波）
│   │   ├── gesture_recognizer.py  # 手势识别与挥动检测
│   │   ├── inference_worker.py    # 异步推理工作线程
│   │   ├── mouse_controller.py    # 鼠标控制（Win32 API）
│   │   ├── ppt_controller.py      # PPT/WPS 控制
│   │   ├── shape_recognizer.py    # 形状识别器
│   │   ├── voice_assistant.py     # 语音助手服务
│   │   ├── voice_command.py       # 语音命令处理（KWS）
│   │   └── voice_dictation.py     # 语音听写（SenseVoice-Small）
│   └── voice_keywords/            # 语音关键词配置
├── models/
│   ├── kws-zh-wenetspeech/        # 语音唤醒词模型
│   └── sense-voice/               # SenseVoice-Small ASR 模型（手动下载）
├── tests/                         # 单元测试
├── config.json                    # 用户配置文件
├── requirements.txt               # Python 依赖
├── build.py                       # PyInstaller 打包脚本
├── AirControl.spec                # PyInstaller 配置
├── hand_landmarker.task           # MediaPipe 手部模型（7.8MB）
└── gesture_recognizer.task        # 手势识别模型（8.4MB）
```

---

## 🧪 测试

运行单元测试：

```bash
# 运行所有测试
python -m pytest tests/

# 运行特定测试
python -m pytest tests/test_edge_map.py
```

测试覆盖：
- ✅ 配置边界检查
- ✅ 边缘映射算法
- ✅ 鼠标控制器兼容性
- ✅ 光标叠加层
- ✅ 语音助手集成
- ✅ UI 集成测试

---

## 📦 打包为可执行文件

项目已配置 PyInstaller，可一键打包为 Windows 可执行文件：

```bash
python build.py
```

打包后的文件将输出到 `dist/` 目录，包含所有依赖和模型文件。

---

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出建议！

1. Fork 本项目
2. 创建功能分支：`git checkout -b feature/your-feature`
3. 提交更改：`git commit -m 'feat: add your feature'`
4. 推送分支：`git push origin feature/your-feature`
5. 创建 Pull Request

### 开发环境

```bash
# 克隆项目
git clone https://github.com/enixz/AirControl.git
cd AirControl

# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 运行测试
python -m pytest tests/
```

### 代码规范

- 遵循 PEP 8 代码规范
- 添加类型注解
- 编写单元测试
- 更新文档

---

## 🙏 致谢

- [MediaPipe](https://mediapipe.dev/) - 手部检测和关键点追踪
- [Sherpa-ONNX](https://github.com/k2-fsa/sherpa-onnx) - 离线语音识别引擎
- [PyQt6](https://riverbankcomputing.com/software/pyqt/) - GUI 框架
- [OpenCV](https://opencv.org/) - 计算机视觉库

---

## 📄 许可证

本项目采用 [Apache License 2.0](LICENSE) 许可证。

```
Copyright 2026 AirControl

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

<div align="center">

**如果觉得有用，请给个 ⭐ Star 支持一下！**

[![Star History Chart](https://api.star-history.com/svg?repos=enixz/AirControl&type=Date)](https://star-history.com/#enixz/AirControl&Date)

</div>

---

<div align="center">

[⬆ 回到顶部](#-aircontrol---隔空手势语音控制系统)

</div>
