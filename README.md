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
- **离线唤醒**：Sherpa-ONNX 关键词检测（隐私保护）
- **在线识别**：腾讯云 ASR 自由文本输入
- **模式感知**：不同模式自动切换语音指令

</td>
</tr>
<tr>
<td>

### 🧠 智能识别
- **卡尔曼滤波**：21 关键点双重平滑
- **形状校正**：自动识别并修正手绘图形
- **边缘加速**：鼠标靠近屏幕边缘自动加速

</td>
<td>

### ⚡ 高性能
- **异步推理**：后台线程处理 MediaPipe
- **30 FPS**：实时流畅的手势追踪
- **低延迟**：< 50ms 响应时间

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

通过 **双手握拳** 手势循环切换模式：

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
| ☝️ 仅伸出食指 | 书写/绘画 |
| 👌 拇指+食指捏合 | 悬停（不绘制） |
| ✊ 握拳 | 清空画布 |
| ✊✊ 双手握拳 | 切换形状校正 |

**智能形状校正**：开启后，手绘的线条、三角形、矩形、椭圆会自动修正为标准几何图形。

---

### 🎤 语音助手

AirControl 集成双引擎语音系统：

#### 离线关键词检测（KWS）
- **引擎**：Sherpa-ONNX（完全离线，保护隐私）
- **模型**：`kws-zh-wenetspeech`（中文，3.3MB 轻量模型）
- **唤醒词**：可自定义（默认："小助手"）
- **指令**：播放、暂停、上一页、下一页等固定命令

#### 在线语音识别（ASR）
- **引擎**：腾讯云实时语音识别
- **场景**：板书模式"打在屏幕上"功能
- **能力**：自由文本输入，支持中英文混合
- **限制**：需要 API 密钥（5 小时/月免费额度）

#### 模式感知
语音指令会根据当前模式自动切换：
- **演示模式**：播放、暂停、上一页、下一页
- **鼠标模式**：点击、右键、滚动
- **板书模式**：清屏、撤销、打在屏幕上

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
| 语音 ASR | 腾讯云 | 在线语音识别 |
| 音频采集 | sounddevice | 实时音频流 |

---

## ⚙️ 配置选项

点击悬浮窗的 ⚙️ 设置按钮，可调整以下参数：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| 控制目标软件 | PowerPoint 或 WPS | WPS |
| 手势模型精度 | Lite（更快）/ Heavy（更准） | Heavy |
| 交互模式 | presentation / mouse / draw | mouse |
| 手势防抖（冷却） | 连续手势间的最小间隔 | 1000 ms |
| 鼠标灵敏度 | 鼠标模式下的跟踪灵敏度 | 40% |
| 画笔粗细 | 板书模式下的笔触宽度 | 20 px |
| 边缘加速 | 鼠标靠近边缘时自动加速 | 开启 |
| 语音助手 | 选择语音助手应用 | 豆包 |
| 动作映射 | 各手势对应的具体操作 | 见 `config.json` |

配置会自动保存到 `config.json` 文件中。

---

## 📁 项目结构

```
AirControl/
├── app/
│   ├── main.py                    # 命令行版本入口（OpenCV 窗口）
│   ├── main_ui.py                 # GUI 版本入口（PyQt6 悬浮窗）
│   ├── config_manager.py          # 配置文件读写管理
│   ├── mode_manager.py            # 模式管理器（双手握拳切换）
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
│   │   └── voice_command.py       # 语音命令处理
│   └── voice_keywords/            # 语音关键词配置
├── models/
│   └── kws-zh-wenetspeech/       # 语音唤醒词模型
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
