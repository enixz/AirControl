# AirControl MoMA 版

本版本在 `codex/moma` 分支独立维护，用于移动云 MoMA 接入和课堂智能应用。
它保留本地演示、鼠标、板书和离线语音能力，增加文本意图解析、课堂小结与随堂出题。

## 版本区分

| 版本 | Git 引用 | 用途 |
| --- | --- | --- |
| MoMA 版 | `codex/moma` | 持续维护移动云集成与相关修复 |
| 原非 MoMA 发布版 | `v1.4.0` 标签 | 获取原有发布快照 |

截至 2026-10-15，`master` 本身已经包含一次历史 MoMA 接入提交，因此不能将当前
`master` 当作纯离线旧版。建立本分支不改写 `master` 或已有标签。
MoMA 后续提交直接进入本分支，不默认创建合并到 `master` 的 PR。
未来若发布独立安装包，使用 `AirControl-MoMA-<版本>-win64.zip` 命名，避免覆盖旧版资产。
当前应用内核版本仍为 1.4.0，分支名称区分版本系列。

## 获取与运行

```powershell
git clone --branch codex/moma --single-branch https://github.com/enixz/AirControl.git AirControl-MoMA
cd AirControl-MoMA
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.lock
.venv\Scripts\python.exe app/main_ui.py
```

建议 Windows 10/11 x64、Python 3.12。手势需要摄像头，语音需要麦克风，课件控制需要
WPS 或 PowerPoint。模型获取方法见主 README；Git 仓库不包含比赛 ZIP 中全部大型模型。
`models/sense-voice/` 用于本地听写，`ESPCN_x2.pb` 和 `Real-ESRGAN_x2plus.onnx`
用于超分，`models/silero_vad.onnx` 用于语音端点检测。缺少相应模型会使相关能力不可用或降级。
不要将“SR 开关已开启”理解为模型缺失时也能执行超分。

默认配置：

```json
{
  "long_range_enabled": true,
  "zoom_sr_engine": "auto",
  "detection_engine": "mediapipe",
  "engine_auto_switch": false,
  "cloud": { "enabled": false, "provider": "moma" }
}
```

ZOOM 根据手部大小进入或退出裁剪，SR Auto 按小手区域与硬件条件介入。
如果本机已有配置，既有设置优先；需要时在 `config.json` 中开启以上设置。
运行包配置通常位于 `%APPDATA%\AirControl`，也可通过 `AIRCONTROL_DATA_DIR` 指定独立目录，
例如 `%LOCALAPPDATA%\AirControl-MoMA`，避免两个版本共用个人设置。

## 启用 MoMA

1. 在启动程序的环境中设置自己的 `MOMA_API_KEY`。
2. 在配置文件的 `cloud` 节设置 `enabled=true`、`provider="moma"`。
3. 根据账号权限确认端点和模型；预设见 `app/services/cloud/providers.py`。

关键词、语音转写和播报在本地运行；启用云端功能时将转写文本发送给 MoMA，消耗账号 Token。
云不可用时基础手势与本地语音继续可用，依赖云端生成的能力受限。
不要提交 API Key、个人日志、IDE 登录态或比赛报名个人资料。

## 构建与验证

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest tests -q
.venv\Scripts\python.exe build.py --release
.\dist\AirControl\AirControl.exe --self-test
```

构建前将所需可分发模型放入上述路径。`build.py` 会重建当前工作副本的 `build`、`dist`
目录，建议在独立工作副本构建。构建环境 PATH 应以当前 Python 和 Windows 系统目录为主，
避免其他应用附带的 DLL 被 PyInstaller 收集进去。

中文路径适配：MediaPipe 从字节加载模型；受路径编码影响的原生加载器使用内容哈希校验的
ASCII 模型缓存。缓存通常位于系统临时目录下的 `AirControlNativeAssets`，配置和日志仍使用
各自数据目录。OpenCV Haar XML 由构建配置收集，用于人脸引导重捕。

2026-10-15 比赛交付快照通过 742 项测试与 195 项子测试，2 项跳过；EXE 在中文目录中自检成功。
ESPCN、Real-ESRGAN CPU 推理及 SenseVoice 模型加载通过。此结果不代替真实摄像头、
持续拖拽或所有教室环境的验收；固定录像中的拖拽连续性仍需优化。

独立 Git 工作副本复测为 740 项通过、4 项跳过、195 项子测试通过；部分模型按上述说明另行获取。
本次修改文件的 Ruff 检查通过；全仓 Ruff 仍报告 98 项既有问题，分布于 30 个未在本次修改的文件，
主要为实验脚本和既有控制代码的导入、未使用变量等检查项。该分支尚不应表述为全部 CI 门禁通过。
