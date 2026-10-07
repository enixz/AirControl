# M0 · MoMA API 实测验证记录

> 日期：2026-10-01 ｜ Key：MOMA_API_KEY（尾号 …LrU，不入库）｜ 执行方式：curl 直连各端点
> 结论：**LLM 可用（模型远比预期多）；ASR/TTS 免费包内不可用**
> 复验：2026-10-02 抽样复验（见文末「5. 复验记录」），LLM 结论不变；ASR/TTS 错误码修正为 HTTP 404（路由不存在）

## 1. 端点总览

| 端点 | 用途 | 实测结果 |
|---|---|---|
| `https://zhenze-huhehaote.cmecloud.cn/v1`（chat/completions） | LLM | ✅ 全面可用，免订购 |
| `https://moma.cmecloud.cn/v1`（chat/completions） | LLM（模型广场） | ❌ 全线 `subscription_required`（当前账号没有可用套餐） |
| `https://moma.cmecloud.cn/api/v1/services/audio/asr/transcription`（fun-asr 文件转写） | ASR | ❌ HTTP 404（路由不存在）⚠️复验修正 |
| `https://moma.cmecloud.cn/api/v1/services/aigc/multimodal-generation/generation`（qwen3-tts-flash） | TTS | ❌ HTTP 404（路由不存在）⚠️复验修正 |
| `wss://moma.cmecloud.cn/api-ws/v1/inference`（带/不带斜杠、qwen/ 前缀均试） | ASR/TTS 流式 | ❌ 404 |
| `wss://zhenze-huhehaote.cmecloud.cn/api-ws/v1/inference` | ASR/TTS 流式 | ❌ 404（路径不存在） |
| 音频模型名（SenseVoice / CosyVoice / Qwen3-ASR-Flash-Realtime）走 zhenze chat | 探测路由 | ❌ 404 |

## 2. LLM 可用清单（zhenze 端点，实测 chat/completions 返回 200）

| 模型 | 实测 | 备注 |
|---|---|---|
| `DeepSeek-V3.2` | ✅ | 直接回 content，响应快，**强通道首选** |
| `DeepSeek-V3` | ✅ | 后端实为 v3.1，非推理，快 |
| `DeepSeek-R1` / `DeepSeek-R1-0528` | ✅ | 推理模型，重任务用 |
| `deepseek-v4.1-flash` / `deepseek-v4-flash-0731` | ✅ | **推理模型**：先吐 reasoning，max_tokens 要给足，否则 content 为空；快通道候选 |
| `glm-5.3` / `glm-5.3-flash` / `GLM-5.2` | ✅ | 均带 reasoning_content；5.2 可直接出 content |
| `qwen3.5-397b-a17b` / `Qwen3.5-27B` | ✅ | 27B 直接回 content |
| `qwen2.5-coder-32b-Instruct` | ✅ | 湛卢编码辅助可用 |
| `Qwen2.5-VL-72B-Instruct` | ✅ | **视觉模型，"板书成文"可用** |
| `QwQ-32B` | ✅ | 推理模型 |
| `MiniMax-M2.5` | ✅ | |
| `jiutian-35b-256k-chat` 等九天系列 | ✅ | |
| `AITC-Minimax-M2.5` / `Qwen2-VL-72B-Instruct` | ❌ 404 | 名单在列但路由不存在 |

- 响应头**无限流信息**暴露（仅 envoy/x-request-id 等）；3000 token/分钟限流需专门压测，暂未触发。
- 中文请求体注意必须 UTF-8 编码（curl 直贴中文会报 JSON 解析错，属客户端编码问题，非服务端限制）。

## 3. 对方案的影响

1. **LLM 智能层（M3 指令理解 / M4 课堂智能 / 板书成文）完全可行**，且模型选择优于原计划（有 DeepSeek-V3.2/R1 和 VL-72B）。
2. **语音上云（M2 听写云端化 / M5 云端 TTS）在免费包内不可行** → 方案调整：
   - 听写/指令识别保留本地 SenseVoice + KWS（v1.4 成熟能力）；
   - M3 慢通道改为「本地 SenseVoice 转写 → zhenze LLM 意图解析」；
   - TTS 播报改用本地 Windows SAPI（pyttsx3，离线零成本）；
   - `models/sense-voice/` 保留在发布包内（原 M7 瘦身取消）。
3. 若未来比赛方放开语音模型订购或确认语音在其他套餐，可按 v1.1 原计划回补（cloud 包保留 asr/tts 接口位）。

## 4. M0 剩余项

- [x] LLM 端点与模型可用性实测（本文件）
- [x] ASR/TTS 抵扣/可用性实测 → 结论：免费包内不可用
- [x] 复验（2026-10-02）→ LLM 结论不变；ASR/TTS 错误码修正为 404
- [ ] 限流压测（3000 token/分钟行为，开发期顺带观察即可）
- [ ] 当前 v1.4.0 打包体积实测（供材料"工程"章节使用）
- [ ] 大赛作品提交截止日确认（排期倒排依据）

---

## 5. 复验记录（2026-10-02）

> 方法：项目 `.venv` Python + `httpx` 直连；WS 用 `websockets` 真实握手。
> 脚本：临时目录 `m0_recheck.py` / `m0_recheck2.py` / `m0_ws.py`（未入库，可复跑）。
> 目的：核实初测结论是否仍成立。

| 探测项 | 复验结果 | 与初测对比 |
|---|---|---|
| zhenze `chat/completions` · `deepseek-v4.1-flash` | ✅ 200，1.7s，content 正常（带 reasoning） | 一致 |
| zhenze `chat/completions` · `DeepSeek-V3.2` | ✅ 200，**12.9s**，content 正常 | 可用性一致；**延迟波动大**（初测 22s），“响应快”表述需修正 |
| zhenze `GET /v1/models` | ✅ 200，模型列表可达 | 一致 |
| moma `POST /v1/chat/completions` | 402 `subscription_required` | 一致 |
| moma `POST /api/v1/services/audio/asr/transcription` | ❌ **HTTP 404 空 body**（路由不存在） | ⚠️ **修正**：初测记 `subscription_required` |
| moma `POST /api/v1/services/aigc/multimodal-generation/generation` | ❌ **HTTP 404 空 body**（路由不存在） | ⚠️ **修正**：同上 |
| WS `moma` / `zhenze` `/api-ws/v1/inference`（含斜杠、`/v1` 前缀） | ❌ 握手 HTTP 404 | 一致 |
| zhenze chat 送语音模型名（SenseVoice / CosyVoice / qwen3-asr-flash / qwen3-tts-flash / Realtime） | ❌ 404（nginx） | 一致 |

**修正说明**：ASR/TTS 的 HTTP 端点返回 **404（路由不存在）**，而非 `subscription_required`（后者为套餐未订购，HTTP 402，见 moma `/v1/chat/completions`）。两者含义不同：

- `subscription_required` → 订购套餐后可能可解；
- `404` → **平台未开放该 API 路由**，自行订购也未必可用。

因此第 3 节“待平台放开后回补”的前提应理解为**平台开放路由**，不能寄望于自行订购套餐。

**复验结论**：M0 核心判断成立 —— LLM 全面可用、云端 ASR/TTS 不可用。方案调整（听写保留本地 SenseVoice、TTS 走本地 Windows SAPI、M3 慢通道改为本地转写 + zhenze LLM 意图解析）依据充分，维持不变。

**衍生提示**：`DeepSeek-V3.2` 实测 12.9s（初测 22s），而方案原定强通道“允许 3-8s”——M4 课堂小结/出题面向的正是该模型，3-8s 的预期偏乐观，建议在 M4 实测时重新标定超时与 UI 等待提示；快通道 `deepseek-v4.1-flash` 实测 1.7s，与方案“<800ms”目标亦有差距，且属推理型（须给足 max_tokens 512+），M3 指令理解的真实 P50/P95 需在 M9 专门测量。
