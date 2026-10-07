# MoMA 可用模型清单（实测）

> 实测日期：2026-10-01（比赛 10 亿 token 额度）；2026-10-05 复测（Coding Plan Lite 订阅）
> 2026-10-05 变化：订阅计费后部分模型由可用转 402「无可用套餐」（如 Qwen3-VL-32B、Qwen2.5-VL-7B、DeepSeek-R1-64K）；文本主力模型全部复测通过；新增 embeddings / rerank 端点可用
> 验证方法详见 `docs/m0_api_verification.md`

## 统一调用方式（所有可用模型同一路径）

```
POST https://zhenze-huhehaote.cmecloud.cn/v1/chat/completions
Headers: Authorization: Bearer $MOMA_API_KEY
         Content-Type: application/json
Body:    {"model": "<下表模型ID>", "messages": [...], "max_tokens": N}
```

模型列表查询：`GET https://zhenze-huhehaote.cmecloud.cn/v1/models`（同 Bearer 头）。

不可用端点：`https://moma.cmecloud.cn/v1` 及 `/api/*`（全线 subscription_required）；WS 语音端点（`wss://*/api-ws/v1/inference`）404。

## 一、实测可用（chat/completions 返回 200）

### 强通道（内容生成：小结/出题/复杂推理）

| 模型 | 类型 | 实测备注 |
|---|---|---|
| `DeepSeek-V3.2` | 非推理，直接回 content | ✅ 响应快，**首选** |
| `DeepSeek-V3` | 非推理（后端 v3.1） | ✅ 快 |
| `DeepSeek-R1` / `DeepSeek-R1-0528` / `deepseek-r1-128k` | 推理型 | ✅ 重推理任务 |
| `glm-5.3` | 推理型（带 reasoning_content） | ✅ |
| `QwQ-32B` | 推理型 | ✅ |
| `qwen3.5-397b-a17b` | 大参数 | ✅ |
| `MiniMax-M2.5` | — | ✅ |

### 快通道（指令理解：低延迟、短输出）

| 模型 | 类型 | 实测备注 |
|---|---|---|
| `deepseek-v4.1-flash` | **推理型**：先吐 reasoning，max_tokens 需 ≥512，否则 content 为空 | ✅ **默认首选** |
| `glm-5.3-flash` | 推理型（同上注意） | ✅ 备选 |
| `deepseek-v4-flash-0731` | 推理型 | ✅ |
| `GLM-5.2` | 可直接出 content | ✅ |
| `DeepSeek-V3` | 非推理、快 | ✅ 可作快通道备选（规避推理型空 content 坑） |

> **2026-10-02 快通道选型 A/B（意图解析，生产 prompt，max_tokens=512, temperature=0）**：
>
> | 模型 | p50 / max 延迟 | 备注 |
> |---|---|---|
> | `deepseek-v4.1-flash` | **1.06s / 3.65s** | 8 句意图解析全对；reasoning 隐性烧 ~130 tok/次（不计入 content，`usage.completion_tokens_details.reasoning_tokens` 可见），512 预算下 content 始终非空 |
> | `DeepSeek-V3` | 2.38s / 4.03s | 输出最省（21 tok），稳定但慢一倍 |
> | `GLM-5.2` | 1.8~6.7s（波动大） | "直接出 content"理论优势实测不成立，延迟反而最差，弃用 |
>
> 结论：**v4.1-flash 维持默认首选**。m0 文档"reasoning 型不适合快通道"的担忧
> 在该平台上被实测推翻——reasoning 与 content 并行吐出，延迟未受影响。
> 真正要防的坑已实测定位：`max_tokens < 256` 时 reasoning 烧光预算、content
> 返回空（128 时必空，256 时 reasoning≈130 刚好够）。所以任何走 flash 的
> 调用 max_tokens 下限 **256**（生产用 512 留余量）。健康 ping（max_tokens=1）
> 例外：flash 的 1-token ping 平均 0.23s 正常返回，不受此坑影响。

### 视觉（板书成文）

> 2026-10-01 同图 A/B（模拟板书：手写体公式+图形+噪点）：

| 模型 | 识图能力 | 实测结果 |
|---|---|---|
| `Qwen2.5-VL-72B-Instruct` | ✅ 专职 VL | 10.0s，全部识别正确，**默认首选**（密集手写板书留余量） |
| `Qwen2.5-VL-32B-Instruct` | ✅ 专职 VL | 10.0s，全部正确，轻量备选 |
| `glm-5.3-flash` | ✅ **支持识图**（本平台为多模态） | 9.2s，全部正确，可作备用视觉通道 |
| `deepseek-v4.1-flash` | ✅ **支持识图** | 8.6s，全部正确，可作备用视觉通道 |
| `glm-5.3` | ❌ 纯文本 | 明确回复看不到图，**勿用于视觉任务** |

注意：简单样本上四家都能识，"VL-72B 更强"只在密集/潦草真实板书上才有意义——M4 验收时用真实板书图复测选型。

### 编码（湛卢 IDE 消耗比赛 token）

> 2026-10-01 真实编码任务 A/B（jsonl 聚合函数 + PyQt6 状态灯组件，同题三家对比）：

| 模型 | 实测结果 |
|---|---|
| `deepseek-v4.1-flash` | ✅ **首选**：6.6-6.8s 出完整可用代码，边界处理干净，PyQt6 写法地道 |
| `glm-5.3` | ⚠️ 备选：代码质量好，但慢（25-41s）且 reasoning 烧光 max_tokens 出现过**空输出**（必须给足 max_tokens），适合评审而非主力编码 |
| `qwen2.5-coder-32b-Instruct` | ❌ 弃用：健壮性最差（缺编码参数、键校验顺序错、返回内层 defaultdict），PyQt6 任务布局也不合理 |

### 其他实测可用

`Qwen3.5-27B`、`jiutian-35b-256k-chat`

## 二、名单在列但实测不可用（404 路由不存在）

`AITC-Minimax-M2.5`、`AITC-Qwen3.5-35b`、`AITC-DeepSeek-V4-flash`、`Qwen2-VL-72B-Instruct`、`Qwen2-VL-7B-Instruct`、`Qwen3.5-4B-AWQ`、`DeepSeek-V2`、`SenseVoice`、`CosyVoice`、`Qwen3-ASR-Flash-Realtime`（音频类模型名走 chat 均 404）

**402 无套餐（2026-10-05 订阅制下存在但不开放给 Coding Plan Lite）**：`Qwen3-VL-32B-Instruct`、`Qwen2.5-VL-7B-Instruct`、`DeepSeek-R1-64K`

**端点级 404（2026-10-05 复测）**：`/v1/audio/speech`（TTS）、`/v1/audio/transcriptions`（ASR）、`/v1/images/generations`（Flux 绘图）均无路由——语音/绘图类在 chat 网关上仍不可用。

## 三、名单在列、2026-10-05 新增实测可用（Coding Plan Lite）

> 复测方法：chat/completions「回复两个字：收到」max_tokens=64~256，200 即记可用。

| 模型 | 延迟 | 备注 |
|---|---|---|
| `Qwen2.5-72B-Instruct` / `-64K` | ~2.2s | ✅ 直接出 content |
| `Qwen2.5-32B-Instruct`、`Qwen2.5-14B-Instruct(-1M)`、`Qwen2.5-7B-Instruct` | ~2.0-2.2s | ✅ |
| `Qwen3-32B` | 3.2s | ✅ 吐 reasoning（`<think:` 前缀混在 content） |
| `Qwen3.5-9B` | 2.2s | ✅ |
| `Qwen3.5-4B`、`Qwen3.5-35B-A3B`、`DeepSeek-V4-Flash`、`glm-5.1` | 2-9s | ⚠️ 200 但空 content（max_tokens=64 太小，reasoning 烧光；给足 512 应可用，未复测） |
| `glm-4-9b`、`Glm-4-9B-Chat` | ~2.1s | ✅ |
| `DeepSeek-V2-Lite-Chat`、`DeepSeek-V3-0324`、`DeepSeek-V3.1` | ~2.3s | ✅ |
| `DeepSeek-R1-Distill-Qwen-7B/32B` | ~3.7s | ✅ 推理型，reasoning 在 content 里 |
| 九天全家：`JIUTIAN-75B-32K`、`JIUTIAN-75B-8K-Chat`、`jiutian-safe-40b`、`jiutian-ict-35b-256k` | ~1.7s | ✅ |
| `bge-m3`、`bge-base-zh-v1.5` | — | ✅ **`/v1/embeddings` 端点可用**（200） |
| `bge-reranker-v2-m3` | — | ✅ **`/v1/rerank` 端点可用**（200，返回 relevance_score） |
| `qwen3-livetranslate-flash` | — | ❌ 404（小写变体也无路由，同 ASR 类） |

**503**：`Qwen3-235B-A22B`（服务暂时不可用，非套餐问题）

### 视觉复测（2026-10-05，手绘图「3+5=8」+三角形）

| 模型 | 结果 |
|---|---|
| `Qwen2.5-VL-72B-Instruct` | ✅ 2.3s 全对（维持默认首选） |
| `Qwen2.5-VL-32B-Instruct` | ✅ 2.6s 全对（轻量备选） |

## 三b、名单在列、仍未实测

- Qwen 视觉：`Qwen2-VL` 系全部 404；`Qwen3-VL-32B-Instruct` 402（见上）
- 其他非文本模态：`Flux.1-dev`（绘图）、`AICC-doubao-seedance-2.0`（视频）、`Fun-ASR-Flash-8k-Realtime` / `Qwen3-ASR-Flash-Realtime` / `Qwen3-LiveTranslate-Flash-Realtime`（语音，WS 端点 404，chat 网关无路由，当前无法使用）

## 四、AirControl 2.0 选型结论

| 角色 | 配置键 | 默认模型 |
|---|---|---|
| 指令理解（快） | `llm_fast_model` | `deepseek-v4.1-flash`（备选 `DeepSeek-V3`） |
| 内容生成（强） | `llm_smart_model` | `DeepSeek-V3.2`（备选 `glm-5.3`） |
| 板书成文（视觉） | `llm_vision_model` | `Qwen2.5-VL-72B-Instruct` |
| 湛卢编码 | — | `deepseek-v4.1-flash`（A/B 实测胜出；评审可用 `glm-5.3`，注意给足 max_tokens） |
