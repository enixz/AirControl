"""语音指令的 Action 映射常量（纯数据，无音频 / 模型依赖）

从 voice_command.py 抽离，使 llm_intent 等模块可在没有 numpy / sherpa-onnx
的环境下复用同一份白名单，避免重复定义产生漂移。
"""

# ---------------------------------------------------------------------------
# 关键词 → Action 映射
# ---------------------------------------------------------------------------

VOICE_KEYWORD_TO_ACTION = {
    # 全局：助手 = 本程序窗口；豆包 = 外部 AI 助手
    "最小化助手": "minimize_assistant",
    "显示助手": "restore_assistant",
    "召唤豆包": "launch_voice_assistant",
    # 演示模式
    "开始播放": "start_presentation",
    "结束播放": "end_presentation",
    "下一页": "next_slide",
    "上一页": "prev_slide",
    # 鼠标模式
    "点一下": "left_click",
    "双击": "double_click",
    "右键": "right_click",
    # 板书模式
    "清屏": "clear_canvas",
    "开始板书": "start_dictation",
    "结束板书": "stop_dictation",
    # 模式直跳（已经覆盖"切模式"的需求，无需循环切换指令）
    "板书模式": "switch_to_draw",
    "鼠标模式": "switch_to_mouse",
    "演示模式": "switch_to_presentation",
    # 板书模式 — 图形修正
    "图形修正": "toggle_shape_correction",
    # 课堂智能层（M4）：触发词走 KWS 快通道，生成走云端 LLM 后台线程
    "下课总结": "class_summary",
    "出三道题": "class_quiz",
    # LLM 智能指令慢通道（M3）：唤醒词触发一句话录音 → 转写 → LLM 解析
    "小助手": "llm_command",
}

# 各模式可用的关键词（None 表示全部可用）
MODE_KEYWORDS = {
    "presentation": [
        "开始播放", "结束播放", "下一页", "上一页",
        "最小化助手", "显示助手", "召唤豆包",
        "板书模式", "鼠标模式",
        "小助手",
    ],
    "mouse": [
        "点一下", "双击", "右键",
        "最小化助手", "显示助手", "召唤豆包",
        "板书模式", "演示模式",
        "小助手",
    ],
    "draw": [
        "清屏", "开始板书", "结束板书", "图形修正",
        "最小化助手", "显示助手", "召唤豆包",
        "演示模式", "鼠标模式",
        "下课总结", "出三道题",
        "小助手",
    ],
}

# 手势专属、但语音表达同样合理的补充动作：
# orchestrator.execute_action 支持这些动作，语义泛化时纳入 LLM 白名单。
EXTRA_VOICE_ACTIONS = {
    "switch_app": ["切换应用", "换个应用", "切换窗口"],
    "hang_up_voice_assistant": ["挂断", "挂断助手", "关闭助手"],
}
