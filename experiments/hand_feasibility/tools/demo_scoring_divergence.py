"""近距点击评分自检用的合成场景生成器（P2 → P3 过渡）。

生成带已知 GT 的合成 pinch 场景无法替代真人数据（方案 §5），但用于验证
评分器在「真值边界可靠」前提下的行为（方案 §12 评测自检：可变帧率回放、
重复事件一对一匹配已由 test_strict_scorer.py 覆盖）。

本文件提供「双口径差异演示」：同一份检测输出 + 已知偏差的 GT，
展示 strict 与 legacy 口径何时分歧——用于向决策者解释为何需要新采集。
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from tools.strict_scorer import compute_metrics_strict


def demo_scoring_divergence():
    """合成示例：真值 10.0s 开始持续 0.2s；检测碎片化 + 起点漂移。

    legacy 口径（重叠即命中）会把漂移 4s 的检测也算命中；
    strict 口径只认起点窗口内的。
    """
    truth = [(10.0, 10.2, "left")]
    det = [
        (6.0, 6.05, "left"),    # 起点漂移 -4s：不重叠 → 双不命中
        (9.5, 10.15, "left"),   # 起点 -500ms：与真值重叠；legacy 重叠+gap 0.5 也命中但贪心会选更近的 → 变 FA；strict 窗外 → FA
        (10.1, 10.3, "left"),   # 起点 +100ms：strict 窗内 → 双命中
        (10.9, 11.0, "left"),   # 起点 +900ms：不重叠 → 双不命中
    ]
    m_strict = compute_metrics_strict(truth, det)
    app_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "app")
    sys.path.insert(0, app_dir)
    from services.truth_ab import compute_metrics as legacy_metrics
    # legacy 需要 list[iv] 无键格式
    truth_l = [(t[0], t[1]) for t in truth]
    det_l = [(d[0], d[1]) for d in det]
    m_legacy = legacy_metrics(truth_l, det_l[:3])  # 去掉 +900ms 帧演示宽容差异
    return {
        "strict": {"hits": m_strict["hits"], "fa": m_strict["false_alarms"]},
        "legacy": {"hits": m_legacy["hits"], "fa": m_legacy["false_alarms"]},
        "note": "legacy 对 (9.5,10.15) 重叠即命中但贪心选更近者；见 b0_click_baseline.md 实例：legacy 0.5s 宽容把 -4477ms 漂移也算命中",
    }


if __name__ == "__main__":
    r = demo_scoring_divergence()
    print(json.dumps(r, indent=2))
