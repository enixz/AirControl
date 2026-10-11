"""可行性论证统一适配器基类与 ReplayClock（T0 契约，方案 §7）。

所有候选路线的实验适配器继承 BaseHandModelAdapter；输出统一 per-frame dict，
计时约定：状态判断用 source_timestamp（ReplayClock），性能用 perf_counter。
"""
import time


class ReplayClock:
    """录像原始时间的只读时钟，替代状态机里的 time.monotonic()。

    离线回放时处理速度不受真实帧率约束，若状态机用 perf_counter，持有/
    投票/冷却判定会随本机处理速度漂移。ReplayClock 让 now = source_timestamp，
    与录制现场一致。铁律：一个进程顺序回放多段录像时，clock.value 必须随
    每帧更新（被 patch 的 services.* 模块挂旧 clock 的坑，见项目 memory）。
    """

    def __init__(self, start=0.0):
        self._value = start

    @property
    def value(self):
        return self._value

    def set(self, t):
        self._value = float(t)


class BaseHandModelAdapter:
    """候选模型统一接口。子类实现 _load_model / _infer_impl / close。"""

    def __init__(self, route_id, config=None):
        self.route_id = route_id
        self.config = config or {}
        self._loaded = False

    # -- 生命周期 ---------------------------------------------------------
    def reset(self):
        """清空跟踪/时序状态；每段录像开始前调用一次。"""

    def close(self):
        """释放模型资源。"""

    def _load_model(self):
        raise NotImplementedError

    # -- 主入口 -----------------------------------------------------------
    def infer(self, frame, source_timestamp):
        """喂一帧 BGR 图与录像时间戳，返回统一 per-frame dict（契约见 protocol.json）。

        计时：inference_start/end 用 perf_counter（真实耗时）；状态判断用
        source_timestamp。子类不要在返回 dict 里混用两种时钟。
        """
        if not self._loaded:
            self._load_model()
            self._loaded = True
        t0 = time.perf_counter()
        result = self._infer_impl(frame, source_timestamp)
        result["inference_start_ms"] = t0 * 1000.0
        result["inference_end_ms"] = time.perf_counter() * 1000.0
        if "frame_index" in result:
            result["frame_index"] = self._frame_index
            self._frame_index += 1
        return result

    def _infer_impl(self, frame, source_timestamp):
        raise NotImplementedError
