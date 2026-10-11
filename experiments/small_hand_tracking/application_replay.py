"""Actual worker, handoff, orchestrator and modes on source-time video.

Offline semantic replay: model construction is synchronous between frames,
prewarming, camera capture, Qt queue delays and mouse watchdog are excluded.
Desktop sinks are inert. This does not measure live frame age or click accuracy.
"""
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import sys
import time
from unittest.mock import MagicMock, patch

from .replay import ReplayClock, encode_hands, tracker_state


@contextmanager
def application_time(clock):
    bindings = []
    try:
        for name, module in list(sys.modules.items()):
            if name == "orchestrator" or name == "mode_manager" or name.startswith(("services.", "modes.")):
                if getattr(module, "time", None) is time and name != "services.inference_worker":
                    bindings.append((module, module.time))
                    module.time = clock
        yield
    finally:
        for module, original in reversed(bindings):
            module.time = original


def inert_sink():
    sink = MagicMock()
    sink.width.return_value = 1920
    sink.height.return_value = 1080
    sink.isVisible.return_value = True
    sink.move_to_normalized.return_value = (960, 540)
    sink.get_position.return_value = (960, 540)
    sink.cursor_pos = None
    return sink


class ApplicationReplay:
    def __init__(self, config, initial_timestamp):
        from config_manager import ConfigManager
        from orchestrator import AirControlOrchestrator
        from services.gesture_recognizer import GestureRecognizer
        from services.temporal_voter import TemporalGestureVoter
        from services.geometric_classifier import WeightedVoteClassifier
        from services.inference_worker import InferenceWorker
        from modes.mouse_mode import MouseMode

        self.clock = ReplayClock()
        self.clock.set(initial_timestamp)
        self.index = 0
        self.row = None
        self.events = []
        self.patches = ExitStack()
        self.trackers = []
        self.initialization = []
        if config.get("experiment_capture_handoff", False):
            from .capture_handoff import install_capture_handoff
            install_capture_handoff(self.patches)
        self.sinks = {name: inert_sink() for name in ("mouse", "overlay", "cursor", "toolbar", "ppt")}
        manager = ConfigManager.__new__(ConfigManager)
        manager.config = deepcopy(config)
        # Config mutations (gesture mode changes) remain confined to memory.
        manager._batch_depth = 0
        manager._dirty = False
        manager._do_save = lambda: True
        self.config = manager
        original_create = AirControlOrchestrator._create_tracker

        def create(owner, *args, **kwargs):
            started = time.perf_counter()
            tracker = original_create(owner, *args, **kwargs)
            self.initialization.append({"engine": tracker.engine_name,
                                        "duration_ms": (time.perf_counter() - started) * 1000})
            self.trackers.append(tracker)
            self.instrument(tracker)
            return tracker

        def services(owner):
            owner._tracker_config_signature = owner._tracker_signature()
            owner.tracker = owner._create_tracker()
            owner._engine_switcher._clock = self.clock.monotonic
            owner._configure_engine_switcher()
            owner.recognizer = GestureRecognizer(cooldown=manager.get("cooldown"), swipe_threshold=manager.get("swipe_threshold"))
            for key in ("pinch_hysteresis_enabled", "pinch_exit_hysteresis_enabled", "thumb_perp_ratio_enabled"):
                setattr(owner.recognizer, key, bool(manager.get(key, False)))
            owner._temporal_voter_enabled = bool(manager.get("temporal_voter_enabled", False))
            owner.gesture_voter = TemporalGestureVoter(config=manager)
            owner.geo_classifier = WeightedVoteClassifier(ml_weight=manager.get("geo_ml_weight", .4), geo_weight=manager.get("geo_geo_weight", .6))
            owner.ppt = self.sinks["ppt"]
            owner.inference_worker = InferenceWorker(None, owner.tracker)
            # No wall-clock pacing or QThread camera loop in offline replay.
            owner.inference_worker._frame_interval = 0
            owner.inference_worker.tracker_swapped.connect(owner._on_tracker_swapped)
            owner.inference_worker.frame_ready.connect(self.process_result)
            # 影子 ZOOM 确认晋升：生产接线（init_services 里 connect 的那个）在
            # init_services 被本函数替换后不会运行，这里接上同款生产 handler，
            # 使确认状态机在回放里真实走通（含冷启动播种与晋升交接）。
            owner.inference_worker.shadow_hand_ready.connect(owner._on_shadow_hand_ready)

        def build_inline(owner, target, name, args=()):
            # Preserve production build/ready/update/swap methods, serialize only
            # the background construction to obtain reproducible frame order.
            target(*args)

        for target, name, replacement in (
            (AirControlOrchestrator, "init_services", services),
            (AirControlOrchestrator, "_create_tracker", create),
            (AirControlOrchestrator, "_start_background_thread", build_inline),
            (AirControlOrchestrator, "_schedule_yolo_prewarm", lambda *a, **kw: None),
            (MouseMode, "_watchdog_loop", lambda *a: None),
        ):
            self.patches.enter_context(patch.object(target, name, replacement))
        # No sound when a real global gesture changes mode during replay.
        self.patches.enter_context(patch("orchestrator.winsound.PlaySound"))
        self.patches.enter_context(patch("orchestrator.winsound.MessageBeep"))
        try:
            with application_time(self.clock):
                self.app = AirControlOrchestrator(self.sinks["overlay"], self.sinks["cursor"], self.sinks["toolbar"], config=manager, mouse=self.sinks["mouse"])
            self.app.frame_processed.connect(self.on_processed)
        except BaseException:
            self.close()
            raise

    def instrument(self, tracker):
        if self.config.get("experiment_responsive_crop", False):
            from .responsive_crop import install_responsive_crop
            install_responsive_crop(tracker, self.patches)
        if self.config.get("experiment_identity_continuity", False):
            from .identity_continuity import install_identity_continuity
            install_identity_continuity(tracker, self.patches)
        original_find = tracker.find_hands
        original_detect = tracker._run_detection

        def raw(*args, **kwargs):
            if getattr(tracker, "_shadow_zoom", False):
                # 影子旁路调用不写行：rows 记录主 tracker 的输出，影子覆盖
                # 会冒充主手结果污染覆盖统计与引擎归因。
                return original_detect(*args, **kwargs)
            result = original_detect(*args, **kwargs)
            self.row["raw_hands"] = encode_hands(result[0], result[1], raw=True)
            self.row["used_crop_zoom"] = bool(result[2])
            return result

        def find(frame, draw=True):
            if getattr(tracker, "_shadow_zoom", False):
                return original_find(frame, draw=draw)
            self.row["actual_engine"] = tracker.engine_name
            self.row["before"] = tracker_state(tracker)
            started = time.perf_counter()
            result = original_find(frame, draw=draw)
            self.row["tracker_ms"] = (time.perf_counter() - started) * 1000
            self.row["hands"] = encode_hands(result[1], result[2])
            for gesture in result[2]:
                if "previous_track_key" in gesture and not gesture.get("predicted", False):
                    self.row["events"].append({"kind": "same_hand_rekey",
                                               "from": gesture["previous_track_key"],
                                               "to": gesture["track_key"]})
            self.row["after"] = tracker_state(tracker)
            return result

        self.patches.enter_context(patch.object(tracker, "_run_detection", raw))
        self.patches.enter_context(patch.object(tracker, "find_hands", find))
        if hasattr(tracker, "_next_mp_timestamp"):
            self.patches.enter_context(patch.object(tracker, "_next_mp_timestamp", self.clock.model_timestamp_ms))
        original_tier = tracker._sr.log_tier

        def tier(name, crop_size, target):
            if not getattr(tracker, "_shadow_zoom", False):
                self.row["upscalers"].append({"effective_tier": name, "crop_size": crop_size, "target": target})
            return original_tier(name, crop_size, target)

        self.patches.enter_context(patch.object(tracker._sr, "log_tier", tier))

    def on_processed(self, frame, hands, gestures, gesture):
        self.row["mode_gesture"] = gesture

    def process_result(self, frame, hands, gestures, worker):
        # Calling the method that _on_frame_ready delegates to lets exceptions
        # fail the run rather than being swallowed by the UI callback.
        try:
            # 影子确认状态机的同帧捕获输入：回放直连 _process_frame_results
            # 绕过了 _on_frame_ready，这里补上同款缓存（bbox_ratio 同算）。
            frame_h, frame_w = frame.shape[:2]
            bbox_ratio = 0.0
            if gestures:
                bbox_ratio = float(gestures[0].get("bbox_area", 0.0)) \
                    / max(frame_w * frame_h, 1)
            self.app._last_capture_result = (hands, gestures, bbox_ratio)
            self.row["mode_switch_guard_active"] = self.clock.time() - self.app.mode_manager.last_mode_switch_time < 1.
            self.app._process_frame_results(frame, hands, gestures)
        except BaseException as exc:
            self.callback_error = exc
        finally:
            worker.mark_result_consumed()

    def infer(self, frame, timestamp):
        if self.index or timestamp != self.clock.value:
            self.clock.set(timestamp)
        self.callback_error = None
        self.row = {"schema_version": 2, "pipeline": "application_semantic_replay", "frame_index": self.index,
                    "source_timestamp": float(timestamp), "source_size": [frame.shape[1], frame.shape[0]],
                    "raw_hands": [], "hands": [], "upscalers": [], "events": [], "sr_calls": [], "model_calls": [],
                    "auto_switch_executed": self.app._engine_switcher.enabled, "application_frame_age_ms": None,
                    "fsm_before": self.app._engine_switcher.state, "mode_before": self.app.current_mode_name}
        for sink in self.sinks.values():
            sink.reset_mock()
        previous_tracker = self.app.tracker
        initialization_before = len(self.initialization)
        with application_time(self.clock):
            self.app.inference_worker._process_frame(frame, 0, time.perf_counter())
        if self.callback_error is not None:
            raise self.callback_error
        self.row.update(fsm_state=self.app._engine_switcher.state, mode=self.app.current_mode_name,
                        switch_pending=self.app._engine_switch_pending, status_text=self.app.status_text,
                        construction_calls=self.initialization[initialization_before:])
        if self.row["fsm_before"] != self.row["fsm_state"]:
            self.row["events"].append({"kind": "fsm_transition", "from": self.row["fsm_before"], "to": self.row["fsm_state"], "reason": self.app._engine_switcher.last_reason})
        if previous_tracker is not self.app.tracker:
            self.row["events"].append({"kind": "tracker_swap", "engine": self.app.tracker.engine_name, "seeded_zoom": self.app.tracker._crop_zoom_mode})
        self.row["operation_events"] = [{"sink": name, "method": str(call[0]), "args": list(call[1])}
                                         for name, sink in self.sinks.items() for call in sink.method_calls
                                         if str(call[0]) in ("left_down", "left_up", "right_click", "scroll", "draw_to", "force_lift_pen", "clear_canvas", "toggle_shape_correction", "next_slide", "prev_slide")]
        self.row["operation_counts"] = {name: {str(call[0]): sum(c[0] == call[0] for c in sink.method_calls)
                                               for call in sink.method_calls} for name, sink in self.sinks.items()}
        mode = self.app.mode_manager.current_mode
        self.row["mode_state"] = {"left_holding": bool(getattr(mode, "_is_left_holding", False)),
                                  "writing": bool(getattr(mode, "_was_writing", False))}
        fresh_features = getattr(mode, "_last_fresh_features", None)
        if fresh_features:
            self.row["last_fresh_mouse_features"] = {
                k: fresh_features[k] for k in ("thumb_index_pinch", "thumb_middle_pinch", "hand_width", "geometry_reliable")
                if k in fresh_features}
        for name in ("_left_fsm", "_right_fsm"):
            fsm = getattr(mode, name, None)
            if fsm is not None:
                self.row["mode_state"][name] = {"state": fsm.state, "last_transition_reason": fsm.last_transition_reason,
                                                "blocked_until_open": fsm.blocked_until_open}
        self.index += 1
        return self.row

    def close(self):
        if hasattr(self, "app") and self.app.mode_manager.current_mode:
            self.app.mode_manager.current_mode.on_exit()
        self.patches.close()
        for tracker in self.trackers:
            tracker.close()
        self.trackers.clear()
