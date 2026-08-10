"""手部追踪器工厂 — 根据配置创建对应引擎的追踪器实例。

支持的引擎：
  - "mediapipe": MediaPipe GestureRecognizer
  - "hagrid_yolo": HaGRID YOLOv10n 检测 + MediaPipe HandLandmarker 关键点（实验性）
  - "person_pose_hand": 框人→yolov8-pose 拿手腕→框手→（小手超分）→关键点（实验性，远距/侧位）

用法：
    from services.hand_tracker_factory import create_hand_tracker

    tracker = create_hand_tracker(
        engine="mediapipe",
        max_num_hands=2,
    )
"""

import logging


def create_hand_tracker(engine="mediapipe", **kwargs):
    """根据引擎名称创建对应的手部追踪器。

    Args:
        engine: "mediapipe" | "hagrid_yolo"
        **kwargs: 传递给具体追踪器的参数

    Returns:
        BaseHandTracker 子类实例
    """
    engine = str(engine).lower().strip()

    if engine == "mediapipe":
        return _create_mediapipe_tracker(**kwargs)
    elif engine == "hagrid_yolo":
        return _create_hagrid_yolo_tracker(**kwargs)
    elif engine == "person_pose_hand":
        return _create_person_pose_tracker(**kwargs)
    else:
        logging.warning("未知引擎 '%s'，使用 MediaPipe", engine)
        return _create_mediapipe_tracker(**kwargs)


def _create_mediapipe_tracker(**kwargs):
    """创建 MediaPipe 追踪器。"""
    from .hand_tracker import HandTracker

    # 过滤掉 MediaPipe 不认识的参数
    valid_keys = {
        "static_image_mode", "max_num_hands", "min_detection_confidence",
        "min_presence_confidence", "min_tracking_confidence",
        "preferred_model_type", "dominant_hand", "config",
    }
    filtered = {k: v for k, v in kwargs.items() if k in valid_keys}
    tracker = HandTracker(**filtered)
    logging.info("手部追踪器初始化成功: MediaPipe")
    return tracker


def _create_hagrid_yolo_tracker(**kwargs):
    """创建 HaGRID YOLO + MediaPipe HandLandmarker 混合追踪器。

    ⚠️ 实验性引擎。默认检测器 models/hand_yolov8n.onnx 不随安装包分发
    （AGPL-3.0 许可证），需按 README 指引手动下载。
    """
    from .hagrid_yolo_hand_tracker import HagridYoloHandTracker

    valid_keys = {
        "max_num_hands", "min_detection_confidence",
        "min_presence_confidence", "min_tracking_confidence",
        "dominant_hand", "config",
    }
    filtered = {k: v for k, v in kwargs.items() if k in valid_keys}
    tracker = HagridYoloHandTracker(**filtered)
    logging.info("手部追踪器初始化成功: HaGRID YOLO (hybrid)")
    return tracker


def _create_person_pose_tracker(**kwargs):
    """创建 框人→姿态→手腕→框手 混合追踪器（远距/侧位实验引擎）。

    ⚠️ 实验性。需 models/yolov8n-pose.onnx（不入 git）；缺失时退化为 HaGRID YOLO。
    """
    from .person_pose_hand_tracker import PersonPoseHandTracker

    valid_keys = {
        "max_num_hands", "min_detection_confidence",
        "min_presence_confidence", "min_tracking_confidence",
        "dominant_hand", "config",
    }
    filtered = {k: v for k, v in kwargs.items() if k in valid_keys}
    tracker = PersonPoseHandTracker(**filtered)
    logging.info("手部追踪器初始化成功: Person-Pose-Hand (hybrid)")
    return tracker
