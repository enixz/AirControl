"""进程级 GPU 串行门：DirectML 跨线程并发访问保护。

背景（2026-10-12 实机崩溃，crash.log）：
  推理线程正在跑 SR 的 session.run（Real-ESRGAN on DmlExecutionProvider）
  时，YOLO 预热线程并发创建 InferenceSession——两个 DML session 并发即
  access violation（原生崩溃，Python 层拦不住）。同一窗口新建的 YOLO
  session 只拿到 CPUExecutionProvider，佐证 DML 运行时当时已异常；此前
  的 'utf-8' codec 报错是 DML 抛出的 GBK 中文错误被当 UTF-8 解码所致。

方案：全进程一把可重入锁，串行化所有 ORT(DML) session 的创建与推理。
  - 推理帧率高（~30Hz）而模型重建是低频事件（FSM 交接/预热），串行化
    只在"建/销毁恰好撞上推理"的窗口互斥，常态零竞争。
  - 可重入：同线程自嵌套（如 tracker init 内先建 SR 后建 YOLO）不自锁。
  - MediaPipe HandLandmarker 底层走同一 DML 设备栈，一并纳入保护。

用法：
  from services.gpu_guard import gpu_lock
  with gpu_lock():
      session = ort.InferenceSession(...)   # 创建
      outputs = session.run(...)           # 推理
"""

import threading

# 单例：进程内所有 GPU (DML/CUDA) 访问共用。可重入，允许同线程嵌套
# （tracker 构建内部多次进出；landmarker detect 内含 crop 推理等）。
_gpu_lock = threading.RLock()


def gpu_lock():
    """返回进程级 GPU 串行锁（RLock 单例）。"""
    return _gpu_lock
