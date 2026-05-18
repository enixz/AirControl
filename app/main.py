import cv2
import time
import sys
import os

# 确保能导入同级目录下的包
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.camera import CameraService
from services.hand_tracker import HandTracker
from services.gesture_recognizer import GestureRecognizer
from services.ppt_controller import PptController

def main():
    print("===============================")
    print(" AirControl - PPT手势控制系统")
    print("===============================")
    print("正在初始化模块...")
    
    camera = CameraService(camera_index=0)
    tracker = HandTracker(max_num_hands=1, min_detection_confidence=0.7)
    recognizer = GestureRecognizer(cooldown=1.0, swipe_threshold=60)
    ppt = PptController()
    
    print("正在启动摄像头...")
    try:
        camera.start()
    except Exception as e:
        print(f"摄像头启动失败: {e}")
        print("请检查摄像头是否被其他程序占用，或修改 camera_index。")
        return

    print("系统已准备就绪！")
    print("-------------------------------")
    print("使用说明：")
    print("1. 确保已打开 PowerPoint 或 WPS 演示。")
    print("2. 举起一只手，在摄像头前：")
    print("   - 【握拳】: 查找并切换到 WPS 演示窗口")
    print("   - 【手指并拢伸直(像拍手) + 左右挥动】: 触发 上一页 / 下一页")
    print("   - 【手指并拢伸直(像拍手) + 上下挥动】: 触发 开始 / 结束播放")
    print("   (五指张开随意移动时，不会触发任何操作，防误触)")
    print("3. 按键盘 'q' 键退出程序。")
    print("-------------------------------")
    
    status_text = "Ready"
    status_color = (0, 255, 0)
    status_timer = 0
    
    # 创建一个可调整大小且始终置顶的窗口
    window_name = "AirControl - PPT Gesture Controller"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 320, 240) # 缩小尺寸，避免挡住整个PPT
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1) # 置顶属性
    
    while True:
        success, frame = camera.read_frame()
        if not success:
            print("读取摄像头画面失败，可能摄像头已断开")
            break
            
        # 翻转画面（水平镜像），这样用户的左手在屏幕左边，符合直觉
        frame = cv2.flip(frame, 1)
        
        # 1. 追踪手部关键点
        frame, hands_landmarks, hands_gestures = tracker.find_hands(frame, draw=True)
        
        # 2. 识别手势
        gesture = recognizer.recognize(hands_landmarks, hands_gestures)
        
        # 3. 控制 PPT
        if gesture == "SWIPE_RIGHT":
            status_text = "Next Page ->"
            status_color = (0, 255, 255) # 黄色
            status_timer = time.time()
            ppt.next_slide()
            
        elif gesture == "SWIPE_LEFT":
            status_text = "<- Prev Page"
            status_color = (0, 255, 255)
            status_timer = time.time()
            ppt.prev_slide()
            
        elif gesture == "SWIPE_UP":
            status_text = "^ Start PPT"
            status_color = (255, 0, 255) # 紫色
            status_timer = time.time()
            ppt.start_presentation()
            
        elif gesture == "SWIPE_DOWN":
            status_text = "v End PPT"
            status_color = (255, 0, 255) # 紫色
            status_timer = time.time()
            ppt.end_presentation()
            
        elif gesture == "FIST":
            status_text = "[FIST] Switch WPS"
            status_color = (0, 165, 255) # 橙色
            status_timer = time.time()
            ppt.switch_app()
            
        # 恢复默认状态显示文字
        if time.time() - status_timer > 1.0:
            status_text = "Ready (Waiting for swipe)" if hands_landmarks else "No Hand Detected"
            status_color = (0, 255, 0) if hands_landmarks else (0, 0, 255) # 绿 / 红
            
        # 在画面上绘制状态提示
        cv2.putText(frame, f"Status: {status_text}", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                    
        # 如果在冷却中，显示提示，防止用户以为系统卡死
        if gesture == "COOLDOWN":
            cv2.putText(frame, "Cooldown...", (10, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        
        # 显示预览画面
        cv2.imshow(window_name, frame)
        
        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("正在退出...")
            break
            
    # 释放资源
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
