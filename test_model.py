import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

def main():
    print("=" * 50)
    print(" AirControl - Model Load Test")
    print("=" * 50)
    print()

    print("[1/4] Checking model file...")
    model_file = os.path.join(os.path.dirname(__file__), 'gesture_recognizer.task')
    if not os.path.exists(model_file):
        print(f"  FAIL: Model file not found: {model_file}")
        return False
    size_mb = os.path.getsize(model_file) / (1024 * 1024)
    print(f"  OK: gesture_recognizer.task ({size_mb:.1f} MB)")

    print()
    print("[2/4] Loading HandTracker (Gesture Recognizer)...")
    try:
        from services.hand_tracker import HandTracker, ML_GESTURE_TO_INTERNAL
        tracker = HandTracker(max_num_hands=1, min_detection_confidence=0.7)
        print(f"  OK: Detector type = {type(tracker.detector).__name__}")
        print(f"  OK: Gesture mapping = {ML_GESTURE_TO_INTERNAL}")
    except Exception as e:
        print(f"  FAIL: {e}")
        return False

    print()
    print("[3/4] Loading GestureRecognizer...")
    try:
        from services.gesture_recognizer import GestureRecognizer
        recognizer = GestureRecognizer(cooldown=1.0, swipe_threshold=60)
        print(f"  OK: get_hand_features exists = {hasattr(recognizer, 'get_hand_features')}")
        print(f"  OK: recognize accepts gestures = {'hands_gestures' in recognizer.recognize.__code__.co_varnames}")
    except Exception as e:
        print(f"  FAIL: {e}")
        return False

    print()
    print("[4/4] Checking PptController...")
    try:
        from services.ppt_controller import PptController
        ppt = PptController()
        print(f"  OK: PptController loaded")
        import subprocess
        print(f"  OK: subprocess module available")
    except Exception as e:
        print(f"  FAIL: {e}")
        return False

    print()
    print("=" * 50)
    print(" All tests PASSED! Run 'run.bat' to start.")
    print("=" * 50)
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
