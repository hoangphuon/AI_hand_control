import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pyautogui
import time
import math
from screeninfo import get_monitors
import os

# Cấu hình PyAutoGUI để đảm bảo an toàn
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0

class HandDetector:
    """
    Lớp hỗ trợ phát hiện bàn tay sử dụng MediaPipe Tasks.
    """
    def __init__(self, model_path='hand_landmarker.task', max_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy tệp mô hình tại: {model_path}")
            
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE, # Sử dụng chế độ IMAGE để đơn giản hóa vòng lặp
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.results = None

    def find_hands(self, img):
        # MediaPipe Tasks yêu cầu định dạng mp.Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        self.results = self.detector.detect(mp_image)
        return img

    def find_position(self, img, hand_no=0):
        lm_list = []
        if self.results and self.results.hand_landmarks:
            if hand_no < len(self.results.hand_landmarks):
                my_hand = self.results.hand_landmarks[hand_no]
                h, w, c = img.shape
                for id, lm in enumerate(my_hand):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append([id, cx, cy])
        return lm_list

def draw_neon_line(img, pt1, pt2, color, thickness=2, glow_factor=3):
    """
    Vẽ đường thẳng phong cách Neon (Cyberpunk).
    """
    for i in range(glow_factor, 0, -1):
        t = thickness + i * 2
        alpha = 1.0 / (i + 1)
        overlay = img.copy()
        cv2.line(overlay, pt1, pt2, color, t)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.line(img, pt1, pt2, (255, 255, 255), thickness)

def draw_cyberpunk_ui(img, hands_detected, gesture, fps):
    """
    Vẽ giao diện người dùng phong cách Cyberpunk.
    """
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (300, 150), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    cv2.rectangle(img, (10, 10), (300, 150), (0, 255, 255), 2)
    cv2.putText(img, f"So tay: {hands_detected}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(img, f"Cu chi: {gesture}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    cv2.putText(img, f"FPS: {int(fps)}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

def main():
    w_cam, h_cam = 640, 480
    cap = cv2.VideoCapture(0)
    cap.set(3, w_cam)
    cap.set(4, h_cam)

    # Đường dẫn tới tệp mô hình
    model_path = os.path.join(os.path.dirname(__file__), 'hand_landmarker.task')

    try:
        monitors = get_monitors()
        if monitors:
            w_scr, h_scr = monitors[0].width, monitors[0].height
        else:
            w_scr, h_scr = 1920, 1080
    except Exception as e:
        print(f"Lỗi khi lấy thông tin màn hình: {e}")
        w_scr, h_scr = 1920, 1080

    try:
        detector = HandDetector(model_path=model_path, max_hands=1)
    except Exception as e:
        print(f"Lỗi khởi tạo bộ phát hiện: {e}")
        return
    
    p_time = 0
    smoothening = 5
    p_loc_x, p_loc_y = 0, 0
    c_loc_x, c_loc_y = 0, 0
    frame_reduction = 100
    
    print("Đang khởi động điều khiển bằng tay... Nhấn 'q' để thoát.")
    
    while True:
        success, img = cap.read()
        if not success:
            print("Không thể đọc từ Camera.")
            break
            
        img = cv2.flip(img, 1)
        # Chuyển đổi BGR sang RGB cho MediaPipe Tasks
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = detector.find_hands(img_rgb)
        lm_list = detector.find_position(img)
        
        gesture = "KHONG"
        hands_count = 1 if lm_list else 0
        
        if len(lm_list) != 0:
            x1, y1 = lm_list[8][1:] # Ngón trỏ
            x2, y2 = lm_list[12][1:] # Ngón giữa
            x0, y0 = lm_list[4][1:] # Ngón cái
            
            # Chuyển đổi tọa độ từ camera sang màn hình
            x3 = np.interp(x1, (frame_reduction, w_cam - frame_reduction), (0, w_scr))
            y3 = np.interp(y1, (frame_reduction, h_cam - frame_reduction), (0, h_scr))
            
            # Làm mượt chuyển động chuột
            c_loc_x = p_loc_x + (x3 - p_loc_x) / smoothening
            c_loc_y = p_loc_y + (y3 - p_loc_y) / smoothening
            
            try:
                pyautogui.moveTo(c_loc_x, c_loc_y)
            except Exception:
                pass
            
            p_loc_x, p_loc_y = c_loc_x, c_loc_y
            
            # Phát hiện cử chỉ Click (Chạm ngón trỏ và ngón cái)
            dist_pinch = math.hypot(x1 - x0, y1 - y0)
            if dist_pinch < 30:
                gesture = "CLICK TRAI!"
                pyautogui.click()
                cv2.circle(img, (x1, y1), 30, (0, 255, 255), cv2.FILLED)
                
            # Phát hiện cử chỉ Click Phải (Chạm ngón giữa và ngón cái)
            dist_right = math.hypot(x2 - x0, y2 - y0)
            if dist_right < 30:
                gesture = "CLICK PHAI!"
                pyautogui.rightClick()
                cv2.circle(img, (x2, y2), 30, (255, 0, 255), cv2.FILLED)

            # Vẽ các đường kết nối bàn tay
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
                (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
            ]
            
            for start, end in connections:
                pt1 = (lm_list[start][1], lm_list[start][2])
                pt2 = (lm_list[end][1], lm_list[end][2])
                color = (255, 0, 255) if start % 2 == 0 else (255, 255, 0)
                draw_neon_line(img, pt1, pt2, color)

        c_time = time.time()
        fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
        p_time = c_time
        
        # Chuyển đổi lại sang BGR để hiển thị bằng OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        draw_cyberpunk_ui(img_bgr, hands_count, gesture, fps)
        
        cv2.imshow("Cyberpunk Vision AI Hand Control - VN", img_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
