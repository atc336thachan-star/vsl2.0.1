import cv2
import csv
import os
import mediapipe as mp
from ultralytics import YOLO

def extract_features(source=0, output_csv="data/processed_data/skeleton_data.csv"):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # --- TẠO HEADER CHO CSV ---
    headers = ["frame_id"]
    # 1. Body keypoints (17 điểm từ YOLO: x, y, conf)
    for i in range(17):
        headers.extend([f"body_{i}_x", f"body_{i}_y", f"body_{i}_conf"])
        
    # 2. Hand keypoints (21 điểm/tay từ MediaPipe: x, y, z)
    for i in range(21):
        headers.extend([f"left_hand_{i}_x", f"left_hand_{i}_y", f"left_hand_{i}_z"])
    for i in range(21):
        headers.extend([f"right_hand_{i}_x", f"right_hand_{i}_y", f"right_hand_{i}_z"])
        
    print(f"[INFO] Dữ liệu sẽ được lưu tại: {output_csv}")
    
    # --- KHỞI TẠO MÔ HÌNH ---
    print("[INFO] Đang tải mô hình YOLOv8 Pose (Body)...")
    model = YOLO('yolov8n-pose.pt')
    
    print("[INFO] Đang tải mô hình MediaPipe Hands (Bàn tay)...")
    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils
    
    # --- MỞ CAMERA ---
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("[ERROR] Không thể mở camera.")
        return
        
    print("[INFO] Bắt đầu trích xuất Body + Hands. Bấm phím 'q' trên cửa sổ để kết thúc.")
    
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if source == 0:
                frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (640, 480))
            
            # Khung hình dùng để tính MediaPipe cần hệ màu RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # ---------------------------------------------
            # 1. Xử lý Khung xương Cơ thể bằng YOLOv8
            # ---------------------------------------------
            results = model(frame, verbose=False)
            annotated_frame = results[0].plot() # YOLO vẽ khung xương body
            
            # Khởi tạo mặc định nếu không thấy người
            body_data = [0.0] * (17 * 3) 
            keypoints = results[0].keypoints
            if keypoints is not None and keypoints.data is not None and len(keypoints.data) > 0:
                kpts_np = keypoints.data[0].cpu().numpy()
                body_data = []
                for kp in kpts_np:
                    body_data.extend([kp[0], kp[1], kp[2]])
            
            # ---------------------------------------------
            # 2. Xử lý Bàn tay bằng MediaPipe
            # ---------------------------------------------
            hand_results = hands_detector.process(rgb_frame)
            
            # Khởi tạo mặc định nếu không thấy tay
            left_hand_data = [0.0] * (21 * 3)
            right_hand_data = [0.0] * (21 * 3)
            
            if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
                for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                    # Vẽ xương bàn tay lên ảnh
                    mp_draw.draw_landmarks(annotated_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Trích xuất 21 điểm x,y,z
                    coords = []
                    for lm in hand_landmarks.landmark:
                        # Nhân với kích thước frame (640x480) để ra pixel thực tế
                        coords.extend([lm.x * 640, lm.y * 480, lm.z])
                        
                    label = handedness.classification[0].label # "Left" hoặc "Right"
                    if label == "Left":
                        left_hand_data = coords
                    else:
                        right_hand_data = coords
            
            # ---------------------------------------------
            # 3. Gộp và Ghi vào CSV
            # ---------------------------------------------
            row_data = [frame_count] + body_data + left_hand_data + right_hand_data
            writer.writerow(row_data)
            
            # Cập nhật hiển thị
            cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Buoc 2: Body (YOLO) + Hands (MediaPipe)", annotated_frame)
            
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Người dùng đã dừng quá trình.")
                break
                
    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Hoàn tất! Đã thu thập {frame_count} frames chứa cả BODY và BÀN TAY.")
    print(f"[INFO] File CSV được lưu tại: {output_csv}")

if __name__ == "__main__":
    extract_features()
