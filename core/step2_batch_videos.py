import cv2
import csv
import os
import mediapipe as mp
from ultralytics import YOLO

def process_all_videos(video_dir="data/raw_videos", output_csv="data/processed_data/skeleton_data.csv"):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    
    # Lấy danh sách video trong thư mục
    video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    if not video_files:
        print(f"[ERROR] Không tìm thấy video nào. Bạn hãy copy các file video vào thư mục: {video_dir}")
        print("Ví dụ: data/raw_videos/Xin_Chao.mp4, data/raw_videos/Tam_Biet.mp4")
        return
        
    print(f"[INFO] Tìm thấy {len(video_files)} video. Bắt đầu trích xuất hàng loạt...")
    
    # TẠO HEADER (Lần này có thêm cột 'label' ở vị trí đầu tiên)
    headers = ["label", "frame_id"]
    for i in range(17): headers.extend([f"body_{i}_x", f"body_{i}_y", f"body_{i}_conf"])
    for i in range(21): headers.extend([f"left_hand_{i}_x", f"left_hand_{i}_y", f"left_hand_{i}_z"])
    for i in range(21): headers.extend([f"right_hand_{i}_x", f"right_hand_{i}_y", f"right_hand_{i}_z"])
    
    model = YOLO('yolov8n-pose.pt')
    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
    
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for video_file in video_files:
            # Nhãn (label) chính là tên file (ví dụ: Xin_Chao.mp4 -> label: Xin_Chao)
            label = os.path.splitext(video_file)[0]
            video_path = os.path.join(video_dir, video_file)
            print(f"\n[INFO] Đang xử lý video: {video_path} -> Gán nhãn: {label}")
            
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                frame = cv2.resize(frame, (640, 480))
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # YOLO Body
                results = model(frame, verbose=False)
                body_data = [0.0] * 51
                if results[0].keypoints is not None and results[0].keypoints.data is not None and len(results[0].keypoints.data) > 0:
                    kpts_np = results[0].keypoints.data[0].cpu().numpy()
                    body_data = []
                    for kp in kpts_np: body_data.extend([kp[0], kp[1], kp[2]])
                        
                # MediaPipe Hands
                hand_results = hands_detector.process(rgb_frame)
                left_hand_data = [0.0] * 63
                right_hand_data = [0.0] * 63
                if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
                    for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                        coords = []
                        for lm in hand_landmarks.landmark:
                            coords.extend([lm.x * 640, lm.y * 480, lm.z])
                        if handedness.classification[0].label == "Left":
                            left_hand_data = coords
                        else:
                            right_hand_data = coords
                            
                # Ghi vào CSV (Cột đầu tiên là label)
                row_data = [label, frame_count] + body_data + left_hand_data + right_hand_data
                writer.writerow(row_data)
                frame_count += 1
                
            cap.release()
            print(f"[OK] Đã trích xuất {frame_count} frames cho nhãn '{label}'.")
            
    print(f"\n[INFO] Hoàn tất! Tất cả dữ liệu thực tế đã được gộp và lưu tại: {output_csv}")

if __name__ == "__main__":
    process_all_videos()
