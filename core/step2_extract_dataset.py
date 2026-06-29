import cv2
import csv
import os
import json
import mediapipe as mp
from ultralytics import YOLO

def process_dataset(video_dir="d:/split_1/front_view", 
                    json_path="d:/split_1/front_view.json", 
                    output_csv="data/processed_data/skeleton_data.csv",
                    word_limit=100):
                    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    print(f"[INFO] Đang đọc file metadata: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
        
    # Tạo từ điển tra cứu nhanh video_id -> gloss (nhãn)
    label_map = {}
    for item in metadata:
        label_map[item['video_id']] = item['gloss']
        
    print(f"[INFO] Lấy được {len(label_map)} nhãn từ file JSON.")
    
    # ---------------------------------------------
    # TẠO HEADER
    # ---------------------------------------------
    headers = ["label", "frame_id"]
    for i in range(17): headers.extend([f"body_{i}_x", f"body_{i}_y", f"body_{i}_conf"])
    for i in range(21): headers.extend([f"left_hand_{i}_x", f"left_hand_{i}_y", f"left_hand_{i}_z"])
    for i in range(21): headers.extend([f"right_hand_{i}_x", f"right_hand_{i}_y", f"right_hand_{i}_z"])
    
    print("[INFO] Đang tải AI Model (YOLO & MediaPipe)...")
    model = YOLO('yolov8n-pose.pt')
    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
    
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    
    # Giới hạn số lượng TỪ VỰNG thay vì số lượng video
    if word_limit is not None:
        selected_videos = []
        seen_labels = set()
        for f in video_files:
            v_id = os.path.splitext(f)[0]
            if v_id not in label_map: continue
            label = label_map[v_id]
            
            if label not in seen_labels:
                if len(seen_labels) >= word_limit:
                    continue # Bỏ qua từ vựng mới nếu đã đủ chỉ tiêu
                seen_labels.add(label)
                
            selected_videos.append(f)
            
        video_files = selected_videos
        print(f"[WARNING] Chế độ học {word_limit} từ vựng: Chạy trích xuất {len(video_files)} video liên quan.")
        
    print(f"[INFO] Bắt đầu trích xuất {len(video_files)} video...")
    
    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        count = 0
        for video_file in video_files:
            video_id = os.path.splitext(video_file)[0]
            
            # Bỏ qua nếu video_id không có trong file json
            if video_id not in label_map:
                continue
                
            label = label_map[video_id]
            video_path = os.path.join(video_dir, video_file)
            
            print(f"[{count+1}/{len(video_files)}] Xử lý: {video_file} -> Nhãn: {label}")
            
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                # Resize để tăng tốc độ
                frame = cv2.resize(frame, (640, 480))
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Trích xuất Body (YOLO)
                results = model(frame, verbose=False)
                body_data = [0.0] * 51
                if results[0].keypoints is not None and results[0].keypoints.data is not None and len(results[0].keypoints.data) > 0:
                    kpts_np = results[0].keypoints.data[0].cpu().numpy()
                    body_data = []
                    for kp in kpts_np: body_data.extend([kp[0], kp[1], kp[2]])
                        
                # Trích xuất Hands (MediaPipe)
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
                            
                # Ghi vào CSV
                row_data = [label, frame_count] + body_data + left_hand_data + right_hand_data
                writer.writerow(row_data)
                frame_count += 1
                
            cap.release()
            count += 1
            
    print(f"\n[INFO] Hoàn tất trích xuất dataset! Toàn bộ dữ liệu lưu tại: {output_csv}")

if __name__ == "__main__":
    # Học thêm 200 từ vựng nữa (Tổng cộng 300 từ)
    process_dataset(word_limit=300)
