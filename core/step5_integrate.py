import cv2
import pickle
import numpy as np
import mediapipe as mp
import warnings
from ultralytics import YOLO

# Tắt cảnh báo của scikit-learn khi không truyền tên cột
warnings.filterwarnings("ignore", category=UserWarning)

def real_time_inference(source=0, model_path="core/vsl_model.pkl"):
    print("[INFO] Đang tải mô hình Trí tuệ nhân tạo (Random Forest)...")
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        print(f"[ERROR] Không thể tải model: {e}")
        return
        
    print("[INFO] Đang tải Camera và YOLOv8 & MediaPipe...")
    yolo_model = YOLO('yolov8n-pose.pt')
    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("[ERROR] Không thể mở Camera.")
        return
        
    print("[INFO] Bắt đầu nhận diện Real-time. Múa ký hiệu trước camera để xem AI dịch.")
    print("[INFO] Bấm 'q' trên cửa sổ video để thoát.")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Nếu dùng webcam, lật ảnh để dễ múa
        if source == 0:
            frame = cv2.flip(frame, 1)
            
        frame = cv2.resize(frame, (640, 480))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 1. Trích xuất Body
        results = yolo_model(frame, verbose=False)
        annotated_frame = results[0].plot()
        body_data = [0.0] * 51
        if results[0].keypoints is not None and results[0].keypoints.data is not None and len(results[0].keypoints.data) > 0:
            kpts_np = results[0].keypoints.data[0].cpu().numpy()
            body_data = []
            for kp in kpts_np: body_data.extend([kp[0], kp[1], kp[2]])
                
        # 2. Trích xuất Hands
        hand_results = hands_detector.process(rgb_frame)
        left_hand_data = [0.0] * 63
        right_hand_data = [0.0] * 63
        if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
            for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                mp_draw.draw_landmarks(annotated_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                coords = []
                for lm in hand_landmarks.landmark:
                    coords.extend([lm.x * 640, lm.y * 480, lm.z])
                if handedness.classification[0].label == "Left":
                    left_hand_data = coords
                else:
                    right_hand_data = coords
                    
        # 3. Gộp và Chuẩn hóa (Giống hệt Bước 3)
        features = body_data + left_hand_data + right_hand_data
        
        normalized_features = []
        for i in range(len(features)):
            if i % 3 == 0:   # Tọa độ X
                normalized_features.append(features[i] / 640.0)
            elif i % 3 == 1: # Tọa độ Y
                normalized_features.append(features[i] / 480.0)
            else:            # Độ tự tin / Z
                normalized_features.append(features[i])
                
        # Đưa thành mảng 2D cho model scikit-learn
        X_input = np.array(normalized_features).reshape(1, -1)
        
        # 4. NHẬN DIỆN TỪ VỰNG VÀ HIỂN THỊ
        predicted_label = model.predict(X_input)[0]
        confidence_scores = model.predict_proba(X_input)[0]
        max_conf = max(confidence_scores)
        
        # Chỉ hiển thị kết quả nếu độ tự tin > 40%
        if max_conf > 0.4:
            text = f"{predicted_label} ({max_conf*100:.1f}%)"
            # OpenCV không hỗ trợ hiển thị Tiếng Việt chuẩn nên có thể bị lỗi font nhẹ,
            # Ta dùng font cơ bản hiển thị tạm nghiệm thu MVP
            cv2.putText(annotated_frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        cv2.imshow("Buoc 5: Nhan dien Real-time VSL", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_inference()
