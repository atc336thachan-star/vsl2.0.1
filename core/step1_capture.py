import cv2
import time
from ultralytics import YOLO

def start_video_capture(source=0):
    print("[INFO] Đang khởi tạo mô hình YOLOv8 Pose...")
    # Tự động tải yolov8n-pose.pt (phiên bản nhẹ, nhanh nhất) trong lần chạy đầu tiên
    model = YOLO('yolov8n-pose.pt')
    
    # Mở webcam (source=0) hoặc truyền đường dẫn file video (source="video.mp4")
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print("[ERROR] Không thể mở camera hoặc file video.")
        return
        
    print("[INFO] Đã bật camera. Bấm phím 'q' trên cửa sổ để thoát.")
    prev_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Không nhận được frame. Kết thúc luồng.")
            break
            
        # Lật ngược hình ảnh (như soi gương) nếu dùng webcam cho dễ thao tác
        if source == 0:
            frame = cv2.flip(frame, 1)
            
        # Resize để đảm bảo tốc độ xử lý ổn định
        frame = cv2.resize(frame, (640, 480))
        
        # Đưa frame vào YOLOv8 để xử lý (verbose=False để không in log quá nhiều ra terminal)
        results = model(frame, verbose=False)
        
        # Vẽ các điểm mốc (keypoints) và khung xương lên frame
        annotated_frame = results[0].plot()
        
        # Tính toán FPS để kiểm tra hiệu năng
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if curr_time - prev_time > 0 else 0
        prev_time = curr_time
        
        # In FPS lên góc màn hình
        cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Hiển thị
        cv2.imshow("Buoc 1: Tien xu ly Video & YOLOv8 Pose", annotated_frame)
        
        # Điều kiện thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Người dùng đã đóng ứng dụng.")
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_video_capture()
