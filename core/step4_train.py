import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_model(input_csv="data/processed_data/skeleton_data_normalized.csv",
                model_out="core/vsl_model.pkl"):
                
    if not os.path.exists(input_csv):
        print(f"[ERROR] Không tìm thấy file {input_csv}.")
        return
        
    print("[INFO] Đang tải dữ liệu chuẩn hóa...")
    df = pd.read_csv(input_csv)
    
    if 'label' not in df.columns:
        print("[ERROR] Dữ liệu thiếu cột 'label'. Vui lòng dùng script trích xuất video hàng loạt để lấy nhãn.")
        return
        
    # ---------------------------------------------------------
    # CHUẨN BỊ DỮ LIỆU HUẤN LUYỆN
    # ---------------------------------------------------------
    # X (Features): Lấy tất cả tọa độ (bỏ label và frame_id)
    X = df.drop(columns=['label', 'frame_id'])
    # y (Target): Cột nhãn cần dự đoán
    y = df['label']
    
    print(f"[INFO] Tổng số mẫu: {X.shape[0]} frames.")
    print(f"[INFO] Số lượng từ vựng (classes): {len(y.unique())}")
    print(f"       Danh sách: {y.unique()}")
    
    # Chia dữ liệu: 80% Train, 20% Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # ---------------------------------------------------------
    # HUẤN LUYỆN MÔ HÌNH BẰNG RANDOM FOREST
    # ---------------------------------------------------------
    print("[INFO] Khởi tạo AI: Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    print("[INFO] Đang tiến hành huấn luyện (Training) trên dữ liệu THỰC TẾ...")
    model.fit(X_train, y_train)
    
    # ---------------------------------------------------------
    # ĐÁNH GIÁ MÔ HÌNH
    # ---------------------------------------------------------
    print("[INFO] Đang làm bài test đánh giá (Testing)...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*50)
    print(f"[KẾT QUẢ THÀNH CÔNG] Độ chính xác (Accuracy): {acc * 100:.2f}%")
    print("Báo cáo chi tiết từng nhãn:")
    print(classification_report(y_test, y_pred))
    print("="*50 + "\n")
    
    # ---------------------------------------------------------
    # LƯU MÔ HÌNH ĐỂ SỬ DỤNG CHO BƯỚC 5 (TÍCH HỢP)
    # ---------------------------------------------------------
    with open(model_out, 'wb') as f:
        pickle.dump(model, f)
    print(f"[INFO] Đã xuất mô hình AI ra file: {model_out}")

if __name__ == "__main__":
    train_model()
