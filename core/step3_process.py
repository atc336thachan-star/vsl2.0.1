import pandas as pd
import numpy as np
import os

def process_data(input_csv="data/processed_data/skeleton_data.csv", 
                 output_csv="data/processed_data/skeleton_data_normalized.csv"):
    
    if not os.path.exists(input_csv):
        print(f"[ERROR] Không tìm thấy file {input_csv}.")
        return
        
    print(f"[INFO] Đang đọc dữ liệu từ: {input_csv}")
    df = pd.read_csv(input_csv)
    
    print(f"[INFO] Số lượng frame thu thập được: {len(df)}")
    print(f"[INFO] Tổng số cột: {len(df.columns)}")
    
    # ---------------------------------------------------------
    # 1. Xử lý giá trị lỗi / nhiễu
    # ---------------------------------------------------------
    df.fillna(0.0, inplace=True)
    
    # ---------------------------------------------------------
    # 2. Chuẩn hóa dữ liệu (Normalization)
    # ---------------------------------------------------------
    WIDTH = 640.0
    HEIGHT = 480.0
    
    print("[INFO] Đang chuẩn hóa tọa độ (đưa về khoảng 0.0 -> 1.0)...")
    
    cols_x = [col for col in df.columns if col.endswith('_x')]
    cols_y = [col for col in df.columns if col.endswith('_y')]
    
    df[cols_x] = df[cols_x] / WIDTH
    df[cols_y] = df[cols_y] / HEIGHT
    
    # ---------------------------------------------------------
    # 3. Lưu dữ liệu
    # ---------------------------------------------------------
    df.to_csv(output_csv, index=False)
    print(f"[INFO] Đã lưu dữ liệu chuẩn hóa thành công tại: {output_csv}")
    print("-" * 50)
    print(df[['label', 'frame_id', 'body_0_x', 'left_hand_0_x']].head())

if __name__ == "__main__":
    process_data()
