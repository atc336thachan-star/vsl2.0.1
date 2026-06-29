import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Không hiển thị cửa sổ vẽ, chạy ngầm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from scipy.interpolate import CubicSpline

# Không vẽ các điểm trên mặt thành đường BODY nữa để vẽ riêng
BODY_LINKS = [(5,7), (7,9), (6,8), (8,10), (5,6), (5,11), (6,12), (11,12), (11,13), (13,15), (12,14), (14,16)]
FACE_LINKS = [(0,1), (1,3), (0,2), (2,4), (1,2)]
HAND_LINKS = [(0,1),(1,2),(2,3),(3,4), (0,5),(5,6),(6,7),(7,8), (5,9),(9,10),(10,11),(11,12), (9,13),(13,14),(14,15),(15,16), (13,17),(17,18),(18,19),(19,20), (0,17)]

# Bộ nhớ đệm ở cấp độ module để lưu trữ dữ liệu khung xương nhằm tránh việc đọc lại file CSV 482MB nhiều lần
_df_cache = None

def get_skeleton_data(csv_path):
    global _df_cache
    if _df_cache is None:
        print(f"[INFO] Loading skeleton data from {csv_path} (482MB, chỉ thực hiện một lần)...")
        _df_cache = pd.read_csv(csv_path)
        print("[INFO] Skeleton data loaded into memory successfully.")
    return _df_cache

def generate_vsl_animation(text_or_list, csv_path="data/processed_data/skeleton_data.csv", output_file="output.gif", speed=1.0):
    """
    Tìm kiếm từ vựng trong file CSV và vẽ lại animation khung xương thành file GIF/MP4.
    Hỗ trợ ghép câu dài và nội suy khung hình chuyển tiếp (Interpolation).
    Trả về True nếu thành công, False nếu không tìm thấy từ vựng.
    """
    # Vì ứng dụng gọi từ vsl_web, cần dùng đường dẫn tuyệt đối cho an toàn
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_csv_path = os.path.join(base_dir, csv_path)
    
    if not os.path.exists(full_csv_path):
        print(f"[ERROR] Không tìm thấy file dữ liệu: {full_csv_path}")
        return False
        
    df = get_skeleton_data(full_csv_path)
    
    # Xử lý tham số đầu vào
    if isinstance(text_or_list, str):
        words = [text_or_list]
    else:
        words = text_or_list
        
    if not words:
        return False
        
    all_seqs = []
    
    for word in words:
        # Tìm nhãn (không phân biệt chữ hoa chữ thường)
        df_match = df[df['label'].str.lower() == word.lower()]
        if df_match.empty:
            continue
            
        # Lấy 1 chuỗi liên tục (1 video) đầu tiên
        seq = []
        prev_frame_id = -1
        for idx, row in df_match.iterrows():
            if row['frame_id'] == 0 and prev_frame_id != -1:
                break # Hết 1 video thì dừng
            seq.append(row)
            prev_frame_id = row['frame_id']
            
        if seq:
            all_seqs.append(pd.DataFrame(seq))
            
    if not all_seqs:
        return False
        
    # Ghép chuỗi và Áp dụng Nội suy (Interpolation)
    final_seq = []
    for i in range(len(all_seqs)):
        seq = all_seqs[i]
        final_seq.append(seq)
        
        # Nếu chưa phải từ cuối cùng, tạo frames chuyển tiếp tới từ tiếp theo
        if i < len(all_seqs) - 1:
            next_seq = all_seqs[i+1]
            
            num_interp = 10 # 10 khung hình chuyển tiếp (mượt hơn)
            coord_cols = [c for c in seq.columns if c not in ['label', 'frame_id']]
            
            n_pad = min(2, len(seq), len(next_seq))
            if n_pad >= 2:
                # Cubic Spline Interpolation: Lấy 2 frame trước, 2 frame sau
                y_points = []
                y_points.append(seq.iloc[-2][coord_cols].values.astype(float))
                y_points.append(seq.iloc[-1][coord_cols].values.astype(float))
                y_points.append(next_seq.iloc[0][coord_cols].values.astype(float))
                y_points.append(next_seq.iloc[1][coord_cols].values.astype(float))
                y_points = np.array(y_points)
                
                x_points = np.array([-1, 0, num_interp + 1, num_interp + 2])
                cs = CubicSpline(x_points, y_points, axis=0, bc_type='natural')
                
                x_interp = np.arange(1, num_interp + 1)
                y_interp = cs(x_interp)
                
                interp_frames = []
                for j in range(num_interp):
                    new_frame = seq.iloc[-1].copy()
                    new_frame['label'] = 'Transition'
                    new_frame['frame_id'] = -1
                    new_frame[coord_cols] = y_interp[j]
                    interp_frames.append(new_frame)
                final_seq.append(pd.DataFrame(interp_frames))
            else:
                # Fallback: Cubic Smoothstep Hermite
                last_frame = seq.iloc[-1].copy()
                first_frame = next_seq.iloc[0].copy()
                interp_frames = []
                for j in range(1, num_interp + 1):
                    alpha = j / (num_interp + 1)
                    alpha_smooth = alpha * alpha * (3 - 2 * alpha)
                    new_frame = last_frame.copy()
                    new_frame['label'] = 'Transition'
                    new_frame['frame_id'] = -1
                    for col in coord_cols:
                        new_frame[col] = last_frame[col] * (1 - alpha_smooth) + first_frame[col] * alpha_smooth
                    interp_frames.append(new_frame)
                final_seq.append(pd.DataFrame(interp_frames))
            
    # Gộp tất cả lại thành 1 DataFrame hoàn chỉnh
    df_seq = pd.concat(final_seq, ignore_index=True)
    
    # Tự động tính toán Bounding Box tối ưu cho chuỗi chuyển động hiện tại (Dynamic Zoom)
    # Loại bỏ chân (body landmarks 13, 14, 15, 16) khỏi Y-bounds để không lộ quá nửa chân
    x_cols = [c for c in df_seq.columns if c.endswith('_x') and c != 'frame_id']
    y_cols = [c for c in df_seq.columns if c.endswith('_y') and c != 'frame_id']
    y_cols_no_legs = [c for c in y_cols if not any(f'body_{i}_' in c for i in [13, 14, 15, 16])]
    
    valid_xs = df_seq[x_cols].values
    valid_xs = valid_xs[valid_xs > 0]
    valid_ys_no_legs = df_seq[y_cols_no_legs].values
    valid_ys_no_legs = valid_ys_no_legs[valid_ys_no_legs > 0]
    
    if len(valid_xs) > 0 and len(valid_ys_no_legs) > 0:
        seq_min_x, seq_max_x = np.min(valid_xs), np.max(valid_xs)
        seq_min_y, seq_max_y = np.min(valid_ys_no_legs), np.max(valid_ys_no_legs)
    else:
        # Fallback to dataset bounds if no valid coordinates found
        seq_min_x, seq_max_x = 40.0, 580.0
        seq_min_y, seq_max_y = -25.0, 480.0
        
    width = seq_max_x - seq_min_x
    height = seq_max_y - seq_min_y
    
    center_x = (seq_min_x + seq_max_x) / 2
    
    # Kích thước hình vuông bao quanh thân trên (giới hạn tối thiểu 320)
    side = max(width, height, 320)
    
    # Đẩy thân người sát xuống dưới cùng (chừa 4% khoảng đệm dưới cùng cho tay)
    pad_bottom = side * 0.04
    view_bottom_y = seq_max_y + pad_bottom
    view_top_y = view_bottom_y - side
    
    view_min_x = center_x - side / 2
    view_max_x = center_x + side / 2
    
    # Thiết lập Matplotlib với nền đen (black background) - Giảm kích thước xuống 4x4, dpi=80 để tăng tốc độ vẽ ảnh (render GIF) gấp 3-4 lần
    fig, ax = plt.subplots(figsize=(4, 4), facecolor='black', dpi=80)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    
    ax.set_facecolor('black')
    
    # Ép tỉ lệ 1:1 để cơ thể không bị bóp méo
    ax.set_aspect('equal', adjustable='datalim')
    
    # Giới hạn trục ngang
    ax.set_xlim(view_min_x, view_max_x)
    
    # Giới hạn trục dọc. Hệ tọa độ ảnh: y lộn ngược
    ax.set_ylim(view_bottom_y, view_top_y)
    
    ax.axis('off') # Ẩn trục tọa độ
    
    # Chuẩn bị các nét vẽ (tăng độ dày nét vẽ)
    body_lines = [ax.plot([], [], '#00bfff', lw=5, solid_capstyle='round')[0] for _ in BODY_LINKS]
    face_lines = [ax.plot([], [], '#00bfff', lw=4, solid_capstyle='round')[0] for _ in FACE_LINKS]
    # Trả về độ dày nét vẽ bàn tay ban đầu (lw=3)
    lh_lines = [ax.plot([], [], '#ff4d4d', lw=3, solid_capstyle='round')[0] for _ in HAND_LINKS]
    rh_lines = [ax.plot([], [], '#33cc33', lw=3, solid_capstyle='round')[0] for _ in HAND_LINKS]
    
    # Chuẩn bị vẽ khớp và khuôn mặt (scatter)
    face_scatter = ax.scatter([], [], c='#ffff00', s=80, zorder=5, edgecolors='black', lw=1)
    joint_scatter = ax.scatter([], [], c='white', s=40, zorder=4)
    
    def update(frame_idx):
        row = df_seq.iloc[frame_idx]
        
        # Vẽ Body
        joint_xs = []
        joint_ys = []
        for i, (p1, p2) in enumerate(BODY_LINKS):
            x1, y1 = row[f'body_{p1}_x'], row[f'body_{p1}_y']
            x2, y2 = row[f'body_{p2}_x'], row[f'body_{p2}_y']
            if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                body_lines[i].set_data([x1, x2], [y1, y2])
                joint_xs.extend([x1, x2])
                joint_ys.extend([y1, y2])
            else:
                body_lines[i].set_data([], [])
                
        # Vẽ Face Links
        face_xs = []
        face_ys = []
        for i, (p1, p2) in enumerate(FACE_LINKS):
            x1, y1 = row[f'body_{p1}_x'], row[f'body_{p1}_y']
            x2, y2 = row[f'body_{p2}_x'], row[f'body_{p2}_y']
            if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                face_lines[i].set_data([x1, x2], [y1, y2])
            else:
                face_lines[i].set_data([], [])
                
        # Thêm điểm trên khuôn mặt (mắt, mũi, tai)
        for i in range(5):
            x, y = row[f'body_{i}_x'], row[f'body_{i}_y']
            if x > 0 and y > 0:
                face_xs.append(x)
                face_ys.append(y)
                
        # Tỉ lệ phóng to 2 bàn tay so với cơ thể (trả về tỉ lệ ban đầu 1.0x)
        hand_scale = 1.0

        # Vẽ Tay Trái (Vẽ left_hand của CSV được dịch chuyển tương ứng lên cổ tay phải right_hand_0 ở bên trái màn hình, màu đỏ)
        left_wrist_x = row['right_hand_0_x']
        left_wrist_y = row['right_hand_0_y']
        src_wrist_x = row['left_hand_0_x']
        src_wrist_y = row['left_hand_0_y']
        
        for i, (p1, p2) in enumerate(HAND_LINKS):
            x1, y1 = row[f'left_hand_{p1}_x'], row[f'left_hand_{p1}_y']
            x2, y2 = row[f'left_hand_{p2}_x'], row[f'left_hand_{p2}_y']
            if (x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0 and 
                src_wrist_x > 0 and src_wrist_y > 0 and 
                left_wrist_x > 0 and left_wrist_y > 0):
                # Phóng to khoảng cách từ các khớp bàn tay tới cổ tay theo hand_scale
                x1_new = left_wrist_x + hand_scale * (x1 - src_wrist_x)
                y1_new = left_wrist_y + hand_scale * (y1 - src_wrist_y)
                x2_new = left_wrist_x + hand_scale * (x2 - src_wrist_x)
                y2_new = left_wrist_y + hand_scale * (y2 - src_wrist_y)
                lh_lines[i].set_data([x1_new, x2_new], [y1_new, y2_new])
            else:
                lh_lines[i].set_data([], [])
                
        # Vẽ Tay Phải (Vẽ right_hand của CSV được dịch chuyển tương ứng lên cổ tay trái left_hand_0 ở bên phải màn hình, màu xanh lá)
        right_wrist_x = row['left_hand_0_x']
        right_wrist_y = row['left_hand_0_y']
        src_wrist_rx = row['right_hand_0_x']
        src_wrist_ry = row['right_hand_0_y']
        
        for i, (p1, p2) in enumerate(HAND_LINKS):
            x1, y1 = row[f'right_hand_{p1}_x'], row[f'right_hand_{p1}_y']
            x2, y2 = row[f'right_hand_{p2}_x'], row[f'right_hand_{p2}_y']
            if (x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0 and 
                src_wrist_rx > 0 and src_wrist_ry > 0 and 
                right_wrist_x > 0 and right_wrist_y > 0):
                # Phóng to khoảng cách từ các khớp bàn tay tới cổ tay theo hand_scale
                x1_new = right_wrist_x + hand_scale * (x1 - src_wrist_rx)
                y1_new = right_wrist_y + hand_scale * (y1 - src_wrist_ry)
                x2_new = right_wrist_x + hand_scale * (x2 - src_wrist_rx)
                y2_new = right_wrist_y + hand_scale * (y2 - src_wrist_ry)
                rh_lines[i].set_data([x1_new, x2_new], [y1_new, y2_new])
            else:
                rh_lines[i].set_data([], [])
                
        if face_xs:
            face_scatter.set_offsets(np.c_[face_xs, face_ys])
        else:
            face_scatter.set_offsets(np.empty((0, 2)))
            
        if joint_xs:
            joint_scatter.set_offsets(np.c_[joint_xs, joint_ys])
        else:
            joint_scatter.set_offsets(np.empty((0, 2)))
                
        return body_lines + face_lines + lh_lines + rh_lines + [face_scatter, joint_scatter]
        
    # Tính toán interval dựa trên tốc độ
    # Chuẩn là 40ms (25fps). Nếu speed=0.5 (chậm), interval=80. Nếu speed=1.5 (nhanh), interval=26.
    interval_ms = int(40 / speed)
    ani = animation.FuncAnimation(fig, update, frames=len(df_seq), blit=True, interval=interval_ms)
    
    # Tạo thư mục chứa output nếu chưa có
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    # Lưu dạng GIF để đảm bảo tương thích mọi trình duyệt web mà không cần FFmpeg
    ani.save(output_file, writer='pillow')
    plt.close(fig)
    return True

if __name__ == "__main__":
    # Test thử
    success = generate_vsl_animation("Xin_Chao", output_file="test_xin_chao.gif")
    if success:
        print("[OK] Đã tạo thành công GIF cho chữ 'Xin_Chao'")
    else:
        print("[FAILED] Không tạo được GIF.")
