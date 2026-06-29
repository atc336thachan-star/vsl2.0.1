import os
import sys
import django
import pandas as pd

# Thiết lập môi trường Django
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'vsl_web.settings')
django.setup()

from app_translate.models import Vocab

def import_vocab_from_csv():
    csv_path = r"d:\VSL2.0\data\processed_data\skeleton_data.csv"
    if not os.path.exists(csv_path):
        print(f"[LỖI] File không tồn tại: {csv_path}")
        return
        
    print("Đang đọc dữ liệu AI từ CSV...")
    df = pd.read_csv(csv_path)
    
    # Lấy danh sách các nhãn (từ vựng) duy nhất
    unique_words = df['label'].dropna().unique()
    
    print(f"Phát hiện {len(unique_words)} từ vựng AI đã học. Bắt đầu cập nhật vào Database...")
    
    added_count = 0
    for word in unique_words:
        # Bỏ qua nếu từ trống
        word_str = str(word).strip()
        if not word_str:
            continue
            
        # Tạo mới nếu chưa có
        obj, created = Vocab.objects.get_or_create(word=word_str)
        if created:
            added_count += 1
            print(f" -> Đã thêm mới: {word_str}")
            
    print(f"\n[HOÀN TẤT] Đã cập nhật thành công. Thêm mới {added_count} từ vựng vào bảng Vocab trong Models.")

if __name__ == "__main__":
    import_vocab_from_csv()
