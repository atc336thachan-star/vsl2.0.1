import os
import sys
import unicodedata
from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from .models import Vocab

# Thêm thư mục cha (VSL2.0) vào Python Path để import core
BASE_DIR = settings.BASE_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from core.step6_text_to_animation import generate_vsl_animation
from .qwen_translator import QwenTranslator

# Khởi tạo translator
qwen_translator = QwenTranslator()

def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])

def tokenize_vietnamese(sentence):
    # Chuẩn hóa câu nhập
    sentence = remove_accents(sentence).lower()
    words = sentence.split()
    
    # Lấy danh sách từ vựng từ DB
    vocab_objs = Vocab.objects.all()
    vocab_map = {}
    for v in vocab_objs:
        norm_key = remove_accents(v.word).replace('_', ' ').lower()
        vocab_map[norm_key] = v.word
    
    # Thuật toán Maximum Forward Matching
    tokens = []
    i = 0
    while i < len(words):
        match_found = False
        # Thử ghép cụm từ tối đa 4 từ đơn
        for j in range(min(len(words), i + 4), i, -1):
            phrase = " ".join(words[i:j])
            if phrase in vocab_map:
                tokens.append(vocab_map[phrase])
                i = j
                match_found = True
                break
        if not match_found:
            # Bỏ qua từ không có trong từ điển
            i += 1
            
    return tokens

def index(request):
    return render(request, 'index.html')

def dictionary(request):
    vocabs = Vocab.objects.all().order_by('word')
    # Thêm thuộc tính hiển thị (thay _ bằng khoảng trắng)
    for v in vocabs:
        v.display_word = v.word.replace('_', ' ')
    return render(request, 'dictionary.html', {'vocabs': vocabs})

from django.views.decorators.csrf import csrf_exempt

import hashlib

@csrf_exempt
def translate_text(request):
    if request.method == 'POST':
        text = request.POST.get('text', '').strip()
        if not text:
            return JsonResponse({'success': False, 'error': 'Vui lòng nhập từ vựng'})
            
        # Lấy tốc độ
        speed_str = request.POST.get('speed', '1.0')
        try:
            speed = float(speed_str)
        except ValueError:
            speed = 1.0
            
        # Thử dịch câu bằng Qwen LLM
        matched_tokens = []
        
        # Đường truyền nhanh (Fast Path): So khớp trực tiếp từ/cụm từ trong từ điển trước
        norm_text = remove_accents(text).lower()
        vocab_objs = Vocab.objects.all()
        vocab_map = {}
        for v in vocab_objs:
            norm_key = remove_accents(v.word).replace('_', ' ').lower()
            vocab_map[norm_key] = v.word
            
        if norm_text in vocab_map:
            matched_tokens = [vocab_map[norm_text]]
            print(f"[FAST PATH] Khớp trực tiếp trong từ điển cho '{text}': {matched_tokens}")
            
        if not matched_tokens and getattr(settings, 'QWEN_MODE', 'none') != 'none':
            try:
                matched_tokens = qwen_translator.translate(text)
            except Exception as e:
                print(f"[QWEN] View translation error: {e}")
                
        # Cơ chế Fallback: Nếu Qwen lỗi hoặc không ra kết quả, quay về MFM thô
        if not matched_tokens:
            print("[QWEN] Falling back to MFM (Maximum Forward Matching)...")
            matched_tokens = tokenize_vietnamese(text)
            
        if not matched_tokens:
            return JsonResponse({'success': False, 'error': f'Xin lỗi, không nhận diện được từ vựng nào trong câu: "{text}"'})
            
        # Tạo tên file bằng mã băm để tránh tên file quá dài nếu câu nhiều từ
        tokens_str = "_".join(matched_tokens)
        hash_str = hashlib.md5(tokens_str.encode()).hexdigest()[:10]
        filename = f"sentence_{hash_str}_{speed}x.gif"
        
        # Đường dẫn vật lý lưu file
        media_anim_dir = os.path.join(settings.MEDIA_ROOT, 'animations')
        os.makedirs(media_anim_dir, exist_ok=True)
        file_path = os.path.join(media_anim_dir, filename)
        
        # Đường dẫn trả về cho frontend
        file_url = f"{settings.MEDIA_URL}animations/{filename}"
        
        # Dùng cơ chế Cache: Nếu từ vựng đã từng được dịch và tạo GIF rồi, trả về luôn để tăng tốc độ (Real-time cảm giác)
        if os.path.exists(file_path):
            return JsonResponse({'success': True, 'video_url': file_url, 'word': text})
            
        # Nếu chưa có, tiến hành generate từ file CSV
        csv_path = "data/processed_data/skeleton_data.csv"
        
        try:
            # Truyền mảng các từ khóa thay vì 1 chuỗi
            success = generate_vsl_animation(matched_tokens, csv_path=csv_path, output_file=file_path, speed=speed)
            if success:
                # Trả về text gốc (vd: Xin chào bác) để hiển thị đẹp trên UI
                return JsonResponse({'success': True, 'video_url': file_url, 'word': text})
            else:
                return JsonResponse({'success': False, 'error': f'Lỗi khi tạo animation cho câu: "{text}"'})
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
            
    return JsonResponse({'success': False, 'error': 'Method not allowed'})
