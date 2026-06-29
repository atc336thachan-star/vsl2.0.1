import os
import json
import re
import requests
from django.conf import settings
from .models import Vocab

class QwenTranslator:
    def __init__(self):
        self.mode = getattr(settings, 'QWEN_MODE', 'api')
        self.model_name = getattr(settings, 'QWEN_MODEL_NAME', 'Qwen/Qwen2.5-7B-Instruct')
        self.api_token = getattr(settings, 'HF_API_TOKEN', '')
        
        # Load dictionary from Database
        self.allowed_words = self._get_allowed_words()
        
    def _get_allowed_words(self):
        try:
            vocabs = Vocab.objects.all()
            return [v.word for v in vocabs]
        except Exception as e:
            print(f"[QWEN] Error loading dictionary from DB: {e}")
            return []

    def translate(self, text):
        if not text:
            return []
            
        if self.mode == 'none' or not self.allowed_words:
            print("[QWEN] Qwen is disabled or database dictionary is empty. Skipping to fallback.")
            return []

        # Construct Prompt
        system_prompt = (
            "Bạn là trợ lý dịch thuật Ngôn ngữ ký hiệu Việt Nam (VSL) chuyên nghiệp.\n"
            "Hãy dịch câu tiếng Việt tự nhiên của người dùng thành một chuỗi các từ khóa ký hiệu tương ứng.\n\n"
            "BẮT BUỘC CHỈ ĐƯỢC CHỌN TỪ TRONG DANH SÁCH TỪ ĐIỂN VSL DƯỚI ĐÂY (không được tự bịa từ, không được giữ nguyên cả câu, không chọn từ ngoài danh sách):\n"
            f"{self.allowed_words}\n\n"
            "Quy tắc quan trọng:\n"
            "1. Dịch đầy đủ và không được bỏ sót các danh từ, động từ, tính từ chính trong câu.\n"
            "2. Ánh xạ các từ đơn giản sang từ tương ứng trong từ điển bất kể phần chú thích trong ngoặc đơn. Ví dụ: 'đẹp' -> 'Đẹp (người)', 'ghế' -> 'Cái ghế', 'bàn' -> 'Cái bàn', 'xấu' -> 'Xấu (người)'.\n"
            "3. Ngay cả cụm từ rất ngắn như 'ghế đẹp', 'bàn cao' cũng phải dịch đầy đủ các từ: 'ghế đẹp' -> ['Cái ghế', 'Đẹp (người)'], 'bàn cao' -> ['Cái bàn', 'Cao (đồ vật)'].\n"
            "4. Bắt buộc trả về định dạng JSON duy nhất như sau: {\"glosses\": [\"Từ 1\", \"Từ 2\"]}\n\n"
            "Ví dụ minh họa:\n"
            "- Input: 'ghế đẹp'\n"
            "  Output: {\"glosses\": [\"Cái ghế\", \"Đẹp (người)\"]}\n"
            "- Input: 'Cái ghế này đẹp quá'\n"
            "  Output: {\"glosses\": [\"Cái ghế\", \"Đẹp (người)\"]}\n"
            "- Input: 'Tên tôi là An'\n"
            "  Output: {\"glosses\": [\"Anh\"]}\n"
            "- Input: 'Tôi đi xe đạp'\n"
            "  Output: {\"glosses\": [\"Xe đạp\"]}"
        )
        
        user_prompt = f"Dịch câu: '{text}'"

        print(f"[QWEN] Translating sentence: '{text}' using mode: {self.mode}")
        
        try:
            if self.mode == 'api':
                return self._call_api(system_prompt, user_prompt)
            elif self.mode == 'local':
                return self._call_local(system_prompt, user_prompt)
        except Exception as e:
            print(f"[QWEN] Error during Qwen translation execution: {e}")
            
        return []

    def _call_api(self, system_prompt, user_prompt):
        # We call the Hugging Face Serverless Inference API for Qwen using the working router domain
        api_url = f"https://router.huggingface.co/models/{self.model_name}"
        headers = {}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
            
        payload = {
            "inputs": f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n",
            "parameters": {
                "max_new_tokens": 150,
                "temperature": 0.1,
                "return_full_text": False
            }
        }
        
        response = requests.post(api_url, headers=headers, json=payload, timeout=10)
        
        if response.status_code != 200:
            print(f"[QWEN API] Error HTTP {response.status_code}: {response.text}")
            # Try a public fallback endpoint or model if 404 or authorization issue
            if response.status_code in [401, 403, 404]:
                print("[QWEN API] Retrying with public Qwen-2.5-72B-Instruct endpoint...")
                fallback_url = "https://router.huggingface.co/models/Qwen/Qwen2.5-72B-Instruct"
                response = requests.post(fallback_url, json=payload, timeout=12)
                
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get('generated_text', '')
                return self._parse_json_result(generated_text)
            elif isinstance(result, dict):
                generated_text = result.get('generated_text', '')
                return self._parse_json_result(generated_text)
        else:
            print(f"[QWEN API] Fallback API also failed: {response.text}")
            
        return []

    def _call_local(self, system_prompt, user_prompt):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            print("[QWEN LOCAL] Missing 'transformers' or 'torch' package. Please install them to run locally.")
            return []
            
        # Nạp và lưu trữ mô hình/tokenizer vào instance để tái sử dụng ở các lượt gọi sau
        if not hasattr(self, '_tokenizer') or not hasattr(self, '_model'):
            print(f"[QWEN LOCAL] Loading model {self.model_name} on CPU for the first time...")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                device_map="cpu"
            )
            print("[QWEN LOCAL] Model loaded successfully in memory.")
            
        tokenizer = self._tokenizer
        model = self._model
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        text_input = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = tokenizer([text_input], return_tensors="pt")
        
        print("[QWEN LOCAL] Generating response...")
        generated_ids = model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=60,
            temperature=0.1
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return self._parse_json_result(response_text)

    def _parse_json_result(self, text):
        print(f"[QWEN] Raw LLM Response: {text.strip()}")
        try:
            # Clean potential markdown wrappers like ```json ... ```
            clean_text = text.strip()
            if "```" in clean_text:
                # Extract content inside code block
                matches = re.findall(r"```(?:json)?(.*?)```", clean_text, re.DOTALL)
                if matches:
                    clean_text = matches[0].strip()
            
            # Find the JSON object starting with { and ending with }
            json_match = re.search(r"\{.*?\}", clean_text, re.DOTALL)
            if json_match:
                clean_text = json_match.group(0)
                
            data = json.loads(clean_text)
            glosses = data.get("glosses", [])
            
            # Filter output: ONLY include words that exist in allowed_words list (case-insensitive check but return DB casing)
            filtered_glosses = []
            allowed_lower_map = {w.lower(): w for w in self.allowed_words}
            
            for g in glosses:
                g_clean = g.strip()
                if g_clean.lower() in allowed_lower_map:
                    # Return original casing from DB
                    filtered_glosses.append(allowed_lower_map[g_clean.lower()])
                else:
                    print(f"[QWEN] Omitted word '{g}' because it is not in database dictionary.")
                    
            print(f"[QWEN] Output Glosses: {filtered_glosses}")
            return filtered_glosses
            
        except Exception as e:
            print(f"[QWEN] Failed to parse Qwen response as JSON: {e}")
            
        return []
