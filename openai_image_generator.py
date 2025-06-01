# openai_image_generator.py - OpenAI DALL-E 3 AI Görsel Üretici

import requests
import base64
import os
import time
from typing import Optional, Dict, Any
import json
from PIL import Image, ImageDraw, ImageFont
import io

class OpenAIImageGenerator:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # DALL-E 3 endpoint
        self.dalle_url = "https://api.openai.com/v1/images/generations"
        
        # Emotion-based sophisticated prompts
        self.emotion_prompts = {
            "Happy": {
                "base": "A radiantly joyful person with an authentic, luminous smile",
                "style": "warm golden lighting, vibrant and uplifting colors, cheerful atmosphere",
                "mood": "exuberant, blissful, euphoric energy",
                "artistic": "portrait photography style, professional lighting, high resolution"
            },
            "Sad": {
                "base": "A contemplative person with gentle melancholy in their expression",
                "style": "soft blue tones, gentle shadows, introspective lighting",
                "mood": "pensive, emotional depth, touching vulnerability",
                "artistic": "cinematic portrait, dramatic lighting, emotional depth"
            },
            "Angry": {
                "base": "A person with intense, determined expression showing controlled strength",
                "style": "bold contrasts, dramatic red accents, powerful stance",
                "mood": "fierce determination, passionate intensity, strong will",
                "artistic": "dramatic portrait photography, strong lighting contrasts"
            },
            "Fear": {
                "base": "A person with wide, alert eyes showing cautious awareness",
                "style": "mysterious shadows, cool blue tones, tense atmosphere",
                "mood": "heightened alertness, nervous energy, anxious anticipation",
                "artistic": "dramatic lighting, cinematic style, atmospheric"
            },
            "Disgust": {
                "base": "A person with a discerning, critical expression showing refined judgment",
                "style": "cool green undertones, sophisticated lighting, stern composure",
                "mood": "discriminating, sophisticated disapproval, refined critique",
                "artistic": "professional portrait, sharp details, elegant composition"
            },
            "Neutral": {
                "base": "A serene person with peaceful, balanced expression",
                "style": "natural lighting, harmonious colors, calm atmosphere",
                "mood": "tranquil, centered, composed serenity",
                "artistic": "classic portrait style, natural lighting, timeless quality"
            }
        }
        
        # Gender-specific artistic descriptions
        self.gender_styles = {
            "Kadın": "elegant woman with graceful feminine features, sophisticated beauty, refined pose",
            "Erkek": "distinguished man with strong masculine features, confident demeanor, authoritative presence",
            "Belirsiz": "person with androgynous appeal, universal human beauty, timeless features"
        }
        
        # Age-specific characteristics
        self.age_styles = {
            "young": "youthful vitality, smooth complexion, energetic aura, modern style",
            "middle": "mature sophistication, experienced confidence, refined wisdom, professional elegance",
            "older": "distinguished wisdom, noble features, serene composure, timeless dignity"
        }
        
        print("✅ OpenAI DALL-E 3 Image Generator hazır!")
    
    def get_age_category(self, age: int) -> str:
        """Yaşa göre kategori belirle"""
        if age < 30:
            return "young"
        elif age < 55:
            return "middle"
        else:
            return "older"
    
    def create_sophisticated_prompt(self, emotion: str, age: int, gender: str, ensemble_info: Dict[str, Any]) -> str:
        """Gelişmiş ve detaylı prompt oluştur"""
        emotion_data = self.emotion_prompts.get(emotion, self.emotion_prompts["Neutral"])
        gender_style = self.gender_styles.get(gender, self.gender_styles["Belirsiz"])
        age_category = self.get_age_category(age)
        age_style = self.age_styles.get(age_category, self.age_styles["middle"])
        
        # Ensemble güvenilirlik bilgisine göre kalite ayarı
        confidence = ensemble_info.get('confidence', 0.5)
        agreement_score = ensemble_info.get('agreement_score', 0.5)
        
        # Yüksek güvenilirlik için daha detaylı prompt
        if confidence > 0.7 and agreement_score > 0.7:
            quality_boost = "masterpiece quality, highly detailed, photorealistic, award-winning portrait"
        elif confidence > 0.5:
            quality_boost = "high quality, detailed, professional photography"
        else:
            quality_boost = "artistic interpretation, stylized portrait"
        
        # Ana prompt'u birleştir
        prompt = f"""{gender_style}, {age_style}, {emotion_data['base']}, 
        {emotion_data['style']}, {emotion_data['mood']}, 
        {emotion_data['artistic']}, {quality_boost}, 
        perfect facial features, expressive eyes, natural skin texture, 
        studio lighting, shallow depth of field, professional headshot"""
        
        # Prompt'u temizle ve kısalt (DALL-E 3 için optimize)
        prompt = " ".join(prompt.split()).strip()
        if len(prompt) > 1000:
            prompt = prompt[:1000] + "..."
        
        return prompt
    
    def test_api_connection(self) -> bool:
        """OpenAI API bağlantısını test et"""
        try:
            test_url = "https://api.openai.com/v1/models"
            response = requests.get(test_url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                models_data = response.json()
                # DALL-E 3 modelinin mevcut olup olmadığını kontrol et
                dall_e_available = any('dall-e' in model.get('id', '').lower() 
                                     for model in models_data.get('data', []))
                
                if dall_e_available:
                    print("✅ OpenAI API Test başarılı! DALL-E 3 erişilebilir")
                    return True
                else:
                    print("⚠️ OpenAI API bağlandı ama DALL-E 3 bulunamadı")
                    return False
            else:
                print(f"❌ OpenAI API Test başarısız: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ OpenAI API Test hatası: {e}")
            return False
    
    def generate_dalle_image(self, prompt: str, size: str = "1024x1024", quality: str = "standard") -> Optional[str]:
        """DALL-E 3 ile görsel üret"""
        try:
            payload = {
                "model": "dall-e-3",
                "prompt": prompt,
                "n": 1,
                "size": size,
                "quality": quality,
                "response_format": "b64_json"
            }
            
            print(f"🎨 DALL-E 3 görsel üretiliyor...")
            print(f"📝 Prompt: {prompt[:100]}...")
            
            response = requests.post(
                self.dalle_url,
                headers=self.headers,
                json=payload,
                timeout=120  # DALL-E 3 biraz yavaş olabilir
            )
            
            if response.status_code == 200:
                result = response.json()
                image_data = result['data'][0]['b64_json']
                print(f"✅ DALL-E 3 görsel üretildi! Kalite: {quality}, Boyut: {size}")
                return image_data
                
            elif response.status_code == 400:
                error_detail = response.json().get('error', {}).get('message', 'Bilinmeyen hata')
                print(f"❌ DALL-E 3 prompt hatası: {error_detail}")
                return None
                
            elif response.status_code == 429:
                print("⏳ DALL-E 3 rate limit, bekleniyor...")
                return None
                
            else:
                print(f"❌ DALL-E 3 API hatası: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ DALL-E 3 istek hatası: {e}")
            return None
    
    def generate_emotion_image(self, emotion: str, age: int, gender: str, ensemble_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Ana görsel üretim fonksiyonu"""
        print(f"🎭 DALL-E 3 görsel üretimi başlıyor: {emotion}, {age}, {gender}")
        
        # Gelişmiş prompt oluştur
        prompt = self.create_sophisticated_prompt(emotion, age, gender, ensemble_info)
        
        # Ensemble confidence'a göre kalite ayarı
        confidence = ensemble_info.get('confidence', 0.5)
        quality = "hd" if confidence > 0.7 else "standard"
        
        # DALL-E 3 ile görsel üret
        image_base64 = self.generate_dalle_image(prompt, "1024x1024", quality)
        
        if image_base64:
            result = {
                "success": True,
                "image_base64": image_base64,
                "model_used": "dall-e-3",
                "prompt_used": prompt,
                "generation_info": {
                    "emotion": emotion,
                    "age": age,
                    "gender": gender,
                    "quality": quality,
                    "size": "1024x1024",
                    "timestamp": time.time(),
                    "ensemble_confidence": confidence
                }
            }
            print("✅ DALL-E 3 görsel üretimi başarılı!")
            return result
        
        return None
    
    def create_fallback_image(self, emotion: str, age: int, gender: str) -> str:
        """Hata durumunda basit görsel oluştur"""
        try:
            # 1024x1024 canvas
            width, height = 1024, 1024
            
            # Duygu renkleri
            colors = {
                'Happy': '#FFD700', 'Sad': '#4169E1', 'Angry': '#FF4444',
                'Fear': '#9932CC', 'Disgust': '#FF6347', 'Neutral': '#808080'
            }
            
            # Duygu emojileri
            emotions_text = {
                'Happy': '😊 Mutluluk', 'Sad': '😢 Üzüntü', 'Angry': '😠 Öfke',
                'Fear': '😨 Korku', 'Disgust': '🤢 Tiksinme', 'Neutral': '😐 Nötr'
            }
            
            primary_color = colors.get(emotion, '#808080')
            emotion_text = emotions_text.get(emotion, f'🎭 {emotion}')
            
            # Gradient arka plan oluştur
            img = Image.new('RGB', (width, height), primary_color)
            draw = ImageDraw.Draw(img)
            
            # Gradient efekti
            for i in range(height):
                alpha = i / height
                color_val = int(255 * (0.3 + 0.7 * alpha))
                draw.line([(0, i), (width, i)], fill=(color_val, color_val, color_val, 100))
            
            # Merkez daire
            center_x, center_y = width // 2, height // 2
            circle_radius = 200
            
            # Daire çiz
            draw.ellipse([
                center_x - circle_radius, center_y - circle_radius,
                center_x + circle_radius, center_y + circle_radius
            ], fill=primary_color, outline='white', width=8)
            
            # Ana text (emotion + emoji)
            try:
                # Varsayılan font kullan
                font_size = 48
                draw.text((center_x, center_y - 60), emotion_text, 
                         fill='white', anchor='mm', font_size=font_size)
                
                # Demografik bilgi
                demo_text = f"{age} yaş • {gender}"
                draw.text((center_x, center_y + 20), demo_text,
                         fill='white', anchor='mm', font_size=32)
                
                # Alt bilgi
                draw.text((center_x, center_y + 80), "AI Görsel Placeholder",
                         fill='white', anchor='mm', font_size=24)
                
            except:
                # Font hatası durumunda basit text
                draw.text((center_x - 100, center_y - 30), emotion, fill='white')
                draw.text((center_x - 80, center_y + 10), f"{age}, {gender}", fill='white')
            
            # Dekoratif elementler
            for i in range(8):
                angle = i * 45
                x = center_x + 300 * (1 if i % 2 == 0 else -1)
                y = center_y + 300 * (1 if i < 4 else -1)
                if 0 <= x <= width and 0 <= y <= height:
                    draw.ellipse([x-10, y-10, x+10, y+10], fill='white', outline=primary_color)
            
            # Base64'e çevir
            buffer = io.BytesIO()
            img.save(buffer, format='PNG', quality=95)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            print("✅ Fallback görsel oluşturuldu")
            return image_base64
            
        except Exception as e:
            print(f"❌ Fallback görsel hatası: {e}")
            # En basit fallback
            return ""
    
    def batch_generate_images(self, requests_list: list) -> list:
        """Toplu görsel üretimi"""
        results = []
        
        for req in requests_list:
            emotion = req.get('emotion', 'Neutral')
            age = req.get('age', 30)
            gender = req.get('gender', 'Belirsiz')
            ensemble_info = req.get('ensemble_info', {})
            
            result = self.generate_emotion_image(emotion, age, gender, ensemble_info)
            results.append(result)
            
            # Rate limiting için bekle
            time.sleep(2)
        
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """Generator durumu"""
        return {
            'model': 'dall-e-3',
            'api_connected': self.test_api_connection(),
            'features': {
                'high_quality': True,
                'custom_prompts': True,
                'fallback_support': True,
                'batch_generation': True
            },
            'limits': {
                'max_resolution': '1024x1024',
                'qualities': ['standard', 'hd'],
                'rate_limit': '5 requests/minute'
            }
        }
    
    def save_generated_image(self, image_base64: str, filename: str) -> bool:
        """Üretilen görseli dosyaya kaydet"""
        try:
            image_data = base64.b64decode(image_base64)
            
            os.makedirs('generated_images', exist_ok=True)
            filepath = os.path.join('generated_images', filename)
            
            with open(filepath, 'wb') as f:
                f.write(image_data)
            
            print(f"✅ Görsel kaydedildi: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Görsel kaydetme hatası: {e}")
            return False