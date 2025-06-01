# app.py - Sadece OpenAI DALL-E 3 AI Görsel Üretimi ile temizlenmiş versiyon

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import os
import base64
import matplotlib
matplotlib.use('Agg')  # Web sunucusu için gerekli
import matplotlib.pyplot as plt
import io
from datetime import datetime
import tempfile

# Ensemble analyzer import
from ensemble_audio_analyzer import EnsembleAudioAnalyzer

# OpenAI image generator import
try:
    from openai_image_generator import OpenAIImageGenerator
    OPENAI_AVAILABLE = True
    print("✅ OpenAI DALL-E 3 Image Generator modülü yüklendi")
except ImportError:
    print("⚠️ OpenAI Image Generator bulunamadı")
    OPENAI_AVAILABLE = False

app = Flask(__name__)

# CORS ayarları
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://127.0.0.1:5000", "http://localhost:5000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Upload klasörünü oluştur
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Models klasörünü oluştur
MODELS_FOLDER = 'models'
if not os.path.exists(MODELS_FOLDER):
    os.makedirs(MODELS_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# OpenAI API Key - YENİ TOKEN
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'your-key-here')
# Global components
try:
    analyzer = EnsembleAudioAnalyzer()
    print("✅ Ensemble Audio Analyzer hazır!")
except Exception as e:
    print(f"❌ Ensemble analyzer hatası: {e}")
    analyzer = None

# OpenAI DALL-E 3 Image Generator
ai_generator = None

if OPENAI_AVAILABLE and OPENAI_API_KEY and OPENAI_API_KEY.startswith('sk-'):
    try:
        ai_generator = OpenAIImageGenerator(OPENAI_API_KEY)
        if ai_generator.test_api_connection():
            print("✅ OpenAI DALL-E 3 AI Görsel Üretici hazır!")
        else:
            print("⚠️ OpenAI API bağlantı sorunu")
            ai_generator = None
    except Exception as e:
        print(f"❌ OpenAI generator hatası: {e}")
        ai_generator = None

if ai_generator is None:
    print("⚠️ AI görsel üretici deaktif - sadece ensemble analiz çalışacak")

def create_enhanced_visualization(emotion, age, gender, ensemble_info, features):
    """Gelişmiş görselleştirme - ensemble bilgileri dahil"""
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Renk paleti
        colors = {
            'Angry': '#FF4444', 'Happy': '#FFD700', 'Sad': '#4169E1',
            'Fear': '#9932CC', 'Disgust': '#FF6347', 'Neutral': '#808080'
        }
        
        # Emoji'ler için text kullan (font sorunu çözümü)
        emoji_text = {
            'Angry': 'ANGRY', 'Happy': 'HAPPY', 'Sad': 'SAD',
            'Fear': 'FEAR', 'Disgust': 'DISGUST', 'Neutral': 'NEUTRAL'
        }
        
        color = colors.get(emotion, '#808080')
        emotion_display = emoji_text.get(emotion, emotion.upper())
        
        # 1. Ana sonuç + Ensemble bilgisi
        ax1.set_facecolor(color)
        ax1.text(0.5, 0.7, emotion_display, fontsize=32, ha='center', va='center', 
                color='white', weight='bold')
        ax1.text(0.5, 0.55, emotion, fontsize=20, ha='center', va='center',
                color='white', weight='bold')
        ax1.text(0.5, 0.4, f"{age} yaş {gender}", fontsize=16, ha='center', va='center',
                color='white')
        
        # Ensemble bilgisi
        model_count = ensemble_info.get('model_count', 0)
        agreement = ensemble_info.get('agreement_score', 0) * 100
        
        ax1.text(0.5, 0.25, f"Models: {model_count}", fontsize=14, ha='center', va='center',
                color='white', alpha=0.9)
        ax1.text(0.5, 0.15, f"Agreement: {agreement:.0f}%", fontsize=12, ha='center', va='center',
                color='white', alpha=0.8)
        
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        ax1.set_title('Ensemble Result', fontsize=16, pad=20, color='white')
        
        # 2. Model karşılaştırması
        individual_preds = ensemble_info.get('individual_predictions', {})
        if individual_preds:
            models = list(individual_preds.keys())
            predictions = list(individual_preds.values())
            
            # Her model için bar
            y_pos = np.arange(len(models))
            pred_colors = [colors.get(pred, '#808080') for pred in predictions]
            
            bars = ax2.barh(y_pos, [1]*len(models), color=pred_colors, alpha=0.7)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(models)
            ax2.set_xlabel('Model Predictions')
            ax2.set_title('Model Comparison', fontsize=14)
            
            # Bar'ların üzerine emotion yazısı
            for i, (bar, pred) in enumerate(zip(bars, predictions)):
                ax2.text(0.5, bar.get_y() + bar.get_height()/2, pred,
                        ha='center', va='center', fontweight='bold', color='white')
        else:
            ax2.text(0.5, 0.5, 'No model\ninformation', ha='center', va='center',
                    fontsize=16, transform=ax2.transAxes)
            ax2.set_title('Model Comparison', fontsize=14)
        
        # 3. Ses özellikleri
        if features:
            feature_names = ['Spectral\nCentroid', 'RMS\nEnergy', 'Zero\nCrossing', 'Rolloff']
            feature_values = [
                min(features.get('spectral_centroid', 0)/3000, 1),  # Normalize
                min(features.get('rms_energy', 0)*20, 1),
                min(features.get('zcr', 0)*5, 1), 
                min(features.get('rolloff', 0)/5000, 1)
            ]
            
            bars = ax3.bar(feature_names, feature_values, color=[color]*4, alpha=0.7)
            ax3.set_ylim(0, 1)
            ax3.set_title('Audio Features', fontsize=14)
            ax3.set_ylabel('Normalized Value')
            
            # Değerleri bar'ların üstüne yaz
            for bar, value in zip(bars, feature_values):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.2f}', ha='center', va='bottom', fontsize=10)
        
        # 4. Güvenilirlik metrikleri
        contributing_models = ensemble_info.get('contributing_models', [])
        if contributing_models:
            metrics = {
                'Models': len(contributing_models),
                'Agreement': agreement,
                'Confidence': ensemble_info.get('confidence', 0) * 100 if 'confidence' in ensemble_info else 0,
                'Quality': min(100, (len(contributing_models) * 25) + agreement)
            }
            
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())
            
            # Radar chart benzeri
            angles = np.linspace(0, 2*np.pi, len(metric_names), endpoint=False)
            metric_values_norm = [v/100 for v in metric_values]  # 0-1 normalize
            
            # Close the plot
            angles = np.concatenate((angles, [angles[0]]))
            metric_values_norm = metric_values_norm + [metric_values_norm[0]]
            
            ax4.plot(angles, metric_values_norm, 'o-', linewidth=2, color=color)
            ax4.fill(angles, metric_values_norm, alpha=0.25, color=color)
            ax4.set_xticks(angles[:-1])
            ax4.set_xticklabels(metric_names, fontsize=10)
            ax4.set_ylim(0, 1)
            ax4.set_title('Reliability Metrics', fontsize=14)
            ax4.grid(True)
        else:
            ax4.text(0.5, 0.5, 'No metric\ninformation', ha='center', va='center',
                    fontsize=16, transform=ax4.transAxes)
            ax4.set_title('Reliability Metrics', fontsize=14)
        
        plt.tight_layout()
        
        # Base64'e çevir
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight', facecolor='white')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close(fig)
        
        return image_base64
        
    except Exception as e:
        print(f"❌ Görsel oluşturma hatası: {e}")
        return None

@app.route('/')
def index():
    """Ana sayfa"""
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """Ana analiz endpoint'i - Ensemble model + OpenAI DALL-E 3"""
    try:
        if 'audio' not in request.files:
            return jsonify({'success': False, 'error': 'Ses dosyası bulunamadı'})
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'Dosya seçilmedi'})
        
        print(f"📁 Dosya alındı: {file.filename}")
        
        # Güvenli dosya adı oluştur
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Dosyayı kaydet
        file.save(filepath)
        
        if analyzer is None:
            os.remove(filepath)
            return jsonify({'success': False, 'error': 'Analyzer başlatılamadı'})
        
        # Ensemble analizi yap
        print("🤖 Ensemble analiz başlıyor...")
        result = analyzer.analyze_audio_ensemble(filepath)
        
        if not result:
            os.remove(filepath)
            return jsonify({'success': False, 'error': 'Analiz başarısız'})
        
        print(f"✅ Analiz tamamlandı: {result['emotion']} (%{result['confidence']*100:.1f})")
        
        # Matplotlib görsel oluştur
        chart_visualization = create_enhanced_visualization(
            result['emotion'], 
            result['age'], 
            result['gender'], 
            result['ensemble_info'],
            result['features']
        )
        
        # OpenAI DALL-E 3 ile AI görsel üret
        ai_image = None
        ai_info = None
        
        if ai_generator:
            print("🎨 DALL-E 3 görsel üretimi başlıyor...")
            try:
                ai_result = ai_generator.generate_emotion_image(
                    result['emotion'],
                    result['age'],
                    result['gender'],
                    result['ensemble_info']
                )
                
                if ai_result and ai_result.get('success'):
                    ai_image = ai_result['image_base64']
                    ai_info = {
                        'model_used': ai_result['model_used'],
                        'prompt_used': ai_result['prompt_used'][:200] + "...",
                        'generation_info': ai_result['generation_info']
                    }
                    print("✅ DALL-E 3 görsel üretimi başarılı!")
                else:
                    print("⚠️ DALL-E 3 görsel üretimi başarısız, fallback kullanılıyor...")
                    ai_image = ai_generator.create_fallback_image(
                        result['emotion'], result['age'], result['gender']
                    )
                    ai_info = {'model_used': 'fallback', 'prompt_used': 'Basit placeholder görsel'}
                    
            except Exception as e:
                print(f"❌ DALL-E 3 hatası: {e}")
                if ai_generator:
                    ai_image = ai_generator.create_fallback_image(
                        result['emotion'], result['age'], result['gender']
                    )
                    ai_info = {'model_used': 'fallback', 'prompt_used': 'Hata sonrası fallback'}
        else:
            print("⚠️ DALL-E 3 generator yok, sadece chart görseli")
        
        # Geçici dosyayı sil
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'data': {
                'analysis': result,
                'visualizations': {
                    'chart': chart_visualization,
                    'ai_image': ai_image,
                    'ai_info': ai_info
                },
                'timestamp': datetime.now().isoformat(),
                'model_info': {
                    'type': 'ensemble',
                    'models_used': result['ensemble_info']['contributing_models'],
                    'model_count': result['ensemble_info']['model_count'],
                    'agreement_score': result['ensemble_info']['agreement_score']
                },
                'ai_generation': {
                    'enabled': ai_generator is not None,
                    'model_info': ai_info
                }
            }
        })
        
    except Exception as e:
        print(f"❌ API hatası: {e}")
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/generate-image', methods=['POST'])
def api_generate_image():
    """Manuel DALL-E 3 görsel üretimi"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'Veri bulunamadı'})
        
        emotion = data.get('emotion', 'Neutral')
        age = data.get('age', 30)
        gender = data.get('gender', 'Belirsiz')
        ensemble_info = data.get('ensemble_info', {})
        
        if not ai_generator:
            return jsonify({'success': False, 'error': 'DALL-E 3 generator mevcut değil'})
        
        print(f"🎨 Manuel DALL-E 3 görsel talebi: {emotion}, {age}, {gender}")
        
        ai_result = ai_generator.generate_emotion_image(emotion, age, gender, ensemble_info)
        
        if ai_result and ai_result.get('success'):
            return jsonify({
                'success': True,
                'data': {
                    'ai_image': ai_result['image_base64'],
                    'generation_info': ai_result['generation_info'],
                    'model_used': ai_result['model_used'],
                    'prompt_used': ai_result['prompt_used']
                }
            })
        else:
            fallback_image = ai_generator.create_fallback_image(emotion, age, gender)
            return jsonify({
                'success': True,
                'data': {
                    'ai_image': fallback_image,
                    'generation_info': {'type': 'fallback'},
                    'model_used': 'fallback',
                    'prompt_used': 'Basit placeholder görsel'
                }
            })
            
    except Exception as e:
        print(f"❌ Manuel görsel üretim hatası: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/test-openai')
def test_openai():
    """OpenAI API test endpoint'i"""
    if not ai_generator:
        return jsonify({
            'success': False,
            'error': 'OpenAI DALL-E 3 generator mevcut değil',
            'openai_available': OPENAI_AVAILABLE,
            'api_key_valid': bool(OPENAI_API_KEY and OPENAI_API_KEY.startswith('sk-'))
        })
    
    try:
        connection_test = ai_generator.test_api_connection()
        
        return jsonify({
            'success': connection_test,
            'status': 'connected' if connection_test else 'failed',
            'api_key_format': 'valid' if OPENAI_API_KEY.startswith('sk-') else 'invalid',
            'generator_ready': ai_generator is not None,
            'model': 'dall-e-3'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/model-status')
def model_status():
    """Model durumlarını kontrol et"""
    ensemble_status = {}
    if analyzer:
        ensemble_status = analyzer.get_model_status()
    
    openai_status = {
        'available': ai_generator is not None,
        'model': 'dall-e-3',
        'api_key_set': bool(OPENAI_API_KEY and OPENAI_API_KEY.startswith('sk-')),
    }
    
    return jsonify({
        'success': True,
        'ensemble_models': ensemble_status,
        'openai_dalle3': openai_status,
        'system_info': {
            'tensorflow_available': ensemble_status.get('tensorflow_available', False),
            'ai_integration': ai_generator is not None
        }
    })

@app.route('/api/health')
def health_check():
    """Sistem durumu kontrolü"""
    model_status_info = {}
    if analyzer:
        model_status_info = analyzer.get_model_status()
    
    openai_health = {
        'enabled': ai_generator is not None,
        'api_connected': False,
        'model': 'dall-e-3'
    }
    
    if ai_generator:
        try:
            openai_health['api_connected'] = ai_generator.test_api_connection()
        except:
            openai_health['api_connected'] = False
    
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '5.0.0-ensemble-dalle3-only',
        'components': {
            'analyzer_ready': analyzer is not None,
            'ensemble_models': model_status_info,
            'openai_dalle3': openai_health,
            'features': {
                'emotion_analysis': True,
                'ensemble_prediction': len(model_status_info.get('loaded_models', [])) > 0,
                'dalle3_generation': ai_generator is not None,
                'demographic_prediction': True,
                'audio_visualization': True
            }
        }
    })

if __name__ == '__main__':
    print("\n🚀 Ses Analizi Web Uygulaması Başlatılıyor...")
    print("=" * 50)
    print(f"✅ Flask Server: Aktif")
    print(f"✅ Ensemble Models: {len(analyzer.models) if analyzer else 0} model")
    print(f"✅ OpenAI DALL-E 3: {'Aktif' if ai_generator else 'Deaktif'}")
    print(f"🌐 URL: http://127.0.0.1:5000")
    print("=" * 50)
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)