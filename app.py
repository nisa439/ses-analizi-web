from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import librosa
import io
import os
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import tempfile

app = Flask(__name__)
CORS(app)

# Upload klasörünü oluştur
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

class AudioAnalyzer:
    def __init__(self):
        self.sample_rate = 22050
        self.duration = 3
    
    def analyze_audio(self, audio_path):
        """Ses dosyasını analiz et"""
        try:
            # Ses dosyasını yükle
            y, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.duration)
            
            # Özellik çıkarımı
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            rms_energy = np.mean(librosa.feature.rms(y=y))
            zcr = np.mean(librosa.feature.zero_crossing_rate(y))
            rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfccs, axis=1)
            
            # Duygu analizi (geliştirilmiş kurallar)
            emotion = self._predict_emotion(spectral_centroid, rms_energy, zcr, mfcc_mean)
            
            # Demografik analiz
            demographics = self._predict_demographics(spectral_centroid, rms_energy, y)
            
            return {
                'emotion': emotion['emotion'],
                'confidence': emotion['confidence'],
                'age': demographics['age'],
                'gender': demographics['gender'],
                'gender_confidence': demographics['gender_confidence'],
                'features': {
                    'spectral_centroid': float(spectral_centroid),
                    'rms_energy': float(rms_energy),
                    'zcr': float(zcr),
                    'rolloff': float(rolloff),
                    'mfcc_mean': mfcc_mean.tolist()
                }
            }
            
        except Exception as e:
            print(f"Analiz hatası: {e}")
            return None
    
    def _predict_emotion(self, sc, rms, zcr, mfcc):
        """Geliştirilmiş duygu tahmini"""
        # Normalize edilmiş özellikler
        energy_norm = min(1.0, rms * 50)
        pitch_norm = min(1.0, sc / 3000)
        
        # Karmaşık kural tabanlı sistem
        if energy_norm > 0.7 and zcr > 0.15:
            emotion = "Angry"
            confidence = 0.8
        elif pitch_norm > 0.6 and energy_norm > 0.4:
            emotion = "Happy"
            confidence = 0.75
        elif energy_norm < 0.3:
            emotion = "Sad"
            confidence = 0.7
        elif zcr > 0.2 and pitch_norm > 0.5:
            emotion = "Fear"
            confidence = 0.65
        elif mfcc[1] < -10:  # MFCC özelliği
            emotion = "Disgust"
            confidence = 0.6
        else:
            emotion = "Neutral"
            confidence = 0.5
            
        return {'emotion': emotion, 'confidence': confidence}
    
    def _predict_demographics(self, sc, rms, audio):
        """Demografik analiz"""
        # Cinsiyet tahmini (geliştirilmiş)
        if sc > 1800:
            gender = "Kadın"
            gender_conf = 0.8
        elif sc < 1200:
            gender = "Erkek"
            gender_conf = 0.8
        else:
            # Formant analizi ile
            formants = self._estimate_formants(audio)
            if formants and formants[0] > 900:
                gender = "Kadın"
                gender_conf = 0.65
            else:
                gender = "Erkek"
                gender_conf = 0.65
        
        # Yaş tahmini (ses özelliklerine göre)
        if sc > 2000 and rms > 0.02:
            age = np.random.randint(18, 30)
        elif sc > 1500:
            age = np.random.randint(25, 45)
        elif sc > 1000:
            age = np.random.randint(35, 55)
        else:
            age = np.random.randint(45, 70)
            
        return {
            'age': int(age),
            'gender': gender,
            'gender_confidence': gender_conf
        }
    
    def _estimate_formants(self, audio):
        """Basit formant tahmini"""
        try:
            # FFT ile frekans analizi
            fft = np.abs(np.fft.fft(audio))
            freqs = np.fft.fftfreq(len(fft), 1/self.sample_rate)
            
            # Pozitif frekanslar
            positive_freqs = freqs[:len(freqs)//2]
            positive_fft = fft[:len(fft)//2]
            
            # Pikleri bul
            peaks = []
            for i in range(1, len(positive_fft)-1):
                if positive_fft[i] > positive_fft[i-1] and positive_fft[i] > positive_fft[i+1]:
                    if positive_freqs[i] > 200 and positive_freqs[i] < 3000:
                        peaks.append(positive_freqs[i])
            
            return sorted(peaks)[:2] if peaks else None
            
        except:
            return None
    
    def create_visualization(self, emotion, age, gender, features):
        """Gelişmiş görselleştirme"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            
            # Renk paleti
            colors = {
                'Angry': '#FF4444', 'Happy': '#FFD700', 'Sad': '#4169E1',
                'Fear': '#9932CC', 'Disgust': '#FF6347', 'Neutral': '#808080'
            }
            
            # Emoji mapping
            emojis = {
                'Angry': '😠', 'Happy': '😊', 'Sad': '😢',
                'Fear': '😨', 'Disgust': '🤢', 'Neutral': '😐'
            }
            
            color = colors.get(emotion, '#808080')
            emoji = emojis.get(emotion, '🎭')
            
            # 1. Ana sonuç
            ax1.set_facecolor(color)
            ax1.text(0.5, 0.7, emoji, fontsize=60, ha='center', va='center')
            ax1.text(0.5, 0.4, emotion, fontsize=20, ha='center', va='center', 
                    color='white', weight='bold')
            ax1.text(0.5, 0.2, f"{age} yaş {gender}", fontsize=14, ha='center', va='center',
                    color='white')
            ax1.set_xlim(0, 1)
            ax1.set_ylim(0, 1)
            ax1.axis('off')
            ax1.set_title('Analiz Sonucu', fontsize=16, pad=20)
            
            # 2. Özellik grafiği
            feature_names = ['Spektral\nCentroid', 'RMS\nEnergy', 'Zero\nCrossing', 'Rolloff']
            feature_values = [
                features['spectral_centroid']/3000,  # Normalize
                features['rms_energy']*20,
                features['zcr']*5,
                features['rolloff']/5000
            ]
            
            bars = ax2.bar(feature_names, feature_values, color=[color]*4, alpha=0.7)
            ax2.set_ylim(0, 1)
            ax2.set_title('Ses Özellikleri', fontsize=14)
            ax2.set_ylabel('Normalize Değer')
            
            # Değerleri bar'ların üstüne yaz
            for bar, value in zip(bars, feature_values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.2f}', ha='center', va='bottom', fontsize=10)
            
            # 3. MFCC grafiği
            if 'mfcc_mean' in features:
                mfcc_values = features['mfcc_mean'][:12]  # İlk 12 MFCC
                ax3.plot(range(len(mfcc_values)), mfcc_values, 'o-', color=color, linewidth=2)
                ax3.set_title('MFCC Özellikleri', fontsize=14)
                ax3.set_xlabel('MFCC Katsayısı')
                ax3.set_ylabel('Değer')
                ax3.grid(True, alpha=0.3)
            
            # 4. Duygu dağılımı (örnek)
            emotions = ['Angry', 'Happy', 'Sad', 'Fear', 'Disgust', 'Neutral']
            # Mevcut duyguma yüksek skor, diğerlerine rastgele
            scores = [0.1 + np.random.random()*0.2 for _ in emotions]
            emotion_idx = emotions.index(emotion) if emotion in emotions else 0
            scores[emotion_idx] = 0.7 + np.random.random()*0.3
            
            ax4.pie(scores, labels=emotions, autopct='%1.1f%%', startangle=90,
                   colors=[colors.get(e, '#808080') for e in emotions])
            ax4.set_title('Duygu Dağılımı', fontsize=14)
            
            plt.tight_layout()
            
            # Base64'e çevir
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close(fig)
            
            return image_base64
            
        except Exception as e:
            print(f"Görsel oluşturma hatası: {e}")
            return None

# Global analyzer
analyzer = AudioAnalyzer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    try:
        if 'audio' not in request.files:
            return jsonify({'success': False, 'error': 'Ses dosyası bulunamadı'})
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'Dosya seçilmedi'})
        
        # Güvenli dosya adı oluştur
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Dosyayı kaydet
        file.save(filepath)
        
        # Analiz et
        result = analyzer.analyze_audio(filepath)
        if not result:
            os.remove(filepath)
            return jsonify({'success': False, 'error': 'Analiz başarısız'})
        
        # Görsel oluştur
        visualization = analyzer.create_visualization(
            result['emotion'], result['age'], result['gender'], result['features']
        )
        
        # Geçici dosyayı sil
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'data': {
                'analysis': result,
                'visualization': visualization,
                'timestamp': datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/health')
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)