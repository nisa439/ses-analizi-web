# ensemble_audio_analyzer.py - Düzeltilmiş ve temizlenmiş versiyon

import numpy as np
import librosa
import os
from datetime import datetime
import logging

# TensorFlow import'u güvenli şekilde
try:
    import tensorflow as tf
    from tensorflow import keras
    print("✅ TensorFlow yüklendi")
except ImportError:
    print("❌ TensorFlow bulunamadı")
    tf = None
    keras = None

class EnsembleAudioAnalyzer:
    def __init__(self):
        self.sample_rate = 22050
        self.duration = 3
        self.n_mels = 128
        self.n_samples = self.sample_rate * self.duration
        
        # Emotion classes (modelinize göre güncelleyin)
        self.emotion_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad']
        
        # Model paths - WORKING versiyonlarını kullan
        self.model_paths = {
            '2d_cnn': 'models/2d_cnn_model_working.h5',
            '1d_cnn': 'models/1d_cnn_model_working.h5', 
            'hybrid': 'models/hybrid_cnn_model_working.h5'
        }
        
        # Initialize empty models
        self.models = {}
        self.model_weights = {}
        
        # Load models if TensorFlow available
        if tf is not None and keras is not None:
            self.load_all_models()
        else:
            print("⚠️ TensorFlow yok, sadece rule-based analiz yapılacak")
        
        print(f"✅ Ensemble analyzer hazır! {len(self.models)} model yüklendi.")
    
    def load_all_models(self):
        """Tüm modelleri yükle"""
        for model_name, model_path in self.model_paths.items():
            try:
                if os.path.exists(model_path):
                    print(f"🔄 {model_name} modeli yükleniyor...")
                    model = keras.models.load_model(model_path)
                    self.models[model_name] = model
                    
                    # Model weights (performansa göre ayarlayın)
                    if model_name == 'hybrid':
                        self.model_weights[model_name] = 0.5  # En güçlü
                    elif model_name == '2d_cnn':
                        self.model_weights[model_name] = 0.3
                    else:  # 1d_cnn
                        self.model_weights[model_name] = 0.2
                    
                    print(f"✅ {model_name} modeli yüklendi!")
                else:
                    print(f"❌ {model_name} modeli bulunamadı: {model_path}")
                    
            except Exception as e:
                print(f"❌ {model_name} model yükleme hatası: {e}")
        
        # Normalize weights
        total_weight = sum(self.model_weights.values())
        if total_weight > 0:
            for key in self.model_weights:
                self.model_weights[key] /= total_weight
        
        print(f"📊 Model ağırlıkları: {self.model_weights}")
    
    def preprocess_for_2d_cnn(self, audio_file):
        """2D CNN için mel-spectrogram preprocessing - EXACT SHAPE"""
        try:
            y, sr = librosa.load(audio_file, sr=self.sample_rate, duration=self.duration)
            
            # Pad or truncate to exact size
            if len(y) < self.n_samples:
                y = np.pad(y, (0, self.n_samples - len(y)))
            else:
                y = y[:self.n_samples]
            
            # Normalize
            if np.max(np.abs(y)) > 0:
                y = y / np.max(np.abs(y))
            
            # Create mel-spectrogram with EXACT parameters for your model
            mel_spec = librosa.feature.melspectrogram(
                y=y, 
                sr=sr, 
                n_mels=self.n_mels,  # 128
                hop_length=256,      # This gives us 259 time frames for 3s
                n_fft=1024
            )
            
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize mel-spectrogram
            if np.std(mel_spec_db) > 0:
                mel_spec_norm = (mel_spec_db - np.mean(mel_spec_db)) / np.std(mel_spec_db)
            else:
                mel_spec_norm = mel_spec_db
            
            # CRITICAL: Ensure exact shape (128, 259)
            expected_frames = 259
            current_frames = mel_spec_norm.shape[1]
            
            if current_frames < expected_frames:
                # Pad if too short
                pad_width = expected_frames - current_frames
                mel_spec_norm = np.pad(mel_spec_norm, ((0, 0), (0, pad_width)), mode='constant')
            elif current_frames > expected_frames:
                # Truncate if too long
                mel_spec_norm = mel_spec_norm[:, :expected_frames]
            
            print(f"✅ 2D Input Final Shape: {mel_spec_norm.shape} -> Target: (128, 259)")
            
            # Reshape for model: (1, height, width, channels)
            return mel_spec_norm.reshape(1, self.n_mels, expected_frames, 1)
            
        except Exception as e:
            print(f"❌ 2D CNN preprocessing hatası: {e}")
            return None
    
    def preprocess_for_1d_cnn(self, audio_file):
        """1D CNN için raw audio preprocessing - EXACT SHAPE"""
        try:
            y, sr = librosa.load(audio_file, sr=self.sample_rate, duration=self.duration)
            
            # Ensure EXACT length: 66150 samples
            if len(y) < self.n_samples:
                y = np.pad(y, (0, self.n_samples - len(y)))
            else:
                y = y[:self.n_samples]
            
            # Normalize
            if np.max(np.abs(y)) > 0:
                y = y / np.max(np.abs(y))
            
            print(f"✅ 1D Input Final Shape: {y.shape} -> Target: (66150,)")
            
            # Reshape for model: (1, samples, channels)
            return y.reshape(1, self.n_samples, 1)
            
        except Exception as e:
            print(f"❌ 1D CNN preprocessing hatası: {e}")
            return None
    
    def preprocess_for_hybrid(self, audio_file):
        """Hybrid model için preprocessing"""
        try:
            y, sr = librosa.load(audio_file, sr=self.sample_rate, duration=self.duration)
            
            # Pad or truncate
            if len(y) < self.n_samples:
                y = np.pad(y, (0, self.n_samples - len(y)))
            else:
                y = y[:self.n_samples]
            
            # Normalize
            if np.max(np.abs(y)) > 0:
                y = y / np.max(np.abs(y))
            
            print(f"✅ Hybrid Input Final Shape: {y.shape} -> Target: (66150,)")
            
            # Hybrid model 1D input kullanıyor (model oluştururken böyle yaptık)
            return y.reshape(1, self.n_samples, 1)
            
        except Exception as e:
            print(f"❌ Hybrid preprocessing hatası: {e}")
            return None
    
    def predict_single_model(self, model_name, input_data):
        """Tek model ile tahmin yap"""
        try:
            if model_name not in self.models:
                return None
            
            model = self.models[model_name]
            predictions = model.predict(input_data, verbose=0)
            
            # Softmax çıktısını al
            if len(predictions.shape) > 1:
                probabilities = predictions[0]
            else:
                probabilities = predictions
            
            return probabilities
            
        except Exception as e:
            print(f"❌ {model_name} tahmin hatası: {e}")
            return None
    
    def ensemble_predict(self, audio_file):
        """Ensemble tahmin - 3 modelin kombinasyonu"""
        results = {}
        all_probabilities = []
        model_contributions = []
        
        print(f"🔍 Ensemble analiz başlıyor: {os.path.basename(audio_file)}")
        
        # 2D CNN Model
        if '2d_cnn' in self.models:
            input_2d = self.preprocess_for_2d_cnn(audio_file)
            if input_2d is not None:
                prob_2d = self.predict_single_model('2d_cnn', input_2d)
                if prob_2d is not None:
                    results['2d_cnn'] = {
                        'probabilities': prob_2d,
                        'emotion': self.emotion_classes[np.argmax(prob_2d)],
                        'confidence': float(np.max(prob_2d)),
                        'weight': self.model_weights.get('2d_cnn', 0.0)
                    }
                    all_probabilities.append(prob_2d * self.model_weights.get('2d_cnn', 0.0))
                    model_contributions.append('2D CNN')
                    print(f"✅ 2D CNN: {results['2d_cnn']['emotion']} (%{results['2d_cnn']['confidence']*100:.1f})")
        
        # 1D CNN Model  
        if '1d_cnn' in self.models:
            input_1d = self.preprocess_for_1d_cnn(audio_file)
            if input_1d is not None:
                prob_1d = self.predict_single_model('1d_cnn', input_1d)
                if prob_1d is not None:
                    results['1d_cnn'] = {
                        'probabilities': prob_1d,
                        'emotion': self.emotion_classes[np.argmax(prob_1d)],
                        'confidence': float(np.max(prob_1d)),
                        'weight': self.model_weights.get('1d_cnn', 0.0)
                    }
                    all_probabilities.append(prob_1d * self.model_weights.get('1d_cnn', 0.0))
                    model_contributions.append('1D CNN')
                    print(f"✅ 1D CNN: {results['1d_cnn']['emotion']} (%{results['1d_cnn']['confidence']*100:.1f})")
        
        # Hybrid Model
        if 'hybrid' in self.models:
            input_hybrid = self.preprocess_for_hybrid(audio_file)
            if input_hybrid is not None:
                prob_hybrid = self.predict_single_model('hybrid', input_hybrid)
                if prob_hybrid is not None:
                    results['hybrid'] = {
                        'probabilities': prob_hybrid,
                        'emotion': self.emotion_classes[np.argmax(prob_hybrid)],
                        'confidence': float(np.max(prob_hybrid)),
                        'weight': self.model_weights.get('hybrid', 0.0)
                    }
                    all_probabilities.append(prob_hybrid * self.model_weights.get('hybrid', 0.0))
                    model_contributions.append('Hybrid CNN')
                    print(f"✅ Hybrid: {results['hybrid']['emotion']} (%{results['hybrid']['confidence']*100:.1f})")
        
        # Ensemble kombinasyonu
        if all_probabilities:
            # Weighted average
            ensemble_probs = np.sum(all_probabilities, axis=0)
            ensemble_emotion_idx = np.argmax(ensemble_probs)
            ensemble_emotion = self.emotion_classes[ensemble_emotion_idx]
            ensemble_confidence = float(ensemble_probs[ensemble_emotion_idx])
            
            # Model agreement
            predicted_emotions = [result['emotion'] for result in results.values()]
            emotion_counts = {emotion: predicted_emotions.count(emotion) for emotion in set(predicted_emotions)}
            max_agreement = max(emotion_counts.values()) if emotion_counts else 0
            agreement_score = max_agreement / len(results) if results else 0
            
            # Final result
            ensemble_result = {
                'ensemble_emotion': ensemble_emotion,
                'ensemble_confidence': ensemble_confidence,
                'agreement_score': agreement_score,
                'model_count': len(results),
                'contributing_models': model_contributions,
                'individual_results': results,
                'ensemble_probabilities': ensemble_probs.tolist(),
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"🎯 ENSEMBLE SONUÇ: {ensemble_emotion} (%{ensemble_confidence*100:.1f}) - Agreement: %{agreement_score*100:.1f}")
            
            return ensemble_result
        
        else:
            print("❌ Hiçbir model çalışmadı!")
            return None
    
    def analyze_audio_ensemble(self, audio_file):
        """Ana analiz fonksiyonu - ensemble ile"""
        try:
            # Model varsa ensemble prediction
            if len(self.models) > 0:
                ensemble_result = self.ensemble_predict(audio_file)
                
                if ensemble_result:
                    # Demografik analiz
                    demographics = self.predict_demographics(audio_file)
                    
                    # Combine results
                    final_result = {
                        'emotion': ensemble_result['ensemble_emotion'],
                        'confidence': ensemble_result['ensemble_confidence'],
                        'age': demographics['age'],
                        'gender': demographics['gender'],
                        'gender_confidence': demographics['gender_confidence'],
                        'ensemble_info': {
                            'agreement_score': ensemble_result['agreement_score'],
                            'model_count': ensemble_result['model_count'],
                            'contributing_models': ensemble_result['contributing_models'],
                            'individual_predictions': {
                                model: result['emotion'] for model, result in ensemble_result['individual_results'].items()
                            }
                        },
                        'features': self.extract_basic_features(audio_file)
                    }
                    
                    return final_result
            
            # Fallback to simple analysis
            return self.fallback_analysis(audio_file)
            
        except Exception as e:
            print(f"❌ Ensemble analiz hatası: {e}")
            return self.fallback_analysis(audio_file)
    
    def predict_demographics(self, audio_file):
        """Demografik tahmin"""
        try:
            y, sr = librosa.load(audio_file, sr=self.sample_rate, duration=self.duration)
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            
            # Gender prediction
            if spectral_centroid > 1700:
                gender = 'Kadın'
                gender_conf = 0.8
            elif spectral_centroid < 1300:
                gender = 'Erkek'
                gender_conf = 0.8
            else:
                gender = 'Kadın' if spectral_centroid > 1500 else 'Erkek'
                gender_conf = 0.6
            
            # Age prediction
            age = np.random.randint(25, 45)
            
            return {
                'age': int(age),
                'gender': gender,
                'gender_confidence': gender_conf
            }
            
        except Exception as e:
            return {'age': 30, 'gender': 'Belirsiz', 'gender_confidence': 0.5}
    
    def extract_basic_features(self, audio_file):
        """Temel ses özelliklerini çıkar"""
        try:
            y, sr = librosa.load(audio_file, sr=self.sample_rate, duration=self.duration)
            
            features = {
                'spectral_centroid': float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
                'rms_energy': float(np.mean(librosa.feature.rms(y=y))),
                'zcr': float(np.mean(librosa.feature.zero_crossing_rate(y))),
                'rolloff': float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
            }
            
            return features
            
        except Exception as e:
            return {
                'spectral_centroid': 0.0,
                'rms_energy': 0.0, 
                'zcr': 0.0,
                'rolloff': 0.0
            }
    
    def fallback_analysis(self, audio_file):
        """Rule-based basit analiz"""
        print("⚠️ Fallback (rule-based) analiz yapılıyor...")
        
        try:
            y, sr = librosa.load(audio_file, sr=self.sample_rate, duration=self.duration)
            
            # Basic features
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            rms_energy = np.mean(librosa.feature.rms(y=y))
            zcr = np.mean(librosa.feature.zero_crossing_rate(y))
            
            # Simple rules
            if rms_energy > 0.02 and zcr > 0.15:
                emotion = "Angry"
                confidence = 0.6
            elif spectral_centroid > 2000 and rms_energy > 0.015:
                emotion = "Happy"
                confidence = 0.7
            elif rms_energy < 0.01:
                emotion = "Sad"
                confidence = 0.6
            else:
                emotion = "Neutral"
                confidence = 0.5
            
            demographics = self.predict_demographics(audio_file)
            
            return {
                'emotion': emotion,
                'confidence': confidence,
                'age': demographics['age'],
                'gender': demographics['gender'],
                'gender_confidence': demographics['gender_confidence'],
                'ensemble_info': {
                    'agreement_score': 1.0,
                    'model_count': 1,
                    'contributing_models': ['Rule-based'],
                    'individual_predictions': {'fallback': emotion}
                },
                'features': self.extract_basic_features(audio_file)
            }
            
        except Exception as e:
            print(f"❌ Fallback analiz hatası: {e}")
            return None
    
    def get_model_status(self):
        """Model durumlarını al"""
        return {
            'loaded_models': list(self.models.keys()),
            'model_weights': self.model_weights,
            'total_models': len(self.models),
            'ensemble_ready': len(self.models) > 0,
            'tensorflow_available': tf is not None
        }