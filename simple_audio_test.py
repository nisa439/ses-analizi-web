# simple_audio_test.py - Basit ses testi

import numpy as np
import soundfile as sf
import os

def create_valid_test_audio():
    """Geçerli test ses dosyaları oluştur"""
    print("🎵 Geçerli test ses dosyaları oluşturuluyor...")
    
    # Parametreler
    sample_rate = 22050
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # uploads klasörü oluştur
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    
    # Farklı test sinyalleri
    test_signals = {
        'test_sine.wav': np.sin(2 * np.pi * 440 * t) * 0.3,  # 440 Hz sinüs
        'test_mixed.wav': (np.sin(2 * np.pi * 330 * t) + np.sin(2 * np.pi * 660 * t)) * 0.2,  # Karışık
        'test_noise.wav': np.random.normal(0, 0.1, len(t))  # Beyaz gürültü
    }
    
    created_files = []
    
    for filename, signal in test_signals.items():
        try:
            filepath = os.path.join('uploads', filename)
            
            # Ses dosyasını kaydet (16-bit PCM WAV)
            sf.write(filepath, signal, sample_rate, subtype='PCM_16')
            
            # Dosya kontrolü
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                print(f"✅ {filename} oluşturuldu ({file_size} bytes)")
                created_files.append(filepath)
            
        except Exception as e:
            print(f"❌ {filename} oluşturma hatası: {e}")
    
    return created_files

def test_audio_loading():
    """Ses dosyası yükleme testi"""
    print("\n🔍 Ses dosyası yükleme testi...")
    
    # Test dosyalarını oluştur
    created_files = create_valid_test_audio()
    
    if not created_files:
        print("❌ Test dosyası oluşturulamadı!")
        return False
    
    # Librosa ile test et
    try:
        import librosa
        
        for filepath in created_files:
            print(f"\n📁 Test: {os.path.basename(filepath)}")
            
            try:
                # Librosa ile yükle
                y, sr = librosa.load(filepath, sr=22050, duration=3)
                
                print(f"  ✅ Yüklendi: {len(y)} samples, {sr} Hz")
                print(f"  📊 Shape: {y.shape}")
                print(f"  📈 Min/Max: {y.min():.3f} / {y.max():.3f}")
                
                # Mel-spectrogram test
                mel_spec = librosa.feature.melspectrogram(
                    y=y, sr=sr, n_mels=128, hop_length=256, n_fft=1024
                )
                print(f"  🎼 Mel-spec shape: {mel_spec.shape}")
                
                return True  # İlk başarılı dosya ile dön
                
            except Exception as e:
                print(f"  ❌ Librosa hatası: {e}")
                continue
        
        print("❌ Hiçbir dosya yüklenemedi!")
        return False
        
    except ImportError:
        print("❌ Librosa import hatası!")
        return False

def manual_preprocessing_test():
    """Manuel preprocessing testi - TensorFlow/model olmadan"""
    print("\n🧪 Manuel preprocessing testi...")
    
    # Test ses verisi oluştur
    sample_rate = 22050
    duration = 3
    t = np.linspace(0, duration, sample_rate * duration)
    
    # Basit sinüs dalgası
    test_audio = np.sin(2 * np.pi * 440 * t) * 0.5
    
    print(f"✅ Test audio shape: {test_audio.shape}")
    print(f"📊 Sample rate: {sample_rate} Hz")
    print(f"⏱️ Duration: {duration} s")
    
    # 1D CNN formatı
    input_1d = test_audio.reshape(1, -1, 1)
    print(f"🔧 1D CNN input shape: {input_1d.shape}")
    
    # 2D CNN formatı (mel-spectrogram simülasyonu)
    # Reshape to 2D (mel-bins, time-frames)
    n_mels = 128
    # Simulate mel-spectrogram dimensions
    n_frames = 259  # Typical for 3s audio with hop_length=256
    
    # Create fake mel-spectrogram
    fake_mel = np.random.randn(n_mels, n_frames)
    input_2d = fake_mel.reshape(1, n_mels, n_frames, 1)
    print(f"🔧 2D CNN input shape: {input_2d.shape}")
    
    # Emotion prediction simülasyonu
    emotion_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad']
    
    # Fake predictions
    for model_name, input_shape in [('1D CNN', input_1d.shape), ('2D CNN', input_2d.shape)]:
        fake_probs = np.random.rand(6)
        fake_probs = fake_probs / np.sum(fake_probs)
        
        emotion_idx = np.argmax(fake_probs)
        emotion = emotion_classes[emotion_idx]
        confidence = fake_probs[emotion_idx]
        
        print(f"\n🤖 {model_name}:")
        print(f"  📊 Input: {input_shape}")
        print(f"  🎭 Emotion: {emotion}")
        print(f"  📈 Confidence: {confidence:.3f}")
    
    return True

if __name__ == "__main__":
    print("🚀 Basit Ses Testi Başlıyor...")
    print("=" * 50)
    
    # 1. Test dosyası oluştur ve yükle
    success1 = test_audio_loading()
    
    # 2. Manuel preprocessing test
    success2 = manual_preprocessing_test()
    
    print("\n" + "=" * 50)
    
    if success1:
        print("✅ Ses dosyası yükleme BAŞARILI!")
        print("🎯 Modeliniz bu input formatlarını bekliyor:")
        print("   - 2D CNN: (1, 128, 259, 1) mel-spectrogram")
        print("   - 1D CNN: (1, 66150, 1) raw audio")
        print("   - 6 emotion classes")
    elif success2:
        print("✅ Manuel test BAŞARILI!")
        print("🎯 Input shape'ler doğru görünüyor")
    else:
        print("❌ Test başarısız!")
    
    print("\n🎯 Model ayarları için bu bilgileri kullanabiliriz!")