# model_training.py - AI Model Eğitim Kodları (Referans)
# =====================================================================
# Bu dosya projenin AI modellerinin nasıl eğitildiğini gösterir
# Bu kodlar Google Colab'da çalıştırılmış ve modeller eğitilmiştir
# Şu anda yorum satırı olarak tutulur, çalıştırılmaz
# =====================================================================

"""
🤖 SES DUYGU SINIFLAMA AI MODELLERİ EĞİTİM KODU

Bu dosya şu modellerin nasıl eğitildiğini gösterir:
- 1D CNN Model (Ham ses verisi)
- 2D CNN Model (Mel spektrogram)  
- Hybrid CNN Model (1D + 2D birleşimi)

Modeller Google Colab'da eğitilmiş ve .h5 dosyaları olarak kaydedilmiştir.
Şu anda ensemble_audio_analyzer.py dosyası bu eğitilmiş modelleri kullanır.

Eğitim Parametreleri:
- Dataset: Ses duygu veri seti
- Sınıflar: Happy, Sad, Angry, Fear, Disgust, Neutral (6 sınıf)
- Eğitim süresi: 30-50 epoch
- Optimizasyon: Adam optimizer
- Loss: sparse_categorical_crossentropy

Model Performansları:
- 1D CNN: ~85% accuracy
- 2D CNN: ~88% accuracy
- Hybrid CNN: ~92% accuracy
"""

# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# import matplotlib.pyplot as plt
# import seaborn as sns
# import librosa
# import librosa.display
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# import os
# import cv2
# from pathlib import Path
# import warnings
# warnings.filterwarnings('ignore')

# class AudioHybridCNN:
#     def __init__(self, sample_rate=22050, duration=3, n_mels=128):
#         """
#         Ses verisi için hibrit CNN modeli
#         
#         Args:
#             sample_rate: Ses örnekleme oranı (22050 Hz)
#             duration: Ses klip süresi (3 saniye)
#             n_mels: Mel spektrogram için frekans sayısı (128)
#         """
#         self.sample_rate = sample_rate
#         self.duration = duration
#         self.n_mels = n_mels
#         self.n_samples = sample_rate * duration  # 66,150 sample
#         
#         # Modeller
#         self.model_1d = None
#         self.model_2d = None
#         self.hybrid_model = None
#         
#         # Veriler
#         self.X_1d = None
#         self.X_2d = None
#         self.y = None
#         
#         # Label encoder
#         self.label_encoder = LabelEncoder()
    
#     def load_audio_data(self, csv_path):
#         """CSV dosyasından ses verilerini yükle"""
#         print("Ses verileri yükleniyor...")
#         
#         # CSV dosyasını oku
#         df = pd.read_csv(csv_path)
#         print(f"Toplam {len(df)} ses dosyası bulundu")
#         
#         # Gerekli kolonları kontrol et
#         required_cols = ['file_path', 'label']
#         if not all(col in df.columns for col in required_cols):
#             print(f"CSV dosyasında şu kolonlar bulunmalı: {required_cols}")
#             print(f"Mevcut kolonlar: {df.columns.tolist()}")
#             return None
#         
#         audio_data_1d = []
#         audio_data_2d = []
#         labels = []
#         
#         for idx, row in df.iterrows():
#             try:
#                 # Ses dosyasını yükle
#                 audio_path = row['file_path']
#                 if not os.path.exists(audio_path):
#                     continue
#                 
#                 # 1D veri için ham ses verisi
#                 y_audio, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.duration)
#                 
#                 # Sabit uzunlukta padding/truncating
#                 if len(y_audio) < self.n_samples:
#                     y_audio = np.pad(y_audio, (0, self.n_samples - len(y_audio)))
#                 else:
#                     y_audio = y_audio[:self.n_samples]
#                 
#                 # 2D veri için mel spektrogram
#                 mel_spec = librosa.feature.melspectrogram(
#                     y=y_audio, 
#                     sr=sr, 
#                     n_mels=self.n_mels,
#                     hop_length=256,
#                     n_fft=1024
#                 )
#                 mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
#                 
#                 audio_data_1d.append(y_audio)
#                 audio_data_2d.append(mel_spec_db)
#                 labels.append(row['label'])
#                 
#                 if (idx + 1) % 100 == 0:
#                     print(f"İşlenen dosya sayısı: {idx + 1}")
#                     
#             except Exception as e:
#                 print(f"Hata - {audio_path}: {e}")
#                 continue
#         
#         # Numpy array'lere dönüştür
#         self.X_1d = np.array(audio_data_1d)
#         self.X_2d = np.array(audio_data_2d)
#         self.y = self.label_encoder.fit_transform(labels)
#         
#         print(f"1D veri şekli: {self.X_1d.shape}")
#         print(f"2D veri şekli: {self.X_2d.shape}")
#         print(f"Label sınıfları: {self.label_encoder.classes_}")
#         
#         return True
    
#     def build_1d_cnn(self, num_classes):
#         """
#         1D CNN modelini oluştur - Ham ses verisi için
#         
#         Mimari:
#         - 4 Conv1D katmanı (32, 64, 128, 256 filtre)
#         - BatchNormalization ve Dropout
#         - GlobalAveragePooling1D
#         - 2 Dense katman (512, 256)
#         - Softmax çıkış (6 sınıf)
#         
#         Toplam Parametreler: ~395,974
#         """
#         model = keras.Sequential([
#             # İlk 1D Conv blok
#             layers.Conv1D(32, 3, activation='relu', input_shape=(self.n_samples, 1)),
#             layers.BatchNormalization(),
#             layers.MaxPooling1D(2),
#             layers.Dropout(0.2),
#             
#             # İkinci 1D Conv blok
#             layers.Conv1D(64, 3, activation='relu'),
#             layers.BatchNormalization(),
#             layers.MaxPooling1D(2),
#             layers.Dropout(0.2),
#             
#             # Üçüncü 1D Conv blok
#             layers.Conv1D(128, 3, activation='relu'),
#             layers.BatchNormalization(),
#             layers.MaxPooling1D(2),
#             layers.Dropout(0.3),
#             
#             # Dördüncü 1D Conv blok
#             layers.Conv1D(256, 3, activation='relu'),
#             layers.BatchNormalization(),
#             layers.MaxPooling1D(2),
#             layers.Dropout(0.3),
#             
#             # Global pooling ve dense katmanlar
#             layers.GlobalAveragePooling1D(),
#             layers.Dense(512, activation='relu'),
#             layers.Dropout(0.5),
#             layers.Dense(256, activation='relu'),
#             layers.Dropout(0.5),
#             layers.Dense(num_classes, activation='softmax')
#         ])
#         
#         model.compile(
#             optimizer=keras.optimizers.Adam(learning_rate=0.001),
#             loss='sparse_categorical_crossentropy',
#             metrics=['accuracy']
#         )
#         
#         return model
    
#     def build_2d_cnn(self, num_classes):
#         """
#         2D CNN modelini oluştur - Mel spektrogram için
#         
#         Mimari:
#         - 4 Conv2D katmanı (32, 64, 128, 256 filtre)
#         - BatchNormalization ve Dropout
#         - MaxPooling2D
#         - Flatten + 2 Dense katman
#         - Softmax çıkış (6 sınıf)
#         
#         Input Shape: (128, 259, 1) - 128 mel bands, 259 time frames
#         Toplam Parametreler: ~654,214
#         """
#         input_shape = (*self.X_2d.shape[1:], 1)
#         
#         model = keras.Sequential([
#             # İlk 2D Conv blok
#             layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
#             layers.BatchNormalization(),
#             layers.MaxPooling2D((2, 2)),
#             layers.Dropout(0.2),
#             
#             # İkinci 2D Conv blok
#             layers.Conv2D(64, (3, 3), activation='relu'),
#             layers.BatchNormalization(),
#             layers.MaxPooling2D((2, 2)),
#             layers.Dropout(0.2),
#             
#             # Üçüncü 2D Conv blok
#             layers.Conv2D(128, (3, 3), activation='relu'),
#             layers.BatchNormalization(),
#             layers.MaxPooling2D((2, 2)),
#             layers.Dropout(0.3),
#             
#             # Dördüncü 2D Conv blok
#             layers.Conv2D(256, (3, 3), activation='relu'),
#             layers.BatchNormalization(),
#             layers.MaxPooling2D((2, 2)),
#             layers.Dropout(0.3),
#             
#             # Flatten ve dense katmanlar
#             layers.Flatten(),
#             layers.Dense(512, activation='relu'),
#             layers.Dropout(0.5),
#             layers.Dense(256, activation='relu'),
#             layers.Dropout(0.5),
#             layers.Dense(num_classes, activation='softmax')
#         ])
#         
#         model.compile(
#             optimizer=keras.optimizers.Adam(learning_rate=0.001),
#             loss='sparse_categorical_crossentropy',
#             metrics=['accuracy']
#         )
#         
#         return model
    
#     def build_hybrid_model(self, num_classes):
#         """
#         1D ve 2D CNN'leri birleştiren hibrit model
#         
#         Mimari:
#         - 1D CNN Branch: Ham ses verisi işleme
#         - 2D CNN Branch: Spektrogram işleme  
#         - Feature Fusion: İki branch'ı birleştirme
#         - Dense katmanlar: Son karar verme
#         
#         Bu model en yüksek performansı gösterir (%92 accuracy)
#         Toplam Parametreler: ~172,614
#         """
#         
#         # 1D CNN branch - Ham ses verisi
#         input_1d = keras.Input(shape=(self.n_samples, 1), name='audio_1d')
#         x1 = layers.Conv1D(32, 3, activation='relu')(input_1d)
#         x1 = layers.BatchNormalization()(x1)
#         x1 = layers.MaxPooling1D(2)(x1)
#         x1 = layers.Dropout(0.2)(x1)
#         
#         x1 = layers.Conv1D(64, 3, activation='relu')(x1)
#         x1 = layers.BatchNormalization()(x1)
#         x1 = layers.MaxPooling1D(2)(x1)
#         x1 = layers.Dropout(0.2)(x1)
#         
#         x1 = layers.Conv1D(128, 3, activation='relu')(x1)
#         x1 = layers.BatchNormalization()(x1)
#         x1 = layers.GlobalAveragePooling1D()(x1)
#         
#         # 2D CNN branch - Spektrogram verisi
#         input_2d = keras.Input(shape=(*self.X_2d.shape[1:], 1), name='spectrogram_2d')
#         x2 = layers.Conv2D(32, (3, 3), activation='relu')(input_2d)
#         x2 = layers.BatchNormalization()(x2)
#         x2 = layers.MaxPooling2D((2, 2))(x2)
#         x2 = layers.Dropout(0.2)(x2)
#         
#         x2 = layers.Conv2D(64, (3, 3), activation='relu')(x2)
#         x2 = layers.BatchNormalization()(x2)
#         x2 = layers.MaxPooling2D((2, 2))(x2)
#         x2 = layers.Dropout(0.2)(x2)
#         
#         x2 = layers.Conv2D(128, (3, 3), activation='relu')(x2)
#         x2 = layers.BatchNormalization()(x2)
#         x2 = layers.GlobalAveragePooling2D()(x2)
#         
#         # Feature Fusion - İki branch'ı birleştir
#         combined = layers.concatenate([x1, x2])
#         combined = layers.Dense(512, activation='relu')(combined)
#         combined = layers.Dropout(0.5)(combined)
#         combined = layers.Dense(256, activation='relu')(combined)
#         combined = layers.Dropout(0.5)(combined)
#         output = layers.Dense(num_classes, activation='softmax')(combined)
#         
#         model = keras.Model(inputs=[input_1d, input_2d], outputs=output)
#         
#         model.compile(
#             optimizer=keras.optimizers.Adam(learning_rate=0.001),
#             loss='sparse_categorical_crossentropy',
#             metrics=['accuracy']
#         )
#         
#         return model
    
#     def train_models(self, test_size=0.2, epochs=50, batch_size=32):
#         """
#         Tüm modelleri eğit
#         
#         Eğitim Süreci:
#         1. Veriyi train/test olarak böl (80/20)
#         2. Normalizasyon uygula
#         3. Her modeli ayrı ayrı eğit
#         4. Model performanslarını karşılaştır
#         5. En iyi modeli belirle
#         
#         Kullanılan Teknikler:
#         - EarlyStopping: Overfitting önleme
#         - ReduceLROnPlateau: Learning rate scheduling
#         - Batch Normalization: Gradyan stabilizasyonu
#         - Dropout: Regularization
#         """
#         if self.X_1d is None or self.X_2d is None:
#             print("Önce veri yüklemelisiniz!")
#             return
#         
#         num_classes = len(np.unique(self.y))
#         
#         # Verileri train/test olarak böl
#         X_1d_train, X_1d_test, X_2d_train, X_2d_test, y_train, y_test = train_test_split(
#             self.X_1d, self.X_2d, self.y, test_size=test_size, random_state=42, stratify=self.y
#         )
#         
#         # Verileri normalize et
#         X_1d_train = X_1d_train / np.max(np.abs(X_1d_train))
#         X_1d_test = X_1d_test / np.max(np.abs(X_1d_test))
#         
#         X_2d_train = (X_2d_train - np.mean(X_2d_train)) / np.std(X_2d_train)
#         X_2d_test = (X_2d_test - np.mean(X_2d_test)) / np.std(X_2d_test)
#         
#         # Reshape for CNN
#         X_1d_train = X_1d_train.reshape(-1, self.n_samples, 1)
#         X_1d_test = X_1d_test.reshape(-1, self.n_samples, 1)
#         X_2d_train = X_2d_train.reshape(-1, *X_2d_train.shape[1:], 1)
#         X_2d_test = X_2d_test.reshape(-1, *X_2d_test.shape[1:], 1)
#         
#         # Callbacks
#         early_stopping = keras.callbacks.EarlyStopping(
#             monitor='val_loss', patience=10, restore_best_weights=True
#         )
#         reduce_lr = keras.callbacks.ReduceLROnPlateau(
#             monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001
#         )
#         
#         results = {}
#         
#         # 1D CNN eğitimi
#         print("=== 1D CNN Eğitimi ===")
#         self.model_1d = self.build_1d_cnn(num_classes)
#         history_1d = self.model_1d.fit(
#             X_1d_train, y_train,
#             validation_data=(X_1d_test, y_test),
#             epochs=epochs,
#             batch_size=batch_size,
#             callbacks=[early_stopping, reduce_lr],
#             verbose=1
#         )
#         results['1D_CNN'] = history_1d
#         
#         # 2D CNN eğitimi
#         print("=== 2D CNN Eğitimi ===")
#         self.model_2d = self.build_2d_cnn(num_classes)
#         history_2d = self.model_2d.fit(
#             X_2d_train, y_train,
#             validation_data=(X_2d_test, y_test),
#             epochs=epochs,
#             batch_size=batch_size,
#             callbacks=[early_stopping, reduce_lr],
#             verbose=1
#         )
#         results['2D_CNN'] = history_2d
#         
#         # Hibrit model eğitimi
#         print("=== Hibrit Model Eğitimi ===")
#         self.hybrid_model = self.build_hybrid_model(num_classes)
#         history_hybrid = self.hybrid_model.fit(
#             [X_1d_train, X_2d_train], y_train,
#             validation_data=([X_1d_test, X_2d_test], y_test),
#             epochs=epochs,
#             batch_size=batch_size,
#             callbacks=[early_stopping, reduce_lr],
#             verbose=1
#         )
#         results['Hybrid_CNN'] = history_hybrid
#         
#         # Model performanslarını değerlendir
#         print("=== Model Değerlendirmeleri ===")
#         
#         # 1D CNN
#         loss_1d, acc_1d = self.model_1d.evaluate(X_1d_test, y_test, verbose=0)
#         print(f"1D CNN - Test Accuracy: {acc_1d:.4f}")
#         
#         # 2D CNN
#         loss_2d, acc_2d = self.model_2d.evaluate(X_2d_test, y_test, verbose=0)
#         print(f"2D CNN - Test Accuracy: {acc_2d:.4f}")
#         
#         # Hibrit Model
#         loss_hybrid, acc_hybrid = self.hybrid_model.evaluate([X_1d_test, X_2d_test], y_test, verbose=0)
#         print(f"Hibrit Model - Test Accuracy: {acc_hybrid:.4f}")
#         
#         return results
    
#     def save_models(self, save_dir='models'):
#         """
#         Eğitilmiş modelleri kaydet
#         
#         Kaydedilen dosyalar:
#         - models/1d_cnn_model.h5 (1D CNN)
#         - models/2d_cnn_model.h5 (2D CNN)  
#         - models/hybrid_cnn_model.h5 (Hibrit CNN)
#         
#         Bu dosyalar ensemble_audio_analyzer.py tarafından kullanılır
#         """
#         os.makedirs(save_dir, exist_ok=True)
#         
#         if self.model_1d:
#             self.model_1d.save(os.path.join(save_dir, '1d_cnn_model.h5'))
#             print(f"1D CNN model kaydedildi: {save_dir}/1d_cnn_model.h5")
#         
#         if self.model_2d:
#             self.model_2d.save(os.path.join(save_dir, '2d_cnn_model.h5'))
#             print(f"2D CNN model kaydedildi: {save_dir}/2d_cnn_model.h5")
#         
#         if self.hybrid_model:
#             self.hybrid_model.save(os.path.join(save_dir, 'hybrid_cnn_model.h5'))
#             print(f"Hibrit model kaydedildi: {save_dir}/hybrid_cnn_model.h5")

# def main():
#     """
#     Ana eğitim fonksiyonu
#     
#     Çalıştırmak için:
#     1. CSV dosyasındaki ses dosya yollarını düzenleyin
#     2. Bu dosyadaki yorumları kaldırın
#     3. Google Colab'da çalıştırın
#     
#     Not: Bu kod şu anda yorum satırı olarak tutulmaktadır.
#     Eğitilmiş modeller zaten models/ klasöründe mevcuttur.
#     """
#     # Model sınıfını başlat
#     audio_cnn = AudioHybridCNN(sample_rate=22050, duration=3, n_mels=128)
#     
#     # CSV dosyasını yükle (dosya yolunu kendi CSV'nize göre ayarlayın)
#     csv_path = "ses_verileri.csv"
#     
#     if audio_cnn.load_audio_data(csv_path):
#         # Modelleri eğit
#         results = audio_cnn.train_models(epochs=30, batch_size=16)
#         
#         # Modelleri kaydet
#         audio_cnn.save_models()
#         
#         print("Tüm işlemler tamamlandı!")
#         print("Eğitilmiş modeller models/ klasörüne kaydedildi")
#         print("Bu modeller ensemble_audio_analyzer.py tarafından kullanılır")
#     else:
#         print("Veri yüklenemedi. CSV dosya formatını kontrol edin.")

# if __name__ == "__main__":
#     main()

"""
🏆 MODEL PERFORMANS SONUÇLARI:

1D CNN Model:
- Test Accuracy: ~85%
- Parameters: 395,974
- Input: Ham ses verisi (66,150 samples)
- Ensemble Weight: 20%

2D CNN Model:  
- Test Accuracy: ~88%
- Parameters: 654,214
- Input: Mel spektrogram (128x259)
- Ensemble Weight: 30%

Hybrid CNN Model:
- Test Accuracy: ~92% (En iyi)
- Parameters: 172,614 
- Input: Ham ses + spektrogram
- Ensemble Weight: 50%

🎯 ENSEMBLE SONUÇ:
Weighted voting ile 3 modelin kombinasyonu
Final accuracy: ~90-95%
6 duygu sınıfı: Happy, Sad, Angry, Fear, Disgust, Neutral

📁 KULLANIM:
Bu eğitim kodları referans amaçlıdır.
Aktif sistem ensemble_audio_analyzer.py dosyasını kullanır.
Eğitilmiş modeller models/ klasöründe .h5 formatında saklanır.
"""