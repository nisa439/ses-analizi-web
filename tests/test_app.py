import unittest
import tempfile
import os
from app import app, AudioAnalyzer

class AudioAnalysisTestCase(unittest.TestCase):
    
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        self.analyzer = AudioAnalyzer()
    
    def test_health_endpoint(self):
        """API health endpoint testi"""
        response = self.app.get('/api/health')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['status'], 'healthy')
    
    def test_home_page(self):
        """Ana sayfa testi"""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'AI Ses Analizi', response.data)
    
    def test_about_page(self):
        """Hakkında sayfası testi"""
        response = self.app.get('/about')
        self.assertEqual(response.status_code, 200)
    
    def test_analyze_no_file(self):
        """Dosya olmadan analiz testi"""
        response = self.app.post('/api/analyze')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertFalse(data['success'])
    
    def test_analyzer_features(self):
        """Analyzer özelliklerini test et"""
        # Test verileri oluştur
        test_features = {
            'spectral_centroid': 1500.0,
            'rms_energy': 0.02,
            'zcr': 0.1,
            'rolloff': 2000.0
        }
        
        # Duygu tahmini testi
        emotion_result = self.analyzer._predict_emotion(
            1500.0, 0.02, 0.1, [-5, -3, -2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8]
        )
        
        self.assertIn('emotion', emotion_result)
        self.assertIn('confidence', emotion_result)
        self.assertIsInstance(emotion_result['confidence'], float)

if __name__ == '__main__':
    unittest.main()