# API Dokümantasyonu

## Endpoints

### POST /api/analyze
Ses dosyası analizi yapar.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: audio file (max 10MB)

**Response:**
```json
{
    "success": true,
    "data": {
        "analysis": {
            "emotion": "Happy",
            "confidence": 0.85,
            "age": 25,
            "gender": "Kadın",
            "gender_confidence": 0.78,
            "features": {
                "spectral_centroid": 1850.5,
                "rms_energy": 0.024,
                "zcr": 0.12,
                "rolloff": 2100.3,
                "mfcc_mean": [...]
            }
        },
        "visualization": "base64_encoded_image",
        "timestamp": "2024-01-01T12:00:00"
    }
}