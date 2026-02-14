import base64
import tempfile
import librosa
import numpy as np
import torch
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

# ==========================
# SETTINGS
# ==========================
API_KEY = "sk_test_123456789"
MODEL_PATH = "./FINAL_AI_DETECTOR_SUPER_v2"
MAX_SEC = 7

# ==========================
# LOAD MODEL ON START
# ==========================
print("Loading model...")

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_PATH)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

print("Model loaded!")

# ==========================
# FASTAPI APP
# ==========================
app = FastAPI()

# ==========================
# REQUEST FORMAT
# ==========================
class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

# ==========================
# AUDIO PREPROCESS
# ==========================
def process_audio(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    y = y[:16000 * MAX_SEC]
    y = y.astype(np.float32)

    inputs = feature_extractor(y, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)[0]

    ai_prob = probs[1].item()
    human_prob = probs[0].item()

    # 🔥 IMPORTANT CHANGE
    if ai_prob > 0.45:
        label = "AI_GENERATED"
        conf = ai_prob
    else:
        label = "HUMAN"
        conf = human_prob

    return label, conf


# ==========================
# API ENDPOINT
# ==========================
@app.post("/api/voice-detection")
def detect_voice(req: VoiceRequest, x_api_key: str = Header(...)):

    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        # decode base64
        audio_bytes = base64.b64decode(req.audioBase64)

        # save temp mp3
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            f.write(audio_bytes)
            temp_path = f.name

        label, conf = process_audio(temp_path)

        explanation = (
            "Speech patterns match AI synthesis"
            if label == "AI_GENERATED"
            else "Natural human speech variations detected"
        )

        return {
            "status": "success",
            "language": req.language,
            "classification": label,
            "confidenceScore": float(conf),
            "explanation": explanation
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
