import os
import io
import base64
import tempfile
import requests
import numpy as np
import soundfile as sf
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ── Config ────────────────────────────────────────────────────────────────────
ELEVENLABS_API_KEY  = os.environ.get("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "FGY2WhTYpPnrIDTdsKH5")
ELEVENLABS_URL      = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
RVC_MODEL_PATH      = os.environ.get("RVC_MODEL_PATH",  "models/BMO.pth")
RVC_INDEX_PATH      = os.environ.get("RVC_INDEX_PATH",  "models/BMO.index")
# ─────────────────────────────────────────────────────────────────────────────

# Load RVC pipeline once at startup
vc = None

def load_rvc():
    global vc
    try:
        import torch
        from rvc_infer import get_vc, vc_single
        get_vc(RVC_MODEL_PATH)
        vc = vc_single
        print("✅ RVC loaded")
    except Exception as e:
        print(f"⚠️ RVC load failed: {e}")
        vc = None

def rvc_infer(wav_bytes: bytes) -> bytes:
    """Run RVC voice conversion on raw WAV bytes, return converted WAV bytes."""
    import torch
    import librosa
    import soundfile as sf
    from scipy.io import wavfile
    from rvc_infer import vc_single

    # Write input to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        in_path = f.name

    out_path = in_path.replace(".wav", "_out.wav")

    try:
        vc_single(
            sid=0,
            input_audio_path=in_path,
            f0_up_key=4,           # +4 semitones = higher/more child-like
            f0_method="harvest",   # pitch extraction method
            file_index=RVC_INDEX_PATH,
            index_rate=0.75,
            filter_radius=3,
            resample_sr=0,
            rms_mix_rate=0.25,
            protect=0.33,
            output_audio_path=out_path,
        )
        with open(out_path, "rb") as f:
            return f.read()
    finally:
        for p in [in_path, out_path]:
            try: os.unlink(p)
            except: pass

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "rvc_loaded": vc is not None})

@app.route("/speak", methods=["POST"])
def speak():
    data     = request.get_json()
    text     = data.get("text", "").strip()
    lang     = data.get("lang", "en")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # ── Step 1: ElevenLabs → WAV ──────────────────────────────────────────────
    try:
        el_res = requests.post(
            ELEVENLABS_URL,
            headers={
                "xi-api-key":   ELEVENLABS_API_KEY,
                "Content-Type": "application/json",
                "Accept":       "audio/wav",
            },
            json={
                "text":       text,
                "model_id":   "eleven_multilingual_v2",
                "language_code": "en" if lang == "en" else "ar",
                "voice_settings": {
                    "stability":        0.20,
                    "similarity_boost": 0.75,
                    "style":            0.80,
                    "use_speaker_boost": True,
                    "speed":            0.85,
                },
            },
            timeout=20,
        )
        if not el_res.ok:
            return jsonify({"error": f"ElevenLabs {el_res.status_code}"}), 502
        wav_bytes = el_res.content
    except Exception as e:
        return jsonify({"error": f"ElevenLabs failed: {e}"}), 502

    # ── Step 2: RVC → BMO voice ───────────────────────────────────────────────
    try:
        wav_bytes = rvc_infer(wav_bytes)
    except Exception as e:
        print(f"⚠️ RVC failed, using raw audio: {e}")
        # Silently fall back to raw ElevenLabs audio

    # ── Step 3: Return base64 WAV ─────────────────────────────────────────────
    b64 = base64.b64encode(wav_bytes).decode("utf-8")
    return jsonify({"audio": b64, "format": "wav"})

# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    load_rvc()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)