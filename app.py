import os
import io
import base64
import tempfile
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import soundfile as sf

app = Flask(__name__)
CORS(app)

# ── Config from environment variables (set these in Render dashboard) ─────────
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "FGY2WhTYpPnrIDTdsKH5")  # Laura
ELEVENLABS_URL = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"

RVC_MODEL_PATH  = os.environ.get("RVC_MODEL_PATH",  "models/BMO.pth")
RVC_INDEX_PATH  = os.environ.get("RVC_INDEX_PATH",  "models/BMO.index")
# ─────────────────────────────────────────────────────────────────────────────

# Load RVC pipeline once at startup
rvc_pipeline = None

def load_rvc():
    global rvc_pipeline
    try:
        from rvc_python.infer import RVCInference
        rvc_pipeline = RVCInference()
        rvc_pipeline.load_model(RVC_MODEL_PATH, RVC_INDEX_PATH)
        print("✅ RVC model loaded")
    except Exception as e:
        print(f"⚠️ RVC load failed: {e}")
        rvc_pipeline = None

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "rvc_loaded": rvc_pipeline is not None})

@app.route("/speak", methods=["POST"])
def speak():
    data = request.get_json()
    text = data.get("text", "").strip()
    lang = data.get("lang", "en")  # "en" or "ar"

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # ── Step 1: ElevenLabs → WAV ──────────────────────────────────────────────
    try:
        el_response = requests.post(
            ELEVENLABS_URL,
            headers={
                "xi-api-key": ELEVENLABS_API_KEY,
                "Content-Type": "application/json",
                "Accept": "audio/wav",
            },
            json={
                "text": text,
                "model_id": "eleven_multilingual_v2",
                "language_code": "en" if lang == "en" else "ar",
                "voice_settings": {
                    "stability": 0.20,
                    "similarity_boost": 0.75,
                    "style": 0.80,
                    "use_speaker_boost": True,
                    "speed": 0.85,
                },
            },
            timeout=15,
        )
        if not el_response.ok:
            return jsonify({"error": f"ElevenLabs error: {el_response.status_code}"}), 502

        wav_bytes = el_response.content

    except Exception as e:
        return jsonify({"error": f"ElevenLabs request failed: {str(e)}"}), 502

    # ── Step 2: RVC voice conversion → BMO voice ──────────────────────────────
    if rvc_pipeline is not None:
        try:
            # Write input WAV to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in:
                tmp_in.write(wav_bytes)
                tmp_in_path = tmp_in.name

            tmp_out_path = tmp_in_path.replace(".wav", "_bmo.wav")

            # Run RVC conversion
            rvc_pipeline.infer(
                input_path=tmp_in_path,
                output_path=tmp_out_path,
                f0_up_key=4,          # pitch shift in semitones (positive = higher)
                f0_method="rmvpe",    # best quality pitch extraction
                index_rate=0.75,      # how much to use the index file (0-1)
                protect=0.33,
            )

            with open(tmp_out_path, "rb") as f:
                wav_bytes = f.read()

            # Cleanup temp files
            os.unlink(tmp_in_path)
            os.unlink(tmp_out_path)

        except Exception as e:
            print(f"⚠️ RVC conversion failed, using raw ElevenLabs audio: {e}")
            # fallback to raw ElevenLabs audio — don't crash the app

    # ── Step 3: Return base64 WAV to the app ─────────────────────────────────
    b64 = base64.b64encode(wav_bytes).decode("utf-8")
    return jsonify({"audio": b64, "format": "wav"})


if __name__ == "__main__":
    load_rvc()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
