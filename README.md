# BMO Voice Server — Render Deployment Guide

## What this does
Text → ElevenLabs TTS → RVC (BMO voice model) → returns BMO's actual voice as WAV

---

## Step 1 — Prepare your files

After downloading and extracting BMO.zip you should have:
- `BMO.pth`
- `BMO.index`

Create a `models/` folder in this project and put both files inside:
```
bmo-server/
  app.py
  requirements.txt
  render.yaml
  models/
    BMO.pth
    BMO.index
```

---

## Step 2 — Push to GitHub

```bash
git init
git add .
git commit -m "BMO voice server"
git remote add origin https://github.com/YOUR_USERNAME/bmo-server.git
git push -u origin main
```

---

## Step 3 — Deploy on Render

1. Go to https://render.com → sign up free
2. New → Web Service → connect your GitHub repo
3. Settings:
   - **Runtime:** Python
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `gunicorn app:app --timeout 60`
4. Add environment variable:
   - `ELEVENLABS_API_KEY` → your key
5. Click Deploy

Render gives you a free URL like:
`https://bmo-voice-server.onrender.com`

---

## Step 4 — Update your React Native app

In BmoChatScreen.tsx, replace the ElevenLabs call with:

```js
const BMO_SERVER_URL = "https://bmo-voice-server.onrender.com/speak";

const response = await fetch(BMO_SERVER_URL, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ text: spokenText, lang: currentLang }),
});
const data = await response.json();
const dataUri = `data:audio/wav;base64,${data.audio}`;
```

---

## API

**POST /speak**
```json
{ "text": "Hello!", "lang": "en" }
```
Returns:
```json
{ "audio": "<base64 wav>", "format": "wav" }
```

**GET /health**
Returns server status and whether RVC model is loaded.

---

## Notes
- Free Render tier spins down after 15min inactivity → first request after idle takes ~30s
- Upgrade to $7/month Render plan to keep it always on
- RVC inference takes ~2-4 seconds on Render's free CPU
