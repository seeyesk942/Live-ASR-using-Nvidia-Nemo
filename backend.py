import json
import base64
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import nemo.collections.asr as nemo_asr
import torch
from googletrans import Translator

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global buffer
streaming_buffer = np.array([], dtype=np.int16)
SAMPLE_RATE = 16000
# Load NeMo RNNT model
print("Loading ASR model...")
# asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(
#     model_name="stt_en_fastconformer_hybrid_large_streaming_multi"
# )
asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained("stt_en_conformer_ctc_small")
asr_model.eval()
print("Model loaded!")

def transcribe_chunk(audio_int16: np.ndarray):
    audio_float = audio_int16.astype(np.float32) / 32768.0
    with torch.no_grad():
        hyps = asr_model.transcribe([audio_float], batch_size=1)
    text = hyps[0].text if hasattr(hyps[0], "text") else ""
    return text

@app.websocket("/ws/asr")
async def websocket_asr(ws: WebSocket):
    await ws.accept()
    global streaming_buffer
    try:
        synthesizing_sent = False
        while True:
            data = await ws.receive_text()
            obj = json.loads(data)
            if obj.get("type") == "audio":
                audio_b64 = obj.get("data")
                if audio_b64:
                    audio_chunk = np.frombuffer(base64.b64decode(audio_b64), dtype=np.int16)
                    streaming_buffer = np.concatenate([streaming_buffer, audio_chunk])

                if not synthesizing_sent:
                    await ws.send_text(json.dumps({"type": "partial", "text": "Synthesizing..."}))
                    synthesizing_sent = True

            elif obj.get("type") == "stop":
                translator = Translator()
                text = transcribe_chunk(streaming_buffer)
                detected_lang = str(translator.detect(text).lang)
                text = transcribe_chunk(streaming_buffer)
                streaming_buffer = np.array([], dtype=np.int16)
                if detected_lang=='en':
                    text_block = 'Original:'+text + ' ' + '{' + 'detected_language:'+ detected_lang + '|'  + "translation:" + 'NA' + '}'
                else:
                    translated_text = translator.translate(text, src=detected_lang, dest='en').text
                    text_block = 'Original:'+text + ' ' + '{' + 'detected_language:'+ detected_lang + '|'  + "translation:" + translated_text + '}'    
                await ws.send_text(json.dumps({"type": "final", "text": text_block}))
                await ws.close()
                break

    except Exception as e:
        print("WebSocket closed:", e)
        await ws.close()

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, log_level="info")
