from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import os
import uuid
from gtts import gTTS
import whisper
import openai
import subprocess

# OpenRouter key e modelo
openai.api_key = os.getenv("OPENROUTER_API_KEY")
openai.api_base = "https://openrouter.ai/api/v1"
openai_model = "mistral/mistral-7b-instruct"

app = FastAPI()

# CORS liberado para n8n
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Whisper local
model = whisper.load_model("base")

@app.post("/tts")
def tts(text: str = Form(...)):
    filename = f"output_{uuid.uuid4().hex}.mp3"
    tts = gTTS(text=text, lang='pt')
    tts.save(filename)
    return FileResponse(filename, media_type="audio/mpeg", filename="resposta.mp3")

@app.post("/stt")
def stt(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    result = model.transcribe(tmp_path)
    os.remove(tmp_path)
    return {"text": result["text"]}

@app.post("/chat")
def chat(text: str = Form(None), file: UploadFile = File(None)):
    if file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(file.file.read())
            tmp_path = tmp.name
        text_result = model.transcribe(tmp_path)
        os.remove(tmp_path)
        user_input = text_result["text"]
    else:
        user_input = text

    completion = openai.ChatCompletion.create(
        model=openai_model,
        messages=[{"role": "user", "content": user_input}],
    )
    response_text = completion.choices[0].message.content

    # Gerar áudio da resposta
    audio_filename = f"response_{uuid.uuid4().hex}.mp3"
    tts = gTTS(text=response_text, lang='pt')
    tts.save(audio_filename)

    return JSONResponse(
        content={"text": response_text, "audio_file": audio_filename}
    )

@app.get("/audio/{filename}")
def get_audio(filename: str):
    return FileResponse(filename, media_type="audio/mpeg", filename=filename)

@app.get("/")
def root():
    return {"status": "API de voz e IA ativa 🚀"}
