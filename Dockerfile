FROM python:3.10

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y ffmpeg
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install openai-whisper

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
