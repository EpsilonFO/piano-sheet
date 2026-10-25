FROM python:3.12-slim

# ffmpeg decodes MP3, M4A, OGG... audio files.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU build of PyTorch: much smaller than the default CUDA build.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

ENV PIANO_SHEET_MODEL_DIR=/models \
    PIANO_SHEET_DATA_DIR=/data
# Bake the transcription model (~165 MB) into the image.
RUN piano-sheet download-model

VOLUME /data
EXPOSE 8000
CMD ["piano-sheet", "serve", "--host", "0.0.0.0", "--port", "8000"]
