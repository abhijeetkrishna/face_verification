FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# -----------------------
# System dependencies
# -----------------------
RUN apt-get update && apt-get install -y \
    git \
    libglib2.0-0 \
    libgl1 \
    libjpeg62-turbo \
    libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

# -----------------------
# Python dependencies
# -----------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------
# Application code
# -----------------------
COPY scripts/predict.py ./predict.py

# -----------------------
# Model artifacts
# -----------------------
COPY model/ ./model/

# -----------------------
# Expose & run
# -----------------------
EXPOSE 9696

ENTRYPOINT ["gunicorn", "--workers=2", "--threads=2", "--bind=0.0.0.0:9696", "predict:app"]

