FROM python:3.10-slim

# Avoid Python buffering issues
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (needed for torch / tokenizers)
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/predict.py .

# Copy model files
COPY model/ ./model/

# Expose port
EXPOSE 9696

# Run with gunicorn (same as before)
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
