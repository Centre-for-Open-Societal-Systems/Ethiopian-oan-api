# Use Python as the base image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Install system dependencies
# espeak-ng is required by Piper TTS (Amharic phonemization for Selam/Dawit/Meron voices).
RUN apt-get update && apt-get install -y \
    supervisor \
    gcc \
    g++ \
    python3-dev \
    curl \
    netcat-openbsd \
    espeak-ng \
    espeak-ng-data \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Copy application code
COPY . .

# Create logs directory for supervisord
RUN mkdir -p /app/logs

# Copy supervisor configuration
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose FastAPI port
EXPOSE 8000

# Copy entrypoint script outside /app so the volume mount doesn't overwrite it
COPY docker-entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r//' /entrypoint.sh && chmod +x /entrypoint.sh

# Start with entrypoint script
ENTRYPOINT ["/entrypoint.sh"]