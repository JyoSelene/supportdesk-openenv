# SupportDesk-OpenEnv — Docker image
# Compatible with Hugging Face Spaces (SDK: docker)
# Build: docker build -t supportdesk-openenv .
# Run:   docker run -p 7860:7860 supportdesk-openenv

FROM python:3.11-slim

# HF Spaces runs as user 1000
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY email_triage_env/ ./email_triage_env/
COPY app.py .
COPY openenv.yaml .
COPY inference.py .

# Set ownership
RUN chown -R appuser:appuser /app
USER appuser

# Expose port used by HF Spaces
EXPOSE 7860

# Environment defaults (override at runtime)
ENV PORT=7860
ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
