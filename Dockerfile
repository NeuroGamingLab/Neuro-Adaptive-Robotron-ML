# Robotron:2026 with ML — Streamlit app
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project (DONOT-MODIFY-template excluded via .dockerignore)
COPY app.py .
COPY game/ game/
COPY README.md .
COPY DESIGN.md .
COPY ML-README.md .

# Optional: set image name for in-game Docker label (default shown if not set)
ENV DOCKER_IMAGE_NAME=robotron-ml

# Streamlit: port 8501, listen on all interfaces for Docker
EXPOSE 8501

# Run the app (headless = no browser in container)
CMD ["streamlit", "run", "app.py", \
  "--server.port=8501", \
  "--server.address=0.0.0.0", \
  "--server.headless=true", \
  "--browser.gatherUsageStats=false"]
