# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/

# Create data directories
RUN mkdir -p data/raw data/crops/board data/crops/hands data/crops/cells

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose port for future FastAPI service
EXPOSE 8000

# Default command (can be overridden)
CMD ["python", "-c", "print('OthelloniAI container ready. No service running yet.')"]
