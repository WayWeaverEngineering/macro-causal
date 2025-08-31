# Use Python 3.10 slim image for optimal compatibility and size
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Environment variables for pip optimization
ENV PIP_ONLY_BINARY=:all:
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONUNBUFFERED=1

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies with wheel optimization
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application code
COPY main.py .
COPY collectors/ ./collectors/

# Expose port for potential health checks
EXPOSE 8080

# Set the default command
CMD ["python", "main.py"]
