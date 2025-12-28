FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for Pillow and other packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application code
COPY app.py .
COPY auth.py .
COPY rate_limiter.py .

# Copy .env file if it exists (for development)
COPY .env* ./

# Create a non-root user to run the application
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the application (without reload for production)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000","--reload"]