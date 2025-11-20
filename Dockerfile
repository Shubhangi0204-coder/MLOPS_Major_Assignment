
# Use a small official Python image
FROM python:3.9-slim

WORKDIR /app

# Install minimal system dependencies (if needed)
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model (train.py should have created savedmodel.pth on this branch)
COPY . .

# Expose port used by Flask
EXPOSE 5000

# Run the app with gunicorn for production-like behaviour
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--workers", "2"]
