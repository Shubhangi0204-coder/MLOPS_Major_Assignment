# Dockerfile - updated for Debian trixie / python:3.10-slim
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app


RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev \
    zlib1g-dev \
 && rm -rf /var/lib/apt/lists/*


COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy project files
COPY . /app

RUN if [ ! -f artifacts/savedmodel.pth ]; then echo "No model found in build context: running train.py"; python train.py || true; fi

EXPOSE 8080
CMD ["python", "app.py"]

