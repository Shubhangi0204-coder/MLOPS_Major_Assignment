
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app


RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app


RUN python -c "import os,sys; \
if not os.path.exists('artifacts/savedmodel.pth'): \
    print('Model not found in build context — training now...'); \
    sys.exit(0)" \
  && python train.py || true

EXPOSE 8080


CMD ["python", "app.py"]
