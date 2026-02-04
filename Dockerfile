# ---- Base image ----
FROM python:3.10-slim

# ---- Environment ----
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ---- System dependencies ----
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    cmake \
    poppler-utils \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ---- Working directory ----
WORKDIR /app

# ---- Copy requirements first (cache optimization) ----
COPY requirements.txt .

# ---- Install Python deps ----
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

# ---- Copy app code ----
COPY . .

# ---- Expose port ----
EXPOSE 10000

# ---- Start app ----
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000"]
