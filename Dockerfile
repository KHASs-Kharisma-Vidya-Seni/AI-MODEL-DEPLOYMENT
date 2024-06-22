# Gunakan image Python yang diinginkan sebagai base image
FROM python:3.8

# Set working directory di dalam container
WORKDIR /app

# Copy requirements.txt ke dalam container
COPY requirements.txt .

# Install CMake (diperlukan oleh dlib) dan OpenGL (diperlukan oleh OpenCV)
RUN apt-get update \
    && apt-get install -y cmake \
                          libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies termasuk dlib dan OpenCV
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh kode aplikasi ke dalam container
COPY . .

# Copy file .env ke dalam container (jika diperlukan)
COPY .env ./

# Eksekusi perintah saat container dijalankan
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_ENV=development

# Expose the Flask port
EXPOSE 5000

# CMD untuk menjalankan aplikasi Flask
CMD ["flask", "run"]
