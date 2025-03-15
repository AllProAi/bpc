FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ /app/src/
COPY models/ /app/models/
COPY config.yaml /app/

# Create entry point script
COPY docker/entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

# Make port available for API
EXPOSE 8080

# Set environment variables
ENV PYTHONPATH="/app:${PYTHONPATH}"
ENV CUDA_VISIBLE_DEVICES="0"

# Run the entry point script
ENTRYPOINT ["/app/entrypoint.sh"] 