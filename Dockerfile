# Base image with TensorFlow and GPU support
FROM tensorflow/tensorflow:latest-gpu

# Set environment variables to prevent python from writing pyc files
# and to keep stdout/stderr unbuffered
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install required system packages for potential gym dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create application directories
WORKDIR /app
RUN mkdir -p /app/src && mkdir -p /app/weights

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code to container
COPY src /app/src

# Ensure Python finds the gym_gmmcar module inside src
ENV PYTHONPATH=/app/src

# Set working directory to where the execution scripts are located
WORKDIR /app/src

# Default action is to run standard training
CMD ["python", "improved_ddpg.py"]
