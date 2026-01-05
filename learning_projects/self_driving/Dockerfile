FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for OpenAI Gym (rendering, etc)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install RL libraries
RUN pip install --no-cache-dir \
    torch \
    stable-baselines3[extra] \
    gymnasium[classic-control] \
    numpy

COPY . .

# Default command matches "train.py"
CMD ["python", "train.py"]
