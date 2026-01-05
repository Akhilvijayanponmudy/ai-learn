FROM python:3.9-slim

WORKDIR /app

# Install system libs for psycopg2 if needed, though binary version usually works
# RUN apt-get update && apt-get install -y libpq-dev gcc

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-u", "worker.py"]
