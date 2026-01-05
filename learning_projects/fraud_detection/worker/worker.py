import pika
import json
import os
import torch
import psycopg2
import time
import logging
import numpy as np
from model import Autoencoder

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "fraud_db")
DB_USER = os.getenv("DB_USER", "user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")

# --- Database ---
def get_db_connection():
    while True:
        try:
            conn = psycopg2.connect(
                host=DB_HOST,
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD
            )
            return conn
        except psycopg2.OperationalError:
            logger.warning("Database not ready, retrying in 2s...")
            time.sleep(2)

def init_db(conn):
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id VARCHAR(50) PRIMARY KEY,
            amount FLOAT,
            fraud_score FLOAT,
            is_anomaly BOOLEAN,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    cur.close()

# --- Model ---
def load_model():
    model = Autoencoder(input_dim=5)
    # Ideally load_state_dict here from a file
    model.eval()
    return model

def calculate_anomaly_score(model, data):
    # Data expected as list/array of 5 floats: [amount, v1, v2, v3, v4]
    # Simple normalization simulation
    tensor_data = torch.FloatTensor(data)
    
    with torch.no_grad():
        reconstruction = model(tensor_data)
        loss = torch.mean((tensor_data - reconstruction) ** 2)
    
    return loss.item()

# --- Worker ---
def callback(ch, method, properties, body, model, db_conn):
    tx = json.loads(body)
    data_point = [
        tx['amount'] / 1000.0, # Normalize roughly
        tx['v1'], tx['v2'], tx['v3'], tx['v4']
    ]
    
    score = calculate_anomaly_score(model, data_point)
    is_anomaly = score > 0.1 # Threshold
    
    # Save to DB
    cur = db_conn.cursor()
    cur.execute(
        "INSERT INTO transactions (id, amount, fraud_score, is_anomaly) VALUES (%s, %s, %s, %s)",
        (tx['id'], tx['amount'], score, is_anomaly)
    )
    db_conn.commit()
    cur.close()
    
    logger.info(f"Processed {tx['id']}: Score={score:.4f} Anomaly={is_anomaly}")

def main():
    # 1. Setup DB
    conn = get_db_connection()
    init_db(conn)
    logger.info("Database initialized")

    # 2. Load Brain
    model = load_model()
    logger.info("AI Model loaded")

    # 3. Connect Queue
    connection = None
    while connection is None:
        try:
            params = pika.ConnectionParameters(host=RABBITMQ_HOST)
            connection = pika.BlockingConnection(params)
        except Exception as e:
             logger.warning(f"RabbitMQ not ready: {e}, retrying...")
             time.sleep(2)

    channel = connection.channel()
    channel.queue_declare(queue='transactions')

    # 4. Start Consuming
    channel.basic_consume(
        queue='transactions',
        on_message_callback=lambda ch, method, properties, body: callback(ch, method, properties, body, model, conn),
        auto_ack=True
    )

    logger.info("Worker started, waiting for messages...")
    channel.start_consuming()

if __name__ == "__main__":
    main()
