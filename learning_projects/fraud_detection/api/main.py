from fastapi import FastAPI, HTTPException
import pika
import json
import os
from pydantic import BaseModel
import logging

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# RabbitMQ Connection
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
connection = None
channel = None

def get_rabbitmq_channel():
    global connection, channel
    if connection is None or connection.is_closed:
        try:
            params = pika.ConnectionParameters(host=RABBITMQ_HOST)
            connection = pika.BlockingConnection(params)
            channel = connection.channel()
            channel.queue_declare(queue='transactions')
            logger.info("Connected to RabbitMQ")
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            return None
    return channel

class Transaction(BaseModel):
    id: str
    amount: float
    time: float
    v1: float
    v2: float
    v3: float
    v4: float
    # We simulate PCA vectors (v1-v4) similar to the famous CreditCard Fraud dataset

@app.post("/transaction")
async def send_transaction(tx: Transaction):
    ch = get_rabbitmq_channel()
    if ch is None:
        raise HTTPException(status_code=500, detail="Messaging service unavailable")
    
    message = json.dumps(tx.dict())
    ch.basic_publish(exchange='', routing_key='transactions', body=message)
    logger.info(f"Buffered transaction: {tx.id}")
    return {"status": "queued", "id": tx.id}
