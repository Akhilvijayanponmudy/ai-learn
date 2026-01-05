import requests
import random
import time
import uuid
import sys

API_URL = "http://localhost:8000/transaction"

def generate_transaction():
    # Simulate normal vs fraud
    is_fraud = random.random() < 0.05 # 5% fraud rate
    
    if is_fraud:
        # Anomalous pattern: High amount or weird vectors
        amount = random.uniform(5000, 20000)
        v1 = random.uniform(-10, -5)
        v2 = random.uniform(5, 10)
    else:
        # Normal pattern
        amount = random.uniform(10, 500)
        v1 = random.uniform(-2, 2)
        v2 = random.uniform(-2, 2)
        
    return {
        "id": str(uuid.uuid4()),
        "amount": amount,
        "time": time.time(),
        "v1": v1,
        "v2": v2,
        "v3": random.uniform(-1, 1), # Noise
        "v4": random.uniform(-1, 1)  # Noise
    }

def main():
    print(f"🚀 Starting Load Generator -> {API_URL}")
    print("Press Ctrl+C to stop")
    
    count = 0
    try:
        while True:
            tx = generate_transaction()
            try:
                resp = requests.post(API_URL, json=tx)
                if resp.status_code == 200:
                    print(f"Sent: {tx['id']} | Amount: ${tx['amount']:.2f}", end="\r")
                    count += 1
                else:
                    print(f"\n❌ Error: {resp.status_code}")
            except Exception as e:
                print(f"\n❌ Connection failed: {e}")
                time.sleep(1)
            
            time.sleep(0.1) # 10 TPS
            
    except KeyboardInterrupt:
        print(f"\nStopped. Sent {count} transactions.")

if __name__ == "__main__":
    main()
