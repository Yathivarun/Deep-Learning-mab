import socket
import numpy as np
import json

def test_send_embedding():
    # Create a dummy embedding like your real ones
    dummy_embedding = np.random.randn(1, 512).astype(np.float32)
    
    # Convert to list for JSON
    embedding_list = dummy_embedding.flatten().tolist()
    
    # Create test message
    test_data = {
        "action": "test",
        "embedding": embedding_list,
        "timestamp": "2024-01-15T10:30:00",
        "test_id": "pi_to_laptop_test"
    }
    
    # Laptop IP (CHANGE THIS)
    LAPTOP_IP = "192.168.1.50"  # Your laptop's Ethernet IP
    PORT = 5000
    
    try:
        # Send via socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((LAPTOP_IP, PORT))
            s.sendall(json.dumps(test_data).encode('utf-8'))
            response = s.recv(1024).decode('utf-8')
            
        print(f"✓ Sent embedding to {LAPTOP_IP}:{PORT}")
        print(f"  Response: {response}")
        return True
    except Exception as e:
        print(f"✗ Failed to send: {e}")
        return False

if __name__ == "__main__":
    print("Testing Pi → Laptop connection...")
    test_send_embedding()
