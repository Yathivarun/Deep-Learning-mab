"""
Network Protocol for Pi <-> Laptop Communication
Uses TCP sockets for offline operation over Ethernet

Protocol Design:
================
1. All messages are JSON strings terminated with newline
2. Large data (images) sent as length-prefixed binary
3. Heartbeat every 5 seconds to detect disconnection

Message Types:
--------------
Pi -> Laptop:
  - {"type": "heartbeat", "timestamp": "..."}
  - {"type": "embedding", "data": [...], "timestamp": "..."}
  - {"type": "status", "status": "ready/processing/error"}

Laptop -> Pi:
  - {"type": "heartbeat_ack"}
  - {"type": "match_result", "hit": true/false, "person_id": "1001", "score": 0.85}
  - {"type": "images", "count": 3}  # Followed by binary image data
  - {"type": "slideshow"}  # Return to slideshow mode
"""

import socket
import json
import struct
import time
import threading
from typing import Optional, Callable, Dict, Any
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

LAPTOP_IP = "192.168.137.1"
PI_IP = "192.168.137.198"
COMMUNICATION_PORT = 5000
BUFFER_SIZE = 4096
HEARTBEAT_INTERVAL = 5.0  # seconds
CONNECTION_TIMEOUT = 15.0  # seconds

# ============================================================================
# MESSAGE ENCODING/DECODING
# ============================================================================

def encode_message(msg_dict: dict) -> bytes:
    """
    Encode a message dictionary to JSON bytes with newline terminator.
    """
    json_str = json.dumps(msg_dict)
    return (json_str + "\n").encode('utf-8')

def decode_message(data: bytes) -> Optional[dict]:
    """
    Decode JSON message from bytes.
    Returns None if invalid.
    """
    try:
        json_str = data.decode('utf-8').strip()
        return json.loads(json_str)
    except:
        return None

def encode_embedding(embedding: np.ndarray) -> list:
    """
    Convert numpy embedding to JSON-serializable list.
    """
    return embedding.flatten().tolist()

def decode_embedding(embedding_list: list) -> np.ndarray:
    """
    Convert list back to numpy array.
    """
    return np.array(embedding_list, dtype=np.float32)

def send_image_binary(sock: socket.socket, image_data: bytes):
    """
    Send image as length-prefixed binary data.
    Format: [4 bytes length][image data]
    """
    length = len(image_data)
    sock.sendall(struct.pack('!I', length))  # Network byte order, unsigned int
    sock.sendall(image_data)

def receive_image_binary(sock: socket.socket) -> Optional[bytes]:
    """
    Receive length-prefixed binary image data.
    Returns None on error.
    """
    try:
        # Read 4-byte length header
        length_data = sock.recv(4)
        if len(length_data) != 4:
            return None
        
        length = struct.unpack('!I', length_data)[0]
        
        # Read image data in chunks
        image_data = b''
        remaining = length
        
        while remaining > 0:
            chunk = sock.recv(min(remaining, BUFFER_SIZE))
            if not chunk:
                return None
            image_data += chunk
            remaining -= len(chunk)
        
        return image_data
    except:
        return None

# ============================================================================
# PI CLIENT (Sends embeddings, receives images)
# ============================================================================

class PiClient:
    """
    Runs on Raspberry Pi.
    Connects to laptop, sends embeddings, receives display commands.
    """
    
    def __init__(self, laptop_ip: str = LAPTOP_IP, port: int = COMMUNICATION_PORT):
        self.laptop_ip = laptop_ip
        self.port = port
        self.sock = None
        self.connected = False
        self.running = False
        
        # Callbacks
        self.on_match_result: Optional[Callable] = None
        self.on_images_received: Optional[Callable] = None
        self.on_disconnected: Optional[Callable] = None
        
        # Threading
        self.receive_thread = None
        self.heartbeat_thread = None
        self.last_heartbeat_sent = 0
        self.last_heartbeat_received = 0
    
    def connect(self, timeout: int = 10) -> bool:
        """
        Attempt to connect to laptop server.
        Returns True if successful.
        """
        print(f"[PI-CLIENT] Connecting to laptop at {self.laptop_ip}:{self.port}...")
        
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(timeout)
            self.sock.connect((self.laptop_ip, self.port))
            self.sock.settimeout(None)  # Set back to blocking
            
            self.connected = True
            self.running = True
            self.last_heartbeat_received = time.time()
            
            # Start receive thread
            self.receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
            self.receive_thread.start()
            
            # Start heartbeat thread
            self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self.heartbeat_thread.start()
            
            print("[PI-CLIENT] ✓ Connected to laptop!")
            return True
            
        except Exception as e:
            print(f"[PI-CLIENT] ✗ Connection failed: {e}")
            self.connected = False
            return False
    
    def send_embedding(self, embedding: np.ndarray, timestamp: str = None) -> bool:
        """
        Send face embedding to laptop for matching.
        """
        if not self.connected:
            return False
        
        if timestamp is None:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            message = {
                "type": "embedding",
                "data": encode_embedding(embedding),
                "timestamp": timestamp
            }
            
            self.sock.sendall(encode_message(message))
            print(f"[PI-CLIENT] Sent embedding at {timestamp}")
            return True
            
        except Exception as e:
            print(f"[PI-CLIENT] Failed to send embedding: {e}")
            self._handle_disconnection()
            return False
    
    def send_status(self, status: str) -> bool:
        """
        Send status update (ready/processing/error).
        """
        if not self.connected:
            return False
        
        try:
            message = {"type": "status", "status": status}
            self.sock.sendall(encode_message(message))
            return True
        except:
            return False
    
    def _heartbeat_loop(self):
        """
        Send heartbeat every 5 seconds.
        """
        while self.running:
            time.sleep(HEARTBEAT_INTERVAL)
            
            if not self.connected:
                break
            
            try:
                message = {"type": "heartbeat", "timestamp": time.time()}
                self.sock.sendall(encode_message(message))
                self.last_heartbeat_sent = time.time()
            except:
                self._handle_disconnection()
                break
    
    def _receive_loop(self):
        """
        Continuously listen for messages from laptop.
        """
        buffer = ""
        
        while self.running:
            try:
                data = self.sock.recv(BUFFER_SIZE)
                
                if not data:
                    # Connection closed
                    self._handle_disconnection()
                    break
                
                buffer += data.decode('utf-8')
                
                # Process complete messages (terminated by newline)
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    self._handle_message(line)
                    
            except Exception as e:
                print(f"[PI-CLIENT] Receive error: {e}")
                self._handle_disconnection()
                break
    
    def _handle_message(self, json_str: str):
        """
        Process received message from laptop.
        """
        try:
            msg = json.loads(json_str)
            msg_type = msg.get("type")
            
            if msg_type == "heartbeat_ack":
                self.last_heartbeat_received = time.time()
            
            elif msg_type == "match_result":
                # Laptop found a match (or didn't)
                if self.on_match_result:
                    self.on_match_result(msg)
            
            elif msg_type == "images":
                # Laptop is sending images
                count = msg.get("count", 0)
                images = self._receive_images(count)
                
                if self.on_images_received and images:
                    self.on_images_received(images)
            
            elif msg_type == "slideshow":
                # Return to slideshow mode
                if self.on_images_received:
                    self.on_images_received([])  # Empty list = slideshow
                    
        except Exception as e:
            print(f"[PI-CLIENT] Message handling error: {e}")
    
    def _receive_images(self, count: int) -> list:
        """
        Receive multiple binary images from laptop.
        """
        images = []
        
        for i in range(count):
            img_data = receive_image_binary(self.sock)
            if img_data:
                images.append(img_data)
                print(f"[PI-CLIENT] Received image {i+1}/{count} ({len(img_data)} bytes)")
            else:
                print(f"[PI-CLIENT] Failed to receive image {i+1}")
                break
        
        return images
    
    def _handle_disconnection(self):
        """
        Handle lost connection.
        """
        self.connected = False
        self.running = False
        
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
        
        print("[PI-CLIENT] ✗ Disconnected from laptop")
        
        if self.on_disconnected:
            self.on_disconnected()
    
    def disconnect(self):
        """
        Gracefully disconnect.
        """
        self.running = False
        self.connected = False
        
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
        
        print("[PI-CLIENT] Disconnected")

# ============================================================================
# LAPTOP SERVER (Receives embeddings, sends images)
# ============================================================================

class LaptopServer:
    """
    Runs on Laptop.
    Listens for Pi connection, receives embeddings, sends display commands.
    """
    
    def __init__(self, port: int = COMMUNICATION_PORT):
        self.port = port
        self.server_sock = None
        self.client_sock = None
        self.client_addr = None
        self.connected = False
        self.running = False
        
        # Callbacks
        self.on_embedding_received: Optional[Callable] = None
        self.on_status_received: Optional[Callable] = None
        self.on_client_connected: Optional[Callable] = None
        self.on_client_disconnected: Optional[Callable] = None
        
        # Threading
        self.accept_thread = None
        self.receive_thread = None
        self.heartbeat_thread = None
        self.last_heartbeat_received = 0
    
    def start(self, bind_ip: str = "0.0.0.0") -> bool:
        """
        Start server and listen for Pi connection.
        """
        print(f"[LAPTOP-SERVER] Starting server on {bind_ip}:{self.port}...")
        
        try:
            self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_sock.bind((bind_ip, self.port))
            self.server_sock.listen(1)
            
            self.running = True
            
            # Start accept thread
            self.accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
            self.accept_thread.start()
            
            print(f"[LAPTOP-SERVER] ✓ Server listening, waiting for Pi...")
            return True
            
        except Exception as e:
            print(f"[LAPTOP-SERVER] ✗ Failed to start: {e}")
            return False
    
    def _accept_loop(self):
        """
        Accept incoming Pi connection (only 1 client).
        """
        while self.running:
            try:
                self.server_sock.settimeout(1.0)
                client_sock, client_addr = self.server_sock.accept()
                
                # If already connected, reject
                if self.connected:
                    print(f"[LAPTOP-SERVER] Rejected connection from {client_addr} (already connected)")
                    client_sock.close()
                    continue
                
                self.client_sock = client_sock
                self.client_addr = client_addr
                self.connected = True
                self.last_heartbeat_received = time.time()
                
                print(f"[LAPTOP-SERVER] ✓ Pi connected from {client_addr}")
                
                if self.on_client_connected:
                    self.on_client_connected(client_addr)
                
                # Start receive thread for this client
                self.receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
                self.receive_thread.start()
                
                # Start heartbeat monitoring
                self.heartbeat_thread = threading.Thread(target=self._heartbeat_monitor, daemon=True)
                self.heartbeat_thread.start()
                
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"[LAPTOP-SERVER] Accept error: {e}")
    
    def _receive_loop(self):
        """
        Receive messages from Pi.
        """
        buffer = ""
        
        while self.running and self.connected:
            try:
                data = self.client_sock.recv(BUFFER_SIZE)
                
                if not data:
                    self._handle_client_disconnect()
                    break
                
                buffer += data.decode('utf-8')
                
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    self._handle_message(line)
                    
            except Exception as e:
                print(f"[LAPTOP-SERVER] Receive error: {e}")
                self._handle_client_disconnect()
                break
    
    def _handle_message(self, json_str: str):
        """
        Process message from Pi.
        """
        try:
            msg = json.loads(json_str)
            msg_type = msg.get("type")
            
            if msg_type == "heartbeat":
                self.last_heartbeat_received = time.time()
                # Send ACK
                self.send_message({"type": "heartbeat_ack"})
            
            elif msg_type == "embedding":
                # Pi sent a face embedding
                embedding_list = msg.get("data")
                timestamp = msg.get("timestamp")
                
                if self.on_embedding_received and embedding_list:
                    embedding = decode_embedding(embedding_list)
                    self.on_embedding_received(embedding, timestamp)
            
            elif msg_type == "status":
                status = msg.get("status")
                if self.on_status_received:
                    self.on_status_received(status)
                    
        except Exception as e:
            print(f"[LAPTOP-SERVER] Message handling error: {e}")
    
    def _heartbeat_monitor(self):
        """
        Monitor Pi heartbeat. Disconnect if timeout.
        """
        while self.running and self.connected:
            time.sleep(CONNECTION_TIMEOUT / 3)
            
            elapsed = time.time() - self.last_heartbeat_received
            
            if elapsed > CONNECTION_TIMEOUT:
                print("[LAPTOP-SERVER] ✗ Pi heartbeat timeout")
                self._handle_client_disconnect()
                break
    
    def send_match_result(self, hit: bool, person_id: str = None, score: float = 0.0) -> bool:
        """
        Send recognition result to Pi.
        """
        if not self.connected:
            return False
        
        try:
            message = {
                "type": "match_result",
                "hit": hit,
                "person_id": person_id,
                "score": score
            }
            return self.send_message(message)
        except:
            return False
    
    def send_images(self, image_data_list: list) -> bool:
        """
        Send multiple images to Pi for display.
        image_data_list: list of bytes (JPEG data)
        """
        if not self.connected or not image_data_list:
            return False
        
        try:
            # 1. Send count
            message = {"type": "images", "count": len(image_data_list)}
            self.send_message(message)
            
            # 2. Send each image as binary
            for img_data in image_data_list:
                send_image_binary(self.client_sock, img_data)
            
            print(f"[LAPTOP-SERVER] Sent {len(image_data_list)} images to Pi")
            return True
            
        except Exception as e:
            print(f"[LAPTOP-SERVER] Failed to send images: {e}")
            return False
    
    def send_slideshow_command(self) -> bool:
        """
        Tell Pi to return to slideshow mode.
        """
        return self.send_message({"type": "slideshow"})
    
    def send_message(self, msg_dict: dict) -> bool:
        """
        Send any JSON message to Pi.
        """
        if not self.connected:
            return False
        
        try:
            self.client_sock.sendall(encode_message(msg_dict))
            return True
        except:
            self._handle_client_disconnect()
            return False
    
    def _handle_client_disconnect(self):
        """
        Handle Pi disconnection.
        """
        self.connected = False
        
        if self.client_sock:
            try:
                self.client_sock.close()
            except:
                pass
        
        self.client_sock = None
        self.client_addr = None
        
        print("[LAPTOP-SERVER] ✗ Pi disconnected")
        
        if self.on_client_disconnected:
            self.on_client_disconnected()
    
    def stop(self):
        """
        Stop server.
        """
        self.running = False
        self.connected = False
        
        if self.client_sock:
            try:
                self.client_sock.close()
            except:
                pass
        
        if self.server_sock:
            try:
                self.server_sock.close()
            except:
                pass
        
        print("[LAPTOP-SERVER] Server stopped")

# ============================================================================
# SIMPLE TEST FUNCTIONS
# ============================================================================

def test_pi_client():
    """Test Pi client connecting to laptop."""
    client = PiClient()
    
    def on_match(msg):
        print(f"Match result: {msg}")
    
    def on_images(images):
        print(f"Received {len(images)} images")
    
    client.on_match_result = on_match
    client.on_images_received = on_images
    
    if client.connect():
        # Send test embedding
        test_embedding = np.random.rand(512).astype(np.float32)
        client.send_embedding(test_embedding)
        
        # Keep alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            client.disconnect()

def test_laptop_server():
    """Test laptop server waiting for Pi."""
    server = LaptopServer()
    
    def on_embedding(embedding, timestamp):
        print(f"Received embedding at {timestamp}, shape: {embedding.shape}")
        # Send fake match result
        server.send_match_result(hit=True, person_id="1001", score=0.95)
    
    server.on_embedding_received = on_embedding
    
    if server.start():
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            server.stop()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "pi":
        test_pi_client()
    elif len(sys.argv) > 1 and sys.argv[1] == "laptop":
        test_laptop_server()
    else:
        print("Usage:")
        print("  python network_protocol.py pi      # Test Pi client")
        print("  python network_protocol.py laptop  # Test Laptop server")
