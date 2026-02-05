#!/usr/bin/env python3
"""
Phase 1 Network Test Script
Tests the TCP communication between Pi and Laptop

Usage:
------
On Laptop:
  python3 test_network.py laptop

On Pi:
  python3 test_network.py pi

This will test:
1. Connection establishment
2. Heartbeat system
3. Embedding transmission
4. Image transmission
5. Disconnection handling
"""

import time
import numpy as np
from network_protocol import PiClient, LaptopServer

# ============================================================================
# LAPTOP TEST
# ============================================================================

def test_laptop():
    """
    Run this on the LAPTOP.
    Starts server, waits for Pi, receives embeddings, sends test images.
    """
    print("=" * 60)
    print("LAPTOP SERVER TEST")
    print("=" * 60)
    print("Waiting for Pi to connect...")
    print()
    
    server = LaptopServer()
    embedding_count = 0
    
    # Callback: When Pi connects
    def on_connected(addr):
        print(f"✓ Pi connected from {addr}")
        print("  Waiting for embeddings...\n")
    
    # Callback: When Pi disconnects
    def on_disconnected():
        print("\n✗ Pi disconnected")
        print(f"  Total embeddings received: {embedding_count}")
    
    # Callback: When embedding received
    def on_embedding(embedding, timestamp):
        nonlocal embedding_count
        embedding_count += 1
        
        print(f"[{embedding_count}] Received embedding at {timestamp}")
        print(f"    Shape: {embedding.shape}")
        print(f"    Sample values: {embedding[:5]}")
        
        # Simulate matching logic
        # 50% chance of "hit" for testing
        is_hit = (embedding_count % 2 == 0)
        
        if is_hit:
            print(f"    → MATCH! Sending result + images...")
            
            # Send match result
            server.send_match_result(
                hit=True,
                person_id="1001",
                score=0.85
            )
            
            # Send test images (fake JPEG data)
            test_images = [
                b'\xff\xd8\xff\xe0' + b'FAKE_IMAGE_1' * 100,  # Fake JPEG header
                b'\xff\xd8\xff\xe0' + b'FAKE_IMAGE_2' * 100,
                b'\xff\xd8\xff\xe0' + b'FAKE_IMAGE_3' * 100,
            ]
            server.send_images(test_images)
            
        else:
            print(f"    → No match, sending negative result")
            server.send_match_result(hit=False, score=0.35)
        
        print()
    
    # Set callbacks
    server.on_client_connected = on_connected
    server.on_client_disconnected = on_disconnected
    server.on_embedding_received = on_embedding
    
    # Start server
    if not server.start():
        print("Failed to start server!")
        return
    
    print("Server running. Press Ctrl+C to stop.\n")
    
    try:
        while True:
            time.sleep(1)
            
            # Print status every 10 seconds
            if int(time.time()) % 10 == 0:
                status = "CONNECTED" if server.connected else "WAITING"
                print(f"[Status] {status} | Embeddings received: {embedding_count}")
    
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
        server.stop()
        print("✓ Server stopped")

# ============================================================================
# PI TEST
# ============================================================================

def test_pi():
    """
    Run this on the RASPBERRY PI.
    Connects to laptop, sends test embeddings, receives images.
    """
    print("=" * 60)
    print("PI CLIENT TEST")
    print("=" * 60)
    print("Connecting to laptop...")
    print()
    
    client = PiClient()
    images_received = 0
    
    # Callback: When match result received
    def on_match(msg):
        hit = msg.get("hit")
        person_id = msg.get("person_id")
        score = msg.get("score")
        
        if hit:
            print(f"✓ MATCH! Person #{person_id}, Score: {score:.2f}")
        else:
            print(f"✗ No match (score: {score:.2f})")
    
    # Callback: When images received
    def on_images(image_list):
        nonlocal images_received
        
        if not image_list:
            print("← Laptop says: Return to slideshow")
        else:
            images_received += len(image_list)
            print(f"← Received {len(image_list)} images (total: {images_received})")
            for i, img_data in enumerate(image_list, 1):
                print(f"   Image {i}: {len(img_data)} bytes")
    
    # Callback: When disconnected
    def on_disconnect():
        print("\n✗ Disconnected from laptop!")
        print("  Attempting to reconnect in 5 seconds...")
        time.sleep(5)
        # In real code, you'd implement auto-reconnect here
    
    # Set callbacks
    client.on_match_result = on_match
    client.on_images_received = on_images
    client.on_disconnected = on_disconnect
    
    # Connect to laptop
    if not client.connect(timeout=10):
        print("✗ Failed to connect to laptop!")
        print("  Make sure:")
        print("    1. Laptop is running test_network.py laptop")
        print("    2. Ethernet cable is connected")
        print("    3. IPs are correct (Pi: 192.168.137.198, Laptop: 192.168.137.1)")
        print("    4. Firewall allows port 5000")
        return
    
    print("✓ Connected to laptop!")
    print("  Sending test embeddings every 4 seconds...\n")
    
    embedding_count = 0
    
    try:
        while True:
            time.sleep(4)  # 4 second cycle (as per requirements)
            
            # Generate random test embedding (512 dimensions, like ArcFace)
            test_embedding = np.random.rand(512).astype(np.float32)
            embedding_count += 1
            
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{embedding_count}] Sending embedding at {timestamp}...")
            
            if client.send_embedding(test_embedding, timestamp):
                print(f"    ✓ Sent (shape: {test_embedding.shape})")
            else:
                print(f"    ✗ Failed to send (disconnected?)")
                break
            
            print()
    
    except KeyboardInterrupt:
        print("\n\nDisconnecting...")
        client.disconnect()
        print("✓ Disconnected")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys
    
    print("\n")
    
    if len(sys.argv) < 2:
        print("ERROR: Missing argument!")
        print()
        print("Usage:")
        print("  On Laptop: python3 test_network.py laptop")
        print("  On Pi:     python3 test_network.py pi")
        print()
        sys.exit(1)
    
    mode = sys.argv[1].lower()
    
    if mode == "laptop":
        test_laptop()
    elif mode == "pi":
        test_pi()
    else:
        print(f"ERROR: Unknown mode '{mode}'")
        print("Use 'laptop' or 'pi'")
        sys.exit(1)
