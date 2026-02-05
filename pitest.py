#!/usr/bin/env python3
"""
Raspberry Pi Connectivity Test
Run this on: 192.168.137.198
Tests connection to: Laptop at 192.168.137.1
"""

import socket
import subprocess
import platform

LAPTOP_IP = "192.168.137.1"
PI_IP = "192.168.137.198"
TEST_PORT = 5000

def ping_test():
    """Test basic ping connectivity"""
    print(f"\n[1] Pinging Laptop at {LAPTOP_IP}...")
    param = '-n' if platform.system().lower() == 'windows' else '-c'
    command = ['ping', param, '4', LAPTOP_IP]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✓ Ping successful!")
            return True
        else:
            print("✗ Ping failed!")
            return False
    except Exception as e:
        print(f"✗ Ping error: {e}")
        return False

def tcp_client():
    """Connect to laptop as client"""
    print(f"\n[2] Connecting to Laptop at {LAPTOP_IP}:{TEST_PORT}...")
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        sock.connect((LAPTOP_IP, TEST_PORT))
        
        sock.send(b"Hello from Raspberry Pi!")
        data = sock.recv(1024).decode()
        print(f"✓ Connected! Received: {data}")
        
        sock.close()
        return True
        
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False

def tcp_listener():
    """Start TCP listener to receive connection from Laptop"""
    print(f"\n[2] Starting TCP listener on {PI_IP}:{TEST_PORT}...")
    print("    Waiting for connection from Laptop (run laptop_test.py in connect mode)...")
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('', TEST_PORT))  # Bind to all interfaces
        sock.listen(1)
        sock.settimeout(30)
        
        conn, addr = sock.accept()
        print(f"✓ Connection received from {addr[0]}:{addr[1]}")
        
        data = conn.recv(1024).decode()
        print(f"✓ Received message: {data}")
        
        conn.send(b"Hello from Raspberry Pi!")
        conn.close()
        sock.close()
        return True
        
    except socket.timeout:
        print("✗ Timeout - no connection received")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    print("="*50)
    print("RASPBERRY PI CONNECTIVITY TEST")
    print("="*50)
    
    # Test 1: Ping
    ping_test()
    
    # Test 2: Network communication
    print("\n" + "="*50)
    print("Choose test mode:")
    print("1 - Connect to Laptop (run laptop_test.py in listen mode first)")
    print("2 - Listen for connection from Laptop (run this first, then laptop_test.py)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        tcp_client()
    elif choice == "2":
        tcp_listener()
    else:
        print("Invalid choice")
    
    print("\n" + "="*50)
    print("Test complete!")
    print("="*50)
