#!/usr/bin/env python3
"""
Pi Display Module
Displays images on Pi's HDMI output or test window.
Matches the laptop's display.html behavior (4-second rotation, crossfade).

Usage:
    Standalone Test:
        python3 pi_display.py
    
    Integrated with sensor:
        Called by face_snapshot_networked.py
"""

import cv2
import numpy as np
import threading
import time
from pathlib import Path
from typing import List, Optional
import random

class PiDisplay:
    def __init__(self, fullscreen: bool = False, stock_dir: str = "stock_images"):
        """
        Initialize Pi Display.
        
        Args:
            fullscreen: If True, runs fullscreen on HDMI. If False, windowed for testing.
            stock_dir: Directory containing stock slideshow images
        """
        self.fullscreen = fullscreen
        self.stock_dir = Path(stock_dir)
        
        # Display state
        self.current_mode = "slideshow"  # "slideshow" | "poi"
        self.current_images = []  # List of image data (numpy arrays)
        self.current_index = 0
        
        # Stock images
        self.stock_images = []
        self.stock_index = 0
        
        # Display window
        self.window_name = "Museum Display - Pi"
        self.running = False
        self.display_thread = None
        
        # Crossfade state
        self.current_img = None
        self.next_img = None
        self.alpha = 0.0  # Crossfade alpha (0.0 = current, 1.0 = next)
        
        # Timing
        self.rotation_interval = 4.0  # seconds (match display.html)
        self.crossfade_duration = 1.5  # seconds (match display.html transition)
        self.last_switch_time = time.time()
        
        # Thread lock
        self.lock = threading.Lock()
        
        # Load stock images
        self.load_stock_images()
    
    def load_stock_images(self):
        """Loads stock images from directory."""
        self.stock_images = []
        
        if not self.stock_dir.exists():
            print(f"[DISPLAY] Warning: Stock directory not found: {self.stock_dir}")
            print(f"[DISPLAY] Creating directory and using placeholder...")
            self.stock_dir.mkdir(parents=True, exist_ok=True)
            
            # Create placeholder image
            placeholder = np.zeros((1080, 1920, 3), dtype=np.uint8)
            cv2.putText(placeholder, "No Stock Images Found", (600, 500),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            cv2.putText(placeholder, f"Add images to: {self.stock_dir}", (500, 600),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 150, 150), 2)
            self.stock_images = [placeholder]
            return
        
        # Load all images
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            for img_path in self.stock_dir.glob(ext):
                try:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        self.stock_images.append(img)
                except Exception as e:
                    print(f"[DISPLAY] Error loading {img_path.name}: {e}")
        
        if not self.stock_images:
            # Create placeholder if no images loaded
            placeholder = np.zeros((1080, 1920, 3), dtype=np.uint8)
            cv2.putText(placeholder, "No Valid Images Found", (600, 500),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            self.stock_images = [placeholder]
        else:
            # Shuffle for variety
            random.shuffle(self.stock_images)
        
        print(f"[DISPLAY] Loaded {len(self.stock_images)} stock images")
    
    def start(self):
        """Starts the display window in a separate thread."""
        if self.running:
            print("[DISPLAY] Already running")
            return
        
        self.running = True
        self.display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self.display_thread.start()
        
        print(f"[DISPLAY] Started ({'fullscreen' if self.fullscreen else 'windowed'})")
    
    def stop(self):
        """Stops the display window."""
        self.running = False
        if self.display_thread:
            self.display_thread.join(timeout=2.0)
        
        cv2.destroyAllWindows()
        print("[DISPLAY] Stopped")
    
    def show_poi_images(self, image_data_list: List[bytes]):
        """
        Switches to POI mode and displays received images.
        
        Args:
            image_data_list: List of JPEG image data (bytes)
        """
        with self.lock:
            self.current_mode = "poi"
            self.current_images = []
            
            # Decode image data
            for i, img_bytes in enumerate(image_data_list):
                try:
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img is not None:
                        self.current_images.append(img)
                    else:
                        print(f"[DISPLAY] Failed to decode image {i}")
                except Exception as e:
                    print(f"[DISPLAY] Error decoding image {i}: {e}")
            
            self.current_index = 0
            self.last_switch_time = time.time()
            
        print(f"[DISPLAY] POI mode activated ({len(self.current_images)} images)")
    
    def return_to_slideshow(self):
        """Returns to slideshow mode."""
        with self.lock:
            self.current_mode = "slideshow"
            self.current_images = []
            self.current_index = 0
            self.last_switch_time = time.time()
        
        print("[DISPLAY] Returned to slideshow mode")
    
    def _get_current_image(self) -> np.ndarray:
        """Returns the current image to display based on mode."""
        with self.lock:
            if self.current_mode == "poi" and self.current_images:
                return self.current_images[self.current_index].copy()
            else:
                # Slideshow mode
                return self.stock_images[self.stock_index].copy()
    
    def _get_next_image(self) -> np.ndarray:
        """Returns the next image in sequence."""
        with self.lock:
            if self.current_mode == "poi" and self.current_images:
                next_idx = (self.current_index + 1) % len(self.current_images)
                return self.current_images[next_idx].copy()
            else:
                # Slideshow mode
                next_idx = (self.stock_index + 1) % len(self.stock_images)
                return self.stock_images[next_idx].copy()
    
    def _advance_index(self):
        """Advances to the next image."""
        with self.lock:
            if self.current_mode == "poi" and self.current_images:
                self.current_index = (self.current_index + 1) % len(self.current_images)
            else:
                self.stock_index = (self.stock_index + 1) % len(self.stock_images)
    
    def _resize_to_fit(self, img: np.ndarray, target_size: tuple = (1920, 1080)) -> np.ndarray:
        """
        Resizes image to fit within target size while maintaining aspect ratio.
        Adds black bars if needed (letterboxing/pillarboxing).
        """
        target_w, target_h = target_size
        h, w = img.shape[:2]
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create black canvas
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # Center the image
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas
    
    def _crossfade(self, img1: np.ndarray, img2: np.ndarray, alpha: float) -> np.ndarray:
        """
        Crossfades between two images.
        
        Args:
            img1: Current image
            img2: Next image
            alpha: Blend factor (0.0 = img1, 1.0 = img2)
        
        Returns:
            Blended image
        """
        # Ensure both images are the same size
        if img1.shape != img2.shape:
            target_size = (1920, 1080)
            img1 = self._resize_to_fit(img1, target_size)
            img2 = self._resize_to_fit(img2, target_size)
        
        # Blend
        blended = cv2.addWeighted(img1, 1.0 - alpha, img2, alpha, 0)
        return blended
    
    def _display_loop(self):
        """Main display loop (runs in separate thread)."""
        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        if self.fullscreen:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        # Initialize with first image
        self.current_img = self._get_current_image()
        self.next_img = self._get_next_image()
        
        while self.running:
            current_time = time.time()
            elapsed = current_time - self.last_switch_time
            
            # Check if it's time to start crossfade
            if elapsed >= self.rotation_interval:
                # Start crossfade
                crossfade_elapsed = elapsed - self.rotation_interval
                
                if crossfade_elapsed < self.crossfade_duration:
                    # During crossfade
                    self.alpha = crossfade_elapsed / self.crossfade_duration
                    display_img = self._crossfade(self.current_img, self.next_img, self.alpha)
                else:
                    # Crossfade complete, switch images
                    self._advance_index()
                    self.current_img = self.next_img
                    self.next_img = self._get_next_image()
                    self.alpha = 0.0
                    self.last_switch_time = current_time
                    display_img = self.current_img
            else:
                # Show current image
                display_img = self.current_img
            
            # Resize to fit window
            display_img = self._resize_to_fit(display_img)
            
            # Show image
            cv2.imshow(self.window_name, display_img)
            
            # Handle key press (ESC to exit in test mode)
            key = cv2.waitKey(33)  # ~30 FPS
            if key == 27:  # ESC
                print("[DISPLAY] ESC pressed, exiting...")
                self.running = False
                break
        
        cv2.destroyAllWindows()


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PI DISPLAY TEST")
    print("=" * 60)
    print("Press ESC to exit")
    print()
    
    # Create display instance
    display = PiDisplay(fullscreen=False, stock_dir="stock_images")
    
    # Start display
    display.start()
    
    # Test: Switch to POI mode after 10 seconds
    print("Showing slideshow for 10 seconds...")
    time.sleep(10)
    
    print("\nSwitching to test POI images...")
    
    # Create test images
    test_images = []
    for i in range(3):
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        # Different colors for each image
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Blue, Green, Red
        img[:] = colors[i]
        
        # Add text
        cv2.putText(img, f"TEST IMAGE {i+1}", (600, 500),
                   cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)
        cv2.putText(img, "POI Mode Active", (700, 650),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
        # Encode to JPEG bytes (simulating network transmission)
        _, buffer = cv2.imencode('.jpg', img)
        test_images.append(buffer.tobytes())
    
    # Show test images
    display.show_poi_images(test_images)
    
    print("Showing test POI images for 20 seconds...")
    time.sleep(20)
    
    print("\nReturning to slideshow...")
    display.return_to_slideshow()
    
    print("Showing slideshow for 10 more seconds...")
    time.sleep(10)
    
    print("\nStopping display...")
    display.stop()
    
    print("✓ Test complete!")
