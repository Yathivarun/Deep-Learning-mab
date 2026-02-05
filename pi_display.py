"""
Pi Display Module
Shows images received from laptop on HDMI output or window

Can run in two modes:
1. Test mode: OpenCV window (for testing)
2. Production mode: Full-screen HDMI display
"""

import cv2
import numpy as np
import time
import threading
from pathlib import Path

class PiDisplay:
    def __init__(self, fullscreen=False, slideshow_dir=None):
        """
        Initialize display
        
        Args:
            fullscreen: If True, use full HDMI display. If False, use window.
            slideshow_dir: Directory with stock images for idle mode
        """
        self.fullscreen = fullscreen
        self.slideshow_dir = Path(slideshow_dir) if slideshow_dir else None
        self.window_name = "Pi Display"
        
        # State
        self.mode = "idle"  # "idle" or "person"
        self.current_images = []  # List of image data (bytes)
        self.current_index = 0
        self.slideshow_images = []
        
        # Threading
        self.running = False
        self.display_thread = None
        self.lock = threading.Lock()
        
        # Timing
        self.image_duration = 5.0  # seconds per image
        self.last_change_time = 0
        
        # Load slideshow images
        if self.slideshow_dir and self.slideshow_dir.exists():
            self._load_slideshow_images()
    
    def _load_slideshow_images(self):
        """Load stock images for slideshow mode"""
        print(f"[DISPLAY] Loading slideshow images from {self.slideshow_dir}")
        
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            for img_path in self.slideshow_dir.glob(ext):
                try:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        self.slideshow_images.append(img)
                except:
                    pass
        
        print(f"[DISPLAY] Loaded {len(self.slideshow_images)} slideshow images")
    
    def start(self):
        """Start display thread"""
        if self.running:
            return
        
        self.running = True
        self.display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self.display_thread.start()
        
        print("[DISPLAY] Display started")
    
    def stop(self):
        """Stop display"""
        self.running = False
        if self.display_thread:
            self.display_thread.join(timeout=2)
        cv2.destroyAllWindows()
        print("[DISPLAY] Display stopped")
    
    def show_person_images(self, image_data_list):
        """
        Display specific person's images
        
        Args:
            image_data_list: List of bytes (JPEG data from laptop)
        """
        with self.lock:
            self.mode = "person"
            self.current_images = []
            
            # Decode images
            for img_data in image_data_list:
                try:
                    nparr = np.frombuffer(img_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img is not None:
                        self.current_images.append(img)
                except Exception as e:
                    print(f"[DISPLAY] Failed to decode image: {e}")
            
            self.current_index = 0
            self.last_change_time = time.time()
            
            print(f"[DISPLAY] Showing {len(self.current_images)} person images")
    
    def return_to_slideshow(self):
        """Return to idle slideshow mode"""
        with self.lock:
            self.mode = "idle"
            self.current_images = []
            self.current_index = 0
            self.last_change_time = time.time()
            print("[DISPLAY] Returned to slideshow mode")
    
    def _display_loop(self):
        """Main display loop running in background thread"""
        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        if self.fullscreen:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        while self.running:
            current_time = time.time()
            
            # Check if it's time to change image
            if current_time - self.last_change_time >= self.image_duration:
                self.last_change_time = current_time
                
                with self.lock:
                    if self.mode == "person" and self.current_images:
                        # Cycle through person images
                        self.current_index = (self.current_index + 1) % len(self.current_images)
                    elif self.mode == "idle" and self.slideshow_images:
                        # Cycle through slideshow
                        self.current_index = (self.current_index + 1) % len(self.slideshow_images)
            
            # Get current image to display
            with self.lock:
                if self.mode == "person" and self.current_images:
                    img = self.current_images[self.current_index].copy()
                    status_text = f"Person Image {self.current_index + 1}/{len(self.current_images)}"
                    status_color = (0, 255, 0)
                
                elif self.mode == "idle" and self.slideshow_images:
                    img = self.slideshow_images[self.current_index].copy()
                    status_text = f"Slideshow {self.current_index + 1}/{len(self.slideshow_images)}"
                    status_color = (255, 255, 0)
                
                else:
                    # No images - show black screen with message
                    img = np.zeros((720, 1280, 3), dtype=np.uint8)
                    status_text = "Waiting for images..."
                    status_color = (128, 128, 128)
            
            # Add status text (only in window mode, not fullscreen)
            if not self.fullscreen:
                cv2.putText(img, status_text, (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
                
                mode_text = f"Mode: {self.mode.upper()}"
                cv2.putText(img, mode_text, (20, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Show image
            cv2.imshow(self.window_name, img)
            
            # Handle key press
            key = cv2.waitKey(100)
            if key == 27:  # ESC
                self.running = False
                break
            elif key == ord('f'):  # Toggle fullscreen
                self.fullscreen = not self.fullscreen
                if self.fullscreen:
                    cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

# ============================================================================
# STANDALONE TEST
# ============================================================================
if __name__ == "__main__":
    print("Pi Display Test")
    print("================")
    print("Controls:")
    print("  ESC - Exit")
    print("  f   - Toggle fullscreen")
    print()
    
    # Create test display
    display = PiDisplay(fullscreen=False)
    display.start()
    
    print("Display started in window mode")
    print("Simulating slideshow...")
    
    # Create some test images
    test_images = []
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    for i, color in enumerate(colors):
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        img[:] = color
        cv2.putText(img, f"Test Image {i+1}", (400, 360),
                   cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), 5)
        
        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', img)
        test_images.append(buffer.tobytes())
    
    # Show test images
    time.sleep(2)
    print("Showing test person images...")
    display.show_person_images(test_images)
    
    time.sleep(15)
    print("Returning to slideshow...")
    display.return_to_slideshow()
    
    # Keep running
    try:
        while display.running:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    
    display.stop()
    print("Test complete")
