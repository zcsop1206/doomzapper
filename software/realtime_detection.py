"""
Real-time Scroll Detection with Doomscrolling Intervention
Uses trained Random Forest model to detect scrolling and implements bucket algorithm
"""

import serial
import numpy as np
import joblib
import time
from collections import deque
from datetime import datetime, timedelta

class ScrollDetector:
    """Real-time scroll detection using trained Random Forest model"""
    
    def __init__(self, model_path='models/scroll_detector_rf.pkl'):
        """Load trained model"""
        self.model = joblib.load(model_path)
        print(f"✓ Model loaded from {model_path}")
        
        # Window buffer for real-time processing
        self.window_size = 40  # Must match training window size
        self.buffer = deque(maxlen=self.window_size)
        
    def add_sample(self, value):
        """Add new sample to buffer"""
        self.buffer.append(value)
    
    def is_ready(self):
        """Check if buffer has enough samples for prediction"""
        return len(self.buffer) == self.window_size
    
    def extract_features(self):
        """Extract features from current buffer"""
        window = np.array(self.buffer)
        
        # Calculate features (same as training)
        rms = np.sqrt(np.mean(window ** 2))
        mav = np.mean(np.abs(window))
        wl = np.sum(np.abs(np.diff(window)))
        
        # Zero crossings (mean-centered)
        window_centered = window - np.mean(window)
        zc = 0
        threshold = 10
        for i in range(len(window_centered) - 1):
            if abs(window_centered[i] - window_centered[i+1]) >= threshold:
                if window_centered[i] * window_centered[i+1] < 0:
                    zc += 1
        
        # Slope sign changes
        ssc = 0
        for i in range(1, len(window) - 1):
            diff_prev = window[i] - window[i-1]
            diff_next = window[i] - window[i+1]
            if abs(diff_prev) >= threshold or abs(diff_next) >= threshold:
                if diff_prev * diff_next > 0:
                    ssc += 1
        
        return np.array([[rms, mav, wl, zc, ssc]])
    
    def predict(self):
        """Predict if current window is a scroll gesture"""
        if not self.is_ready():
            return None
        
        features = self.extract_features()
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0]
        
        return {
            'is_scroll': bool(prediction),
            'confidence': probability[1] if prediction else probability[0]
        }

class DoomscrollingBucket:
    """
    Exponential bucket algorithm for doomscrolling detection
    
    Mechanics:
    - Each scroll adds to bucket exponentially as scrolling continues
    - Bucket drains gradually over time
    - Scrolling interrupted after 1 minute of no scrolls
    - User is "zapped" when bucket overflows
    """
    
    def __init__(self, 
                 bucket_capacity=100.0,
                 base_scroll_value=1.0,
                 exponential_factor=1.15,
                 drain_rate=0.5,  # units per second
                 timeout_seconds=60,
                 overflow_callback=None):
        """
        Initialize doomscrolling bucket
        
        Args:
            bucket_capacity: Maximum bucket level before overflow/zap
            base_scroll_value: Initial value added per scroll
            exponential_factor: Multiplier for consecutive scrolls
            drain_rate: Bucket drainage per second
            timeout_seconds: Time before scroll chain resets
            overflow_callback: Function to call when bucket overflows
        """
        self.capacity = bucket_capacity
        self.base_value = base_scroll_value
        self.exp_factor = exponential_factor
        self.drain_rate = drain_rate
        self.timeout = timedelta(seconds=timeout_seconds)
        self.overflow_callback = overflow_callback
        
        # State
        self.level = 0.0
        self.consecutive_scrolls = 0
        self.last_scroll_time = None
        self.last_update_time = datetime.now()
        self.total_scrolls = 0
        self.overflow_count = 0
        self.is_overflowed = False
        
    def add_scroll(self):
        """Register a scroll event"""
        now = datetime.now()
        
        # Check if scroll chain is broken (timeout)
        if self.last_scroll_time and (now - self.last_scroll_time) > self.timeout:
            print(f"\n⏸️  Scroll chain broken after {self.timeout.seconds}s timeout")
            print(f"   Consecutive scrolls reset: {self.consecutive_scrolls} -> 0")
            self.consecutive_scrolls = 0
        
        # Update drainage first
        self._update_drain(now)
        
        # Calculate scroll value with exponential growth
        scroll_value = self.base_value * (self.exp_factor ** self.consecutive_scrolls)
        
        # Add to bucket
        self.level += scroll_value
        self.consecutive_scrolls += 1
        self.total_scrolls += 1
        self.last_scroll_time = now
        
        # Check for overflow
        if self.level >= self.capacity and not self.is_overflowed:
            self._trigger_overflow()
        
        return scroll_value
    
    def _update_drain(self, now=None):
        """Update bucket level based on time-based drainage"""
        if now is None:
            now = datetime.now()
        
        time_elapsed = (now - self.last_update_time).total_seconds()
        drain_amount = self.drain_rate * time_elapsed
        
        self.level = max(0, self.level - drain_amount)
        self.last_update_time = now
        
        # Reset overflow state if drained below capacity
        if self.level < self.capacity:
            self.is_overflowed = False
    
    def _trigger_overflow(self):
        """Handle bucket overflow event"""
        self.is_overflowed = True
        self.overflow_count += 1
        
        print("\n" + "="*60)
        print("⚠️  DOOMSCROLLING DETECTED - BUCKET OVERFLOW! ⚠️")
        print("="*60)
        print(f"Bucket level: {self.level:.2f}/{self.capacity}")
        print(f"Consecutive scrolls: {self.consecutive_scrolls}")
        print(f"Total scrolls: {self.total_scrolls}")
        print(f"Overflow count: {self.overflow_count}")
        print("="*60)
        
        # Trigger zap callback
        if self.overflow_callback:
            self.overflow_callback()
    
    def get_status(self):
        """Get current bucket status"""
        self._update_drain()
        
        return {
            'level': self.level,
            'capacity': self.capacity,
            'fill_percentage': (self.level / self.capacity) * 100,
            'consecutive_scrolls': self.consecutive_scrolls,
            'total_scrolls': self.total_scrolls,
            'overflow_count': self.overflow_count,
            'is_overflowed': self.is_overflowed
        }
    
    def reset(self):
        """Reset bucket to initial state"""
        self.level = 0.0
        self.consecutive_scrolls = 0
        self.last_scroll_time = None
        self.last_update_time = datetime.now()
        self.is_overflowed = False
        print("\n🔄 Bucket reset to 0")

def zap_user():
    """
    Callback function to 'zap' user when doomscrolling is detected
    In practice, this could:
    - Trigger a vibration motor
    - Send a notification
    - Activate a small electrical stimulation (with proper safety!)
    - Lock the screen temporarily
    """
    print("\n⚡ ZAP! ⚡")
    print("User intervention triggered!")
    # Add your zap mechanism here
    # Examples:
    # - serial_conn.write(b'Z')  # Send zap command to Arduino
    # - trigger_vibration_motor()
    # - send_push_notification()

def main():
    print("="*60)
    print("REAL-TIME SCROLL DETECTION WITH DOOMSCROLLING INTERVENTION")
    print("="*60)
    
    # Configuration
    SERIAL_PORT = 'COM9'  # Change to your Arduino port
    BAUDRATE = 115200
    SAMPLE_RATE = 200  # Hz, must match Arduino
    
    # Initialize components
    detector = ScrollDetector(model_path='models/scroll_detector_rf.pkl')
    bucket = DoomscrollingBucket(
        bucket_capacity=50.0,
        base_scroll_value=1.0,
        exponential_factor=1.2,
        drain_rate=0.3,
        timeout_seconds=60,
        overflow_callback=zap_user
    )
    
    # Connect to Arduino
    print(f"\nConnecting to Arduino on {SERIAL_PORT}...")
    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
        time.sleep(2)
        print("✓ Connected to Arduino")
    except Exception as e:
        print(f"✗ Error connecting to Arduino: {e}")
        return
    
    print("\n" + "="*60)
    print("MONITORING STARTED")
    print("="*60)
    print("Detecting scroll gestures in real-time...")
    print("Bucket will overflow at 50 units")
    print("Press Ctrl+C to stop\n")
    
    # Real-time monitoring
    scroll_cooldown = 0
    last_status_time = time.time()
    
    try:
        while True:
            # Read sensor data
            if ser.in_waiting:
                try:
                    line = ser.readline().decode('utf-8').strip()
                    
                    # Skip comments
                    if line.startswith('#'):
                        continue
                    
                    # Parse sensor value (assuming raw value transmission)
                    sensor_value = int(line)
                    
                    # Add to detector buffer
                    detector.add_sample(sensor_value)
                    
                    # Make prediction when buffer is ready
                    if detector.is_ready():
                        result = detector.predict()
                        
                        if result and result['is_scroll']:
                            # Cooldown to avoid double-counting
                            if scroll_cooldown <= 0:
                                scroll_value = bucket.add_scroll()
                                status = bucket.get_status()
                                
                                print(f"🔴 SCROLL detected (conf: {result['confidence']:.2f}) | "
                                      f"Bucket: {status['level']:.1f}/{status['capacity']} "
                                      f"({status['fill_percentage']:.1f}%) | "
                                      f"Chain: {status['consecutive_scrolls']} | "
                                      f"+{scroll_value:.2f}")
                                
                                scroll_cooldown = 10  # Cooldown samples
                            else:
                                scroll_cooldown -= 1
                        else:
                            if scroll_cooldown > 0:
                                scroll_cooldown -= 1
                    
                    # Periodic status update
                    if time.time() - last_status_time > 10:
                        status = bucket.get_status()
                        print(f"\n📊 Status: Bucket {status['level']:.1f}/{status['capacity']} "
                              f"| Chain: {status['consecutive_scrolls']} "
                              f"| Total: {status['total_scrolls']} scrolls")
                        last_status_time = time.time()
                        
                except ValueError:
                    continue
                    
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")
        
    finally:
        # Final statistics
        final_status = bucket.get_status()
        print("\n" + "="*60)
        print("SESSION SUMMARY")
        print("="*60)
        print(f"Total scrolls detected: {final_status['total_scrolls']}")
        print(f"Overflow events (zaps): {final_status['overflow_count']}")
        print(f"Final bucket level: {final_status['level']:.2f}/{final_status['capacity']}")
        print("="*60)
        
        ser.close()
        print("\n✓ Serial connection closed")

if __name__ == "__main__":
    main()
