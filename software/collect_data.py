"""
sEMG Data Collection Interface
Receives data from Arduino and saves to CSV files for training and testing
"""

import serial
import time
import csv
import os
from datetime import datetime

class sEMGDataCollector:
    def __init__(self, port='COM3', baudrate=115200):
        """
        Initialize the data collector
        
        Args:
            port: Serial port (e.g., 'COM3' on Windows, '/dev/ttyACM0' on Linux)
            baudrate: Serial communication speed (must match Arduino)
        """
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.data_dir = 'data'
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
    def connect(self):
        """Establish serial connection to Arduino"""
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # Wait for Arduino to reset
            print(f"Connected to Arduino on {self.port}")
            
            # Read initial messages
            for _ in range(5):
                if self.serial_conn.in_waiting:
                    line = self.serial_conn.readline().decode('utf-8').strip()
                    print(line)
                    
            return True
        except Exception as e:
            print(f"Error connecting to Arduino: {e}")
            return False
    
    def collect_dataset(self, dataset_name, num_scroll_windows=100, num_rest_windows=100):
        """
        Collect a complete dataset with scroll and rest samples
        
        Args:
            dataset_name: Name for this dataset (e.g., 'training', 'testing')
            num_scroll_windows: Number of scroll gesture windows to collect
            num_rest_windows: Number of rest state windows to collect
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.data_dir, f'{dataset_name}_{timestamp}.csv')
        
        print(f"\n{'='*60}")
        print(f"Collecting dataset: {dataset_name}")
        print(f"Target: {num_scroll_windows} scroll + {num_rest_windows} rest windows")
        print(f"Saving to: {filename}")
        print(f"{'='*60}\n")
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Collect SCROLL data
            print("\n--- COLLECTING SCROLL GESTURES ---")
            print("Instructions: Perform thumb flexion movements (scrolling motion)")
            print("Press Enter when ready...")
            input()
            
            self.serial_conn.write(b's')  # Start scroll collection
            time.sleep(0.5)
            
            scroll_count = 0
            while scroll_count < num_scroll_windows:
                if self.serial_conn.in_waiting:
                    line = self.serial_conn.readline().decode('utf-8').strip()
                    
                    if line.startswith('#'):
                        print(line)
                    else:
                        writer.writerow(line.split(','))
                        scroll_count += 1
                        if scroll_count % 10 == 0:
                            print(f"Collected {scroll_count}/{num_scroll_windows} scroll windows")
            
            self.serial_conn.write(b'x')  # Stop collection
            time.sleep(1)
            print(f"✓ Scroll data collection complete: {scroll_count} windows\n")
            
            # Collect REST data
            print("\n--- COLLECTING REST DATA ---")
            print("Instructions: Keep your hand completely relaxed")
            print("Press Enter when ready...")
            input()
            
            self.serial_conn.write(b'r')  # Start rest collection
            time.sleep(0.5)
            
            rest_count = 0
            while rest_count < num_rest_windows:
                if self.serial_conn.in_waiting:
                    line = self.serial_conn.readline().decode('utf-8').strip()
                    
                    if line.startswith('#'):
                        print(line)
                    else:
                        writer.writerow(line.split(','))
                        rest_count += 1
                        if rest_count % 10 == 0:
                            print(f"Collected {rest_count}/{num_rest_windows} rest windows")
            
            self.serial_conn.write(b'x')  # Stop collection
            time.sleep(1)
            print(f"✓ Rest data collection complete: {rest_count} windows\n")
        
        print(f"\n{'='*60}")
        print(f"Dataset complete! Saved to {filename}")
        print(f"Total windows: {scroll_count + rest_count}")
        print(f"{'='*60}\n")
        
        return filename
    
    def close(self):
        """Close serial connection"""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print("Serial connection closed")

def main():
    # Configure your Arduino port here
    PORT = 'COM9'  # Change to your port: Windows: 'COM3', Linux: '/dev/ttyACM0', Mac: '/dev/cu.usbmodem'
    
    collector = sEMGDataCollector(port=PORT)
    
    if not collector.connect():
        print("Failed to connect. Check port and try again.")
        return
    
    try:
        # Collect training data
        print("\n" + "="*60)
        print("TRAINING DATA COLLECTION")
        print("="*60)
        collector.collect_dataset('training', num_scroll_windows=150, num_rest_windows=150)
        
        print("\nTake a 2-minute break before test data collection...")
        time.sleep(5)  # Short break
        
        # Collect testing data
        print("\n" + "="*60)
        print("TESTING DATA COLLECTION")
        print("="*60)
        collector.collect_dataset('testing', num_scroll_windows=50, num_rest_windows=50)
        
        print("\n✓ All data collection complete!")
        
    except KeyboardInterrupt:
        print("\n\nCollection interrupted by user")
    finally:
        collector.close()

if __name__ == "__main__":
    main()
