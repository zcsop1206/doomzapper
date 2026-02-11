#!/usr/bin/env python3
"""
Quick Start Guide - sEMG Scroll Detection Pipeline

This script guides you through the entire pipeline step-by-step
"""

import os
import sys

def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def check_dependencies():
    """Check if required packages are installed"""
    print_header("STEP 0: Checking Dependencies")
    
    required = ['numpy', 'pandas', 'scipy', 'sklearn', 'serial', 'joblib']
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - NOT FOUND")
            missing.append(package)
    
    if missing:
        print("\n⚠️  Missing packages detected!")
        print("Install with: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All dependencies installed!")
        return True

def setup_directories():
    """Create necessary directories"""
    print_header("Creating Project Directories")
    
    dirs = ['data', 'models']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"✓ {d}/")
    
    print("\n✓ Directory structure ready!")

def guide_hardware_setup():
    """Guide user through hardware setup"""
    print_header("STEP 1: Hardware Setup")
    
    print("□ Connect Myoware sensor to Arduino:")
    print("  - Sensor Output (Signal) → Arduino Pin A0")
    print("  - Sensor GND → Arduino GND")
    print("  - Sensor VCC → Arduino 5V")
    print()
    print("□ Optional: Connect vibration motor/LED to Arduino Pin 9")
    print()
    print("□ Attach electrodes to Myoware sensor")
    print()
    print("□ Place sensor on thenar eminence (thumb palm muscle)")
    print("   [Reference electrode on wrist or elbow]")
    print()
    print("□ Connect Arduino to computer via USB")
    print()
    
    input("Press Enter when hardware setup is complete...")

def guide_arduino_upload():
    """Guide Arduino sketch upload"""
    print_header("STEP 2: Upload Arduino Sketch")
    
    print("For Data Collection:")
    print("  1. Open Arduino IDE")
    print("  2. Open 'sEMG_data_collector.ino'")
    print("  3. Select your board (Tools > Board)")
    print("  4. Select your port (Tools > Port)")
    print("  5. Click Upload")
    print()
    
    input("Press Enter when Arduino sketch is uploaded...")

def guide_serial_port():
    """Help user find serial port"""
    print_header("STEP 3: Find Serial Port")
    
    print("Finding your Arduino's serial port:")
    print()
    
    if sys.platform.startswith('win'):
        print("Windows: Usually COM3, COM4, COM5, etc.")
        print("  Check in Arduino IDE: Tools > Port")
    elif sys.platform.startswith('linux'):
        print("Linux: Usually /dev/ttyACM0 or /dev/ttyUSB0")
        print("  Run: ls /dev/tty* | grep -E '(ACM|USB)'")
    elif sys.platform.startswith('darwin'):
        print("Mac: Usually /dev/cu.usbmodem* or /dev/cu.usbserial*")
        print("  Run: ls /dev/cu.*")
    
    print()
    port = input("Enter your Arduino port (e.g., COM3): ").strip()
    
    return port

def update_port_in_files(port):
    """Update serial port in Python scripts"""
    files_to_update = ['collect_data.py', 'realtime_detection.py']
    
    for filename in files_to_update:
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Update port
            content = content.replace("PORT = 'COM3'", f"PORT = '{port}'")
            content = content.replace("SERIAL_PORT = 'COM3'", f"SERIAL_PORT = '{port}'")
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✓ Updated {filename}")

def guide_data_collection():
    """Guide through data collection"""
    print_header("STEP 4: Data Collection")
    
    print("You will now collect training and testing data.")
    print()
    print("Tips for good data:")
    print("  • Relax between collections")
    print("  • Vary scroll intensity (soft and hard)")
    print("  • Keep sensor placement consistent")
    print("  • Collect in similar posture you'll use the system")
    print()
    print("The script will guide you through:")
    print("  1. Training data (150 scroll + 150 rest)")
    print("  2. Testing data (50 scroll + 50 rest)")
    print()
    
    ready = input("Ready to start? (y/n): ").lower()
    
    if ready == 'y':
        print("\nRunning data collection...")
        print("Execute: python collect_data.py")
        print()
    else:
        print("Skipping data collection. Run manually: python collect_data.py")

def guide_training():
    """Guide through model training"""
    print_header("STEP 5: Model Training")
    
    print("This step will:")
    print("  • Extract features (RMS, MAV, WL, ZC, SSC)")
    print("  • Perform hyperparameter tuning")
    print("  • Train Random Forest classifier")
    print("  • Evaluate on test set")
    print("  • Save trained model")
    print()
    print("Expected time: 1-3 minutes")
    print()
    
    ready = input("Ready to train model? (y/n): ").lower()
    
    if ready == 'y':
        print("\nRunning training...")
        print("Execute: python train_model.py")
        print()
    else:
        print("Skipping training. Run manually: python train_model.py")

def guide_realtime():
    """Guide through real-time detection"""
    print_header("STEP 6: Real-Time Detection")
    
    print("Before running real-time detection:")
    print()
    print("□ Upload 'sEMG_realtime_monitor.ino' to Arduino")
    print("  (Different sketch than data collection!)")
    print()
    print("□ Ensure trained model exists: models/scroll_detector_rf.pkl")
    print()
    print("The system will:")
    print("  • Detect scrolls in real-time")
    print("  • Track bucket level")
    print("  • Trigger 'zap' on overflow")
    print()
    
    ready = input("Ready to start real-time detection? (y/n): ").lower()
    
    if ready == 'y':
        print("\nRunning real-time detection...")
        print("Execute: python realtime_detection.py")
        print()
        print("Press Ctrl+C to stop")
        print()
    else:
        print("Skipping real-time. Run manually: python realtime_detection.py")

def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║   sEMG Scroll Detection - Quick Start Guide                  ║
    ║   Doomscrolling Intervention System                          ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Check dependencies
    if not check_dependencies():
        print("\nPlease install dependencies first:")
        print("  pip install -r requirements.txt")
        return
    
    # Setup directories
    setup_directories()
    
    # Guide through hardware
    guide_hardware_setup()
    
    # Guide Arduino upload
    guide_arduino_upload()
    
    # Get serial port
    port = guide_serial_port()
    update_port_in_files(port)
    
    # Guide data collection
    guide_data_collection()
    
    # Ask if data collection is complete
    print()
    collected = input("Have you completed data collection? (y/n): ").lower()
    
    if collected == 'y':
        # Guide training
        guide_training()
        
        # Ask if training is complete
        print()
        trained = input("Has model training completed successfully? (y/n): ").lower()
        
        if trained == 'y':
            # Guide real-time
            guide_realtime()
        else:
            print("\nComplete training first, then run: python realtime_detection.py")
    else:
        print("\nComplete data collection first, then run: python train_model.py")
    
    print_header("Setup Complete!")
    print("For detailed information, see README.md")
    print()
    print("Pipeline Summary:")
    print("  1. collect_data.py       → Gather training/testing data")
    print("  2. train_model.py        → Extract features & train classifier")
    print("  3. realtime_detection.py → Real-time detection with bucket algorithm")
    print()
    print("Happy detecting! 📱⚡")

if __name__ == "__main__":
    main()