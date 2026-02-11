/*
 * sEMG Real-time Monitoring for Scroll Detection
 * 
 * This sketch continuously streams sensor data for real-time classification
 * Simpler than the data collection sketch - just reads and transmits
 * 
 * Hardware:
 * - Myoware sensor on thumb palm muscle -> A0
 * - Optional: Zap output on pin 9 (PWM for vibration motor or LED)
 */

const int sensorPin = A0;
const int zapPin = 9;  // For vibration motor or other feedback

const int SAMPLE_RATE = 200;  // 200 Hz
const unsigned long SAMPLE_INTERVAL = 1000000 / SAMPLE_RATE;  // microseconds

unsigned long lastSampleTime = 0;

void setup() {
  Serial.begin(115200);
  pinMode(sensorPin, INPUT);
  pinMode(zapPin, OUTPUT);
  
  digitalWrite(zapPin, LOW);
  
  // Brief startup indicator
  for(int i = 0; i < 3; i++) {
    digitalWrite(zapPin, HIGH);
    delay(100);
    digitalWrite(zapPin, LOW);
    delay(100);
  }
  
  Serial.println("# sEMG Real-time Monitor Ready");
  Serial.println("# Streaming at 200 Hz");
  Serial.println("# Send 'Z' to trigger zap test");
  delay(100);
}

void loop() {
  // Check for commands from Python
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    if (cmd == 'Z' || cmd == 'z') {
      triggerZap();
    }
  }
  
  // Sample at fixed rate
  unsigned long currentMicros = micros();
  if (currentMicros - lastSampleTime >= SAMPLE_INTERVAL) {
    lastSampleTime = currentMicros;
    
    int sensorValue = analogRead(sensorPin);
    Serial.println(sensorValue);
  }
}

void triggerZap() {
  // Zap feedback - vibration pulse or LED flash
  // Adjust intensity and duration as needed
  
  // Vibration pulse pattern
  for(int i = 0; i < 3; i++) {
    analogWrite(zapPin, 200);  // PWM for vibration motor (0-255)
    delay(100);
    analogWrite(zapPin, 0);
    delay(50);
  }
  
  Serial.println("# Zap delivered!");
}
