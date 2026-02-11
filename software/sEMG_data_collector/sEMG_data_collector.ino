/*
 * sEMG Myoware Sensor Data Collector for Scroll Detection
 * 
 * Hardware Setup:
 * - Myoware sensor on thumb palm muscle (thenar eminence)
 * - Sensor output (envelope) -> Arduino analog pin A0
 * - LED on pin 13 for visual feedback during collection
 * - Button on pin 2 for manual event marking (optional)
 * 
 * Data Collection Protocol:
 * 1. Collect "scroll" gestures (thumb flexion)
 * 2. Collect "rest" state (no movement)
 * 3. Data sent via Serial in CSV format with timestamp and label
 */

const int sensorPin = A0;
const int ledPin = 13;
const int buttonPin = 2;

const int SAMPLE_RATE = 200;  // 200 Hz sampling rate
const int WINDOW_SIZE = 40;   // 200ms window (40 samples at 200Hz)
unsigned long lastSampleTime = 0;
const unsigned long SAMPLE_INTERVAL = 1000000 / SAMPLE_RATE; // microseconds

// Collection modes
enum CollectionMode {
  IDLE,
  COLLECT_SCROLL,
  COLLECT_REST
};

CollectionMode currentMode = IDLE;
int windowBuffer[WINDOW_SIZE];
int bufferIndex = 0;
unsigned long windowStartTime = 0;
int windowCount = 0;

void setup() {
  Serial.begin(115200);
  pinMode(sensorPin, INPUT);
  pinMode(ledPin, OUTPUT);
  pinMode(buttonPin, INPUT_PULLUP);
  
  Serial.println("sEMG Data Collector Ready");
  Serial.println("Commands:");
  Serial.println("  's' - Start collecting SCROLL gestures");
  Serial.println("  'r' - Start collecting REST data");
  Serial.println("  'x' - Stop collection");
  //Serial.println("");
  //Serial.println("Format: timestamp,value1,value2,...,valueN,label");
}

void loop() {
  // Check for serial commands
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    handleCommand(cmd);
  }
  
  // Sample at fixed rate
  unsigned long currentMicros = micros();
  if (currentMicros - lastSampleTime >= SAMPLE_INTERVAL) {
    lastSampleTime = currentMicros;
    
    int sensorValue = analogRead(sensorPin);
    
    if (currentMode != IDLE) {
      // Add to window buffer
      windowBuffer[bufferIndex] = sensorValue;
      bufferIndex++;
      
      // When window is full, output it
      if (bufferIndex >= WINDOW_SIZE) {
        outputWindow();
        bufferIndex = 0;
        windowCount++;
        
        // Blink LED to show collection activity
        digitalWrite(ledPin, windowCount % 2);
      }
    }
  }
}

void handleCommand(char cmd) {
  switch(cmd) {
    case 's':
    case 'S':
      currentMode = COLLECT_SCROLL;
      bufferIndex = 0;
      windowCount = 0;
      Serial.println("# Collecting SCROLL gestures. Perform thumb flexion movements.");
      digitalWrite(ledPin, HIGH);
      break;
      
    case 'r':
    case 'R':
      currentMode = COLLECT_REST;
      bufferIndex = 0;
      windowCount = 0;
      Serial.println("# Collecting REST data. Keep hand relaxed.");
      digitalWrite(ledPin, HIGH);
      break;
      
    case 'x':
    case 'X':
      currentMode = IDLE;
      bufferIndex = 0;
      Serial.println("# Collection stopped");
      Serial.print("# Total windows collected: ");
      Serial.println(windowCount);
      digitalWrite(ledPin, LOW);
      break;
  }
}

void outputWindow() {
  // Output format: timestamp,val1,val2,...,valN,label
  Serial.print(millis());
  Serial.print(",");
  
  for (int i = 0; i < WINDOW_SIZE; i++) {
    Serial.print(windowBuffer[i]);
    if (i < WINDOW_SIZE - 1) {
      Serial.print(",");
    }
  }
  
  Serial.print(",");
  Serial.println(currentMode == COLLECT_SCROLL ? "scroll" : "rest");
}
