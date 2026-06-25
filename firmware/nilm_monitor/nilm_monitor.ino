/* ============================================================
   SMART ENERGY MONITOR + NILM  - ESP32  (PC/CLOUD INFERENCE)
   ESP32 + ZMPT101B (voltage) + ACS712-5A (current) + OLED

   WHAT THIS ADDS over the basic monitor:
   - Captures one burst of synchronised voltage+current samples at a
     high rate (SAMPLE_RATE) every cycle.
   - Extracts the SAME NILM feature vector as python/features.py
     (RMS, peak, crest, form, THD, harmonics 2..8 via Goertzel,
      zero-crossings, real/apparent power, power factor).
   - POSTs {voltage, current, power, features[]} as JSON to the PC
     inference server, which classifies + disaggregates + bills.
   - Still shows V/I/P/energy on the OLED.

   WIRING (unchanged):
   - LIVE wire in series through ACS712 IP+ -> IP- screw terminals.
   - Common ground: ACS712 GND, ZMPT GND, ESP32 GND tied together.
   - ACS712 powered from 5V (VIN). ZMPT pot tuned so V is stable.
   ============================================================ */

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <math.h>

// ---------------- WIFI ----------------
const char* WIFI_SSID     = "aditya";
const char* WIFI_PASSWORD = "0307062006";

// ---------------- INFERENCE SERVER ----------------
// IP of the PC running python/inference_server.py, on the same WiFi.
// Find it with `ipconfig` (Windows) / `ip a` (Linux). Keep the :5000 port.
const char* SERVER_URL = "http://192.168.1.50:5000/ingest";

// ---------------- OLED ----------------
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// ---------------- PINS ----------------
#define CURRENT_PIN 34
#define VOLTAGE_PIN 35
#define SDA_PIN     21
#define SCL_PIN     22

// ---------------- SENSOR CONSTANTS ----------------
const float ADC_REF    = 3.3f;
const float ADC_MAX    = 4095.0f;
const float ACS712_VPA = 0.185f;   // 5A module
const float MAINS_HZ   = 50.0f;    // India 50 Hz

float voltageCalibration = 1.75f;  // tune ZMPT to real line voltage
float currentCalibration = 0.39f;  // tune: true_amps / shown_amps
const float CURRENT_FLOOR_A = 0.05f;

// ---------------- HIGH-RATE CAPTURE ----------------
// 4 kHz over 100 ms = 400 sample pairs = 5 mains cycles at 50 Hz.
#define SAMPLE_RATE 4000
#define NUM_SAMPLES 400
#define N_HARM      8              // fundamental + harmonics 2..8
int16_t curRaw[NUM_SAMPLES];
int16_t volRaw[NUM_SAMPLES];

// ---------------- STATE ----------------
float voltage = 0, current = 0, power = 0, energy = 0;
float currentOffset = 0, voltageOffset = 0, currentNoiseCounts = 0;
unsigned long lastUpdateTime = 0;
const unsigned long UPDATE_INTERVAL = 15000UL;
String lastType = "-";

// ============================================================
// Synchronised burst capture: read current then voltage each tick,
// paced to SAMPLE_RATE using micros().
// ============================================================
void captureBurst() {
  const unsigned long periodUs = 1000000UL / SAMPLE_RATE;
  unsigned long next = micros();
  for (int k = 0; k < NUM_SAMPLES; k++) {
    while ((long)(micros() - next) < 0) { /* wait */ }
    curRaw[k] = analogRead(CURRENT_PIN);
    volRaw[k] = analogRead(VOLTAGE_PIN);
    next += periodUs;
  }
}

// Goertzel magnitude of `freq` in a real signal sampled at SAMPLE_RATE.
float goertzel(const float* x, int n, float freq) {
  float w = 2.0f * PI * freq / SAMPLE_RATE;
  float coeff = 2.0f * cosf(w);
  float s0, s1 = 0, s2 = 0;
  for (int k = 0; k < n; k++) {
    s0 = x[k] + coeff * s1 - s2;
    s2 = s1; s1 = s0;
  }
  float power = s1 * s1 + s2 * s2 - coeff * s1 * s2;
  return sqrtf(power > 0 ? power : 0) * 2.0f / n;
}

// ============================================================
// Extract the 16-feature vector (order must match python/features.py).
// Fills feats[16]; also returns Vrms, Irms, P by reference.
// ============================================================
void extractFeatures(float* feats, float& vRms, float& iRms, float& realP) {
  static float cur[NUM_SAMPLES];   // calibrated amps, DC removed
  static float vol[NUM_SAMPLES];   // calibrated volts, DC removed

  // means for DC removal
  double cMean = 0, vMean = 0;
  for (int k = 0; k < NUM_SAMPLES; k++) { cMean += curRaw[k]; vMean += volRaw[k]; }
  cMean /= NUM_SAMPLES; vMean /= NUM_SAMPLES;

  double sqI = 0, sqV = 0, sumAbsI = 0, sumVI = 0;
  float iPeak = 0;
  int zc = 0; float prev = 0;
  const float countsToAmps = (ADC_REF / ADC_MAX) / ACS712_VPA * currentCalibration;

  for (int k = 0; k < NUM_SAMPLES; k++) {
    float i = (curRaw[k] - cMean) * countsToAmps;
    float v = (volRaw[k] - vMean) * voltageCalibration;
    cur[k] = i; vol[k] = v;
    sqI += (double)i * i; sqV += (double)v * v;
    sumAbsI += fabsf(i); sumVI += (double)v * i;
    if (fabsf(i) > iPeak) iPeak = fabsf(i);
    if (k > 0 && ((prev < 0 && i >= 0) || (prev > 0 && i <= 0))) zc++;
    prev = i;
  }

  iRms = sqrtf(sqI / NUM_SAMPLES);
  vRms = sqrtf(sqV / NUM_SAMPLES);
  realP = sumVI / NUM_SAMPLES;
  float meanAbs = sumAbsI / NUM_SAMPLES + 1e-9f;
  float crest = iPeak / (iRms + 1e-9f);
  float form  = iRms / meanAbs;

  // harmonic magnitudes of current (fundamental + 2..8)
  float mags[N_HARM];
  for (int h = 1; h <= N_HARM; h++)
    mags[h - 1] = goertzel(cur, NUM_SAMPLES, h * MAINS_HZ);
  float fund = mags[0] + 1e-9f;
  float thdSq = 0;
  for (int h = 2; h <= N_HARM; h++) thdSq += mags[h - 1] * mags[h - 1];
  float thd = sqrtf(thdSq) / fund;

  float cyclesCaptured = (float)NUM_SAMPLES * MAINS_HZ / SAMPLE_RATE;
  float zeroCross = zc / (cyclesCaptured > 0 ? cyclesCaptured : 1);

  float apparent = vRms * iRms;
  float pf = apparent > 1e-6f ? realP / apparent : 0.0f;
  if (pf > 1) pf = 1; if (pf < -1) pf = -1;

  int idx = 0;
  feats[idx++] = iRms;
  feats[idx++] = iPeak;
  feats[idx++] = crest;
  feats[idx++] = form;
  feats[idx++] = thd;
  feats[idx++] = zeroCross;
  for (int h = 2; h <= N_HARM; h++) feats[idx++] = mags[h - 1] / fund; // h2..h8
  feats[idx++] = realP;
  feats[idx++] = apparent;
  feats[idx++] = pf;
}

// ============================================================
// CALIBRATION (no appliance plugged in)
// ============================================================
struct AdcStats { float mean, rms; };
AdcStats sampleADC(int pin, unsigned long windowMs) {
  analogRead(pin); analogRead(pin);
  unsigned long t0 = millis(); long n = 0; double sum = 0;
  while (millis() - t0 < windowMs) { sum += analogRead(pin); n++; }
  float mean = sum / n;
  t0 = millis(); double sq = 0; long n2 = 0;
  while (millis() - t0 < windowMs) { float d = analogRead(pin) - mean; sq += d * d; n2++; }
  return { mean, (float)sqrt(sq / n2) };
}

void calibrateSensors() {
  Serial.println("\n=== CALIBRATING (no appliance plugged in) ===");
  display.clearDisplay(); display.setCursor(0, 0);
  display.println("Calibrating...");
  display.println("Unplug appliances"); display.display();
  double offC = 0, offV = 0, noiseC = 0; const int W = 5;
  for (int k = 0; k < W; k++) {
    AdcStats c = sampleADC(CURRENT_PIN, 200);
    AdcStats v = sampleADC(VOLTAGE_PIN, 200);
    offC += c.mean; noiseC += c.rms; offV += v.mean;
  }
  currentOffset = offC / W; currentNoiseCounts = noiseC / W; voltageOffset = offV / W;
  Serial.printf("ACS712 offset=%.1f noiseRMS=%.2f | ZMPT offset=%.1f\n",
                currentOffset, currentNoiseCounts, voltageOffset);
  Serial.println("=== CALIBRATION DONE ===\n");
}

// ---------------- WIFI ----------------
void connectWiFi() {
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  int a = 0;
  while (WiFi.status() != WL_CONNECTED && a < 30) { delay(500); Serial.print("."); a++; }
  Serial.println(WiFi.status() == WL_CONNECTED ? "\nWiFi OK" : "\nWiFi FAILED");
}

// ---------------- POST features to inference server ----------------
void sendToServer(float v, float i, float p, const float* feats, int nf) {
  if (WiFi.status() != WL_CONNECTED) { connectWiFi(); return; }
  String json = "{\"voltage\":" + String(v, 2) +
                ",\"current\":" + String(i, 3) +
                ",\"power\":" + String(p, 2) + ",\"features\":[";
  for (int k = 0; k < nf; k++) { json += String(feats[k], 5); if (k < nf - 1) json += ","; }
  json += "]}";

  HTTPClient http;
  http.begin(SERVER_URL);
  http.addHeader("Content-Type", "application/json");
  http.setTimeout(6000);
  int code = http.POST(json);
  if (code == 200) {
    String resp = http.getString();
    Serial.println("SERVER: " + resp);
    int t = resp.indexOf("\"rf_type\":\"");          // crude parse for OLED
    if (t >= 0) { int s = t + 11; int e = resp.indexOf("\"", s); lastType = resp.substring(s, e); }
  } else {
    Serial.println("server err " + String(code));
  }
  http.end();
}

// ---------------- OLED ----------------
void updateOLED(int countdown) {
  display.clearDisplay();
  display.setTextSize(1); display.setCursor(0, 0);
  display.println("Smart Energy + NILM");
  display.drawLine(0, 10, 128, 10, SSD1306_WHITE);
  display.setCursor(0, 14); display.printf("V:%.0f  I:%.3fA", voltage, current);
  display.setCursor(0, 26); display.printf("P:%.1f W", power);
  display.setCursor(0, 38); display.printf("E:%.4f kWh", energy);
  display.setCursor(0, 50); display.printf("Type:%s", lastType.c_str());
  display.setCursor(104, 50); display.printf("T%d", countdown);
  display.display();
}

// ============================================================
void setup() {
  Serial.begin(115200);
  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);
  Wire.begin(SDA_PIN, SCL_PIN);
  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) Serial.println("OLED not found!");
  display.clearDisplay(); display.setTextColor(SSD1306_WHITE); display.setTextSize(1);
  connectWiFi();
  delay(2000);
  calibrateSensors();
  lastUpdateTime = millis();
}

// ============================================================
void loop() {
  unsigned long now = millis();
  int secondsLeft = (UPDATE_INTERVAL - (now - lastUpdateTime)) / 1000;
  if (secondsLeft < 0) secondsLeft = 0;
  updateOLED(secondsLeft);

  if (now - lastUpdateTime >= UPDATE_INTERVAL) {
    lastUpdateTime = millis();

    captureBurst();
    float feats[16]; float vRms, iRms, realP;
    extractFeatures(feats, vRms, iRms, realP);

    voltage = vRms;
    current = (iRms < CURRENT_FLOOR_A) ? 0.0f : iRms;
    power   = (realP > 0) ? realP : voltage * current;   // prefer true real power

    float deltaHours = UPDATE_INTERVAL / 3600000.0f;
    if (power > 0) energy += (power * deltaHours) / 1000.0f;

    Serial.printf("== V=%.1f I=%.3f P=%.1f PF=%.2f THD=%.2f ==\n",
                  voltage, current, power, feats[15], feats[4]);
    sendToServer(voltage, current, power, feats, 16);
    updateOLED(15);
  }
  delay(500);
}
