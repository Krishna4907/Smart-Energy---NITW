/* ============================================================
   NILM FEATURE CAPTURE  - ESP32  (for training your OWN model)
   Same sensors/wiring as nilm_monitor. No WiFi needed.

   It prints one CSV line of the 16-feature vector a few times a
   second, prefixed with "FEAT,". Run python/capture_own_data.py on
   the PC to tag these rows with an appliance label and save them.
   Then python/train_own.py trains a Random Forest on YOUR data --
   this is the most accurate model for your exact hardware because
   the features are produced by the very same code that runs live.

   Procedure:
     1. Flash this, open Serial @115200 (or just run capture script).
     2. Plug in ONE known load (e.g. 100W bulb).
     3. On PC:  python capture_own_data.py --label bulb_100w --seconds 60
     4. Repeat for each load AND each combination you care about
        (bulb_60w, soldering, bulb_60w+bulb_100w, all_three, off ...).
     5. python train_own.py
   ============================================================ */

#include <math.h>

#define CURRENT_PIN 34
#define VOLTAGE_PIN 35
const float ADC_REF = 3.3f, ADC_MAX = 4095.0f, ACS712_VPA = 0.185f, MAINS_HZ = 50.0f;
float voltageCalibration = 1.75f, currentCalibration = 0.39f;

#define SAMPLE_RATE 4000
#define NUM_SAMPLES 400
#define N_HARM 8
int16_t curRaw[NUM_SAMPLES], volRaw[NUM_SAMPLES];

void captureBurst() {
  const unsigned long periodUs = 1000000UL / SAMPLE_RATE;
  unsigned long next = micros();
  for (int k = 0; k < NUM_SAMPLES; k++) {
    while ((long)(micros() - next) < 0) {}
    curRaw[k] = analogRead(CURRENT_PIN);
    volRaw[k] = analogRead(VOLTAGE_PIN);
    next += periodUs;
  }
}

float goertzel(const float* x, int n, float freq) {
  float w = 2.0f * PI * freq / SAMPLE_RATE, coeff = 2.0f * cosf(w);
  float s0, s1 = 0, s2 = 0;
  for (int k = 0; k < n; k++) { s0 = x[k] + coeff * s1 - s2; s2 = s1; s1 = s0; }
  float p = s1 * s1 + s2 * s2 - coeff * s1 * s2;
  return sqrtf(p > 0 ? p : 0) * 2.0f / n;
}

void printFeatures() {
  static float cur[NUM_SAMPLES], vol[NUM_SAMPLES];
  double cMean = 0, vMean = 0;
  for (int k = 0; k < NUM_SAMPLES; k++) { cMean += curRaw[k]; vMean += volRaw[k]; }
  cMean /= NUM_SAMPLES; vMean /= NUM_SAMPLES;

  double sqI = 0, sqV = 0, sumAbsI = 0, sumVI = 0; float iPeak = 0;
  int zc = 0; float prev = 0;
  const float k2a = (ADC_REF / ADC_MAX) / ACS712_VPA * currentCalibration;
  for (int k = 0; k < NUM_SAMPLES; k++) {
    float i = (curRaw[k] - cMean) * k2a;
    float v = (volRaw[k] - vMean) * voltageCalibration;
    cur[k] = i; vol[k] = v;
    sqI += (double)i * i; sqV += (double)v * v; sumAbsI += fabsf(i); sumVI += (double)v * i;
    if (fabsf(i) > iPeak) iPeak = fabsf(i);
    if (k > 0 && ((prev < 0 && i >= 0) || (prev > 0 && i <= 0))) zc++;
    prev = i;
  }
  float iRms = sqrtf(sqI / NUM_SAMPLES), vRms = sqrtf(sqV / NUM_SAMPLES);
  float realP = sumVI / NUM_SAMPLES, meanAbs = sumAbsI / NUM_SAMPLES + 1e-9f;
  float crest = iPeak / (iRms + 1e-9f), form = iRms / meanAbs;
  float mags[N_HARM];
  for (int h = 1; h <= N_HARM; h++) mags[h - 1] = goertzel(cur, NUM_SAMPLES, h * MAINS_HZ);
  float fund = mags[0] + 1e-9f, thdSq = 0;
  for (int h = 2; h <= N_HARM; h++) thdSq += mags[h - 1] * mags[h - 1];
  float thd = sqrtf(thdSq) / fund;
  float cyc = (float)NUM_SAMPLES * MAINS_HZ / SAMPLE_RATE;
  float zeroCross = zc / (cyc > 0 ? cyc : 1);
  float apparent = vRms * iRms, pf = apparent > 1e-6f ? realP / apparent : 0;
  if (pf > 1) pf = 1; if (pf < -1) pf = -1;

  Serial.print("FEAT");
  Serial.printf(",%.5f,%.5f,%.5f,%.5f,%.5f,%.5f", iRms, iPeak, crest, form, thd, zeroCross);
  for (int h = 2; h <= N_HARM; h++) Serial.printf(",%.5f", mags[h - 1] / fund);
  Serial.printf(",%.5f,%.5f,%.5f\n", realP, apparent, pf);
}

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);
  delay(500);
  Serial.println("# nilm_capture ready. Columns:");
  Serial.println("# FEAT,i_rms,i_peak,crest,form,thd,zero_cross,h2,h3,h4,h5,h6,h7,h8,real_power,apparent_power,power_factor");
}

void loop() {
  captureBurst();
  printFeatures();
  delay(300);   // ~3 feature rows per second
}
