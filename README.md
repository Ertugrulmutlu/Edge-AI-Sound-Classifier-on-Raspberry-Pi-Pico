# Edge AI Sound Classifier on Raspberry Pi Pico

This project implements a tiny **edge AI system** on the Raspberry Pi Pico (RP2040) to classify short sound snippets into four classes:

* 👶 Baby cry
* 🔔 Doorbell
* 🚨 Smoke alarm
* 🌫 Other / Background

When one of the alarm classes is detected, the Pico lights its onboard LED and logs probabilities and FSM states via USB serial.

---

## 📂 Repository Structure

```
Python side
├── dataset/
│   ├── raw/         # Raw audio (YouTube/Freesound/recordings)
│   └── prep/        # Preprocessed snippets (WAV)
|
├── dataset_tool/
│   ├── bulk_cut_data.py         # Cut raw audio into labeled snippets 
│   └── feature_extraction.py    # Extract 33-dim feature vectors per snippet
|
├── model_train/
│   └── train_ml.py  # Train Logistic Regression + export weights
|
├── simulation_tool/
│   └── simulation.py  # Live microphone → Pico inference
|
├── features/
│   └── featuresv1.csv  # Extracted features
├── firmware/
│   └── model_params.hpp # Auto-generated logistic regression params
```

```
cpp side
├── main.cpp          # C++ firmware: inference + FSM + LED
├── model_params.hpp  # Auto-generated logistic regression params
|
```

---

## 🔄 Data Pipeline

1. **Collect raw audio**: baby cries, doorbells, smoke alarms, and negatives.
2. **Cut into snippets** using `bulk_cut_data.py` (e.g., 1.5s windows, silence removed).
3. **Extract features** with `feature_extraction.py`:

   * 12 band energies (Goertzel, mean + std, z-scored)
   * RMS, spectral centroid, rolloff, ZCR, flatness
   * Total: **33 features** per snippet
4. **Train model** with `train_ml.py`:

   * Logistic Regression (multinomial, balanced class weights)
   * \~87% accuracy (F1 ≈ 0.86)
   * Best on smoke alarm, weakest on baby cry
   * Exports weights + normalization params → `firmware/model_params.hpp`

---

## 🚀 Deployment on Pico

* Firmware: `firmware/main.cpp`

  * Reads feature vectors via USB-CDC (CSV format)
  * Applies z-score normalization
  * Logistic regression inference → softmax
  * Hysteresis FSM per class (thresholds, consecutive frames)
  * LED ON for baby/doorbell/smoke alarm; OFF for "other"

---

## 🖥️ PC → Pico Communication

**Live microphone**:

   ```bash
   python python/simulation.py
   ```

   Captures from microphone, extracts features in real time, streams to Pico.

---

## 🔧 Requirements

 ```bash
   python pip install requirement.txt
  ````
* Raspberry Pi Pico SDK (for firmware build)

---

## 📊 Results

* **Accuracy**: **0.87** (weighted F1 **0.86**)
* **Per-class performance**

| Class        | Precision | Recall |   F1 | Support |
| ------------ | --------: | -----: | ---: | ------: |
| baby         |      0.82 |   0.69 | 0.75 |      72 |
| doorbell     |      0.79 |   0.90 | 0.84 |      72 |
| other        |      0.86 |   0.88 | 0.87 |      96 |
| smoke\_alarm |      1.00 |   0.99 | 0.99 |      72 |

**Confusion matrix (rows = true, cols = pred):**

```
[[50 12 10  0]
 [ 4 65  3  0]
 [ 7  5 84  0]
 [ 0  0  1 71]]
```

**Notes**

* Strongest: *smoke\_alarm* (almost perfect).
* Weakest recall: *baby* → often confused with *doorbell* and *other*.
* Dataset currently lacks multi-source diversity per class → fell back to stratified split.

---
**🎥 Demo**

[Youtube Video](https://www.youtube.com/watch?v=8fl_OHK0yhM&feature=youtu.be)
---
## ⚠️ Notes

* Snippet size and FSM thresholds trade off **latency vs. stability**.
* If training data has only one source per class, group-aware split is disabled.
* For Pico W, LED pin definition differs (update `LED_PIN`).

---

## 📜 License

MIT License — feel free to use and adapt.
