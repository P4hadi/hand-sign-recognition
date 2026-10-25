# Real-Time Hand Sign Recognition System

A Windows-based academic machine-learning prototype that recognizes eight
static hand gestures from a webcam. The system detects hand landmarks with
MediaPipe, converts them into 63 normalized features, and classifies them with
a scikit-learn Random Forest model.

This project is a gesture classifier, not a complete sign-language translation
system.

## Contents

- [Features](#features)
- [Supported gestures](#supported-gestures)
- [Technology stack](#technology-stack)
- [Project structure](#project-structure)
- [Prerequisites and installation](#prerequisites-and-installation)
- [MediaPipe model setup](#mediapipe-model-setup)
- [Run the application](#run-the-application)
- [Other useful commands](#other-useful-commands)
- [How the system works](#how-the-system-works)
- [Dataset and privacy](#dataset-and-privacy)
- [Evaluation and limitations](#evaluation-and-limitations)
- [Troubleshooting](#troubleshooting)
- [Future improvements](#future-improvements)
- [License](#license)

## Features

- Real-time webcam gesture recognition.
- MediaPipe Hand Landmarker detection and 21-point hand visualization.
- Wrist-centred, scale-normalized landmark features: 63 numeric inputs per hand.
- Random Forest classification with saved Joblib model bundles.
- Prediction probability estimates from the classifier.
- Short temporal majority smoothing to reduce display flicker.
- FPS and per-stage timing diagnostics during live inference.
- Model-contract validation for labels, estimator classes, and feature order.
- CSV validation, session-aware training workflows, and webcam-independent tests.
- Preserved five-class legacy model support through the default inference path.

## Supported gestures

The current eight-gesture vocabulary is:

```text
open_palm
fist
thumbs_up
peace_sign
ok_sign
pointing_up
three_fingers
call_me
```

The definitions and collection descriptions are centralized in
`gesture_config.py`.

## Technology stack

| Component | Version / role |
|---|---|
| Python | 3.12 |
| OpenCV | `opencv-python==5.0.0.93` |
| MediaPipe | `mediapipe==1.1.1` |
| NumPy | `numpy==2.5.4` |
| scikit-learn | `scikit-learn==1.9.1` |
| Joblib | `joblib==1.6.0` |

The exact Python dependencies are recorded in `requirements.txt`.

## Project structure

```text
hand-sign-recognition/
├── collect_data.py             # Collect normalized landmark CSV samples
├── gesture_config.py           # Eight-gesture vocabulary and descriptions
├── hand_landmarks.py           # Landmark-only webcam visualization
├── predict_webcam.py           # Realtime model inference
├── webcam_test.py              # Camera-only diagnostic
├── check_environment.py        # Read-only environment and API check
├── train_model.py              # Five-class baseline training workflow
├── prepare_dataset.py          # Session audit and safe dataset combining
├── train_session_model.py      # Session holdout and final-model workflows
├── requirements.txt
├── data/                       # Local landmark CSVs; not all are public data
├── models/
│   ├── hand_landmarker.task
│   └── gesture_rf_v4_eight_gestures_session06.joblib
├── reports/
│   ├── FINAL_PROJECT_STATUS.md
│   └── metrics/                # Evaluation JSON and confusion-matrix reports
└── tests/
    ├── test_collect_data.py
    ├── test_predict_webcam.py
    ├── test_session_training.py
    └── test_train_model.py
```

Historical datasets, model versions, backups, and reports are retained in the
workspace but are not all listed in this overview.

## Prerequisites and installation

The documented commands below use PowerShell and project-relative paths.

1. Install Python 3.12 for Windows.
2. Clone or copy this project, then open PowerShell in the project directory.
3. Create a virtual environment if one does not already exist:

   ```powershell
   py -3.12 -m venv .venv
   ```

4. Install the pinned dependencies:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

PowerShell activation is optional. Calling `.\.venv\Scripts\python.exe`
directly keeps commands tied to the project environment.

Verify the installation without opening the webcam:

```powershell
.\.venv\Scripts\python.exe check_environment.py
```

The check verifies Python 3.12, the five required packages, the MediaPipe
Tasks API, and the local hand-landmarker file when it is present.

## MediaPipe model setup

The MediaPipe Python package does not include the Hand Landmarker task bundle.
The application expects this file:

```text
models\hand_landmarker.task
```

Check whether it exists:

```powershell
Test-Path .\models\hand_landmarker.task
```

If it is missing, download the official model bundle into `models/`:

```powershell
New-Item -ItemType Directory -Path .\models -Force
Invoke-WebRequest `
  -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" `
  -OutFile .\models\hand_landmarker.task
```

The current project already contains this file. Do not download it again unless
it is missing or invalid.

## Run the application

The current eight-gesture final model is:

```text
models\gesture_rf_v4_eight_gestures_session06.joblib
```

Run it from the project directory:

```powershell
.\.venv\Scripts\python.exe predict_webcam.py `
  --model .\models\gesture_rf_v4_eight_gestures_session06.joblib `
  --hand-model .\models\hand_landmarker.task
```

The application prints the requested and actual camera dimensions, model paths,
and periodic timing information. Press `q` or `Esc` while the OpenCV window is
focused to quit.

Useful inference options include:

```powershell
# Use another camera index.
.\.venv\Scripts\python.exe predict_webcam.py `
  --model .\models\gesture_rf_v4_eight_gestures_session06.joblib `
  --hand-model .\models\hand_landmarker.task `
  --camera-index 1

# Disable periodic timing output.
.\.venv\Scripts\python.exe predict_webcam.py `
  --model .\models\gesture_rf_v4_eight_gestures_session06.joblib `
  --hand-model .\models\hand_landmarker.task `
  --stats-interval 0

# Use a shorter prediction-smoothing window.
.\.venv\Scripts\python.exe predict_webcam.py `
  --model .\models\gesture_rf_v4_eight_gestures_session06.joblib `
  --hand-model .\models\hand_landmarker.task `
  --smoothing-window 5
```

The default command without `--model` remains the original five-class
`models\gesture_rf_baseline_v1.joblib`. Select the eight-class model explicitly
as shown above.

## Other useful commands

These commands are optional development or diagnostic workflows. They do not
run automatically as part of application startup.

### Environment verification

```powershell
.\.venv\Scripts\python.exe check_environment.py
```

### Landmark-only visualization

```powershell
.\.venv\Scripts\python.exe hand_landmarks.py `
  --model .\models\hand_landmarker.task
```

The detector can display up to two hands, but the classifier requires exactly
one detected hand for a gesture prediction.

### Camera-only test

```powershell
.\.venv\Scripts\python.exe webcam_test.py
```

The camera diagnostic requests 1280×720 by default. Use `--camera-index 1` if
camera index 0 is unavailable. Press `q` or `Esc` to stop.

### Automated tests

The tests do not open the webcam, collect data, or retrain the model:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

### Syntax checks

```powershell
.\.venv\Scripts\python.exe -m py_compile `
  check_environment.py `
  webcam_test.py `
  hand_landmarks.py `
  collect_data.py `
  gesture_config.py `
  train_model.py `
  prepare_dataset.py `
  train_session_model.py `
  predict_webcam.py `
  tests\test_collect_data.py `
  tests\test_train_model.py `
  tests\test_session_training.py `
  tests\test_predict_webcam.py
```

### Optional data collection

The collector stores normalized landmark features and metadata in a new CSV.
It does not save raw webcam images. A protected eight-gesture collection run
uses an explicit label list and a new output path:

```powershell
.\.venv\Scripts\python.exe collect_data.py `
  --gestures open_palm fist thumbs_up peace_sign ok_sign pointing_up three_fingers call_me `
  --samples-per-gesture 150 `
  --sample-interval 0.35 `
  --min-feature-distance 0.05 `
  --output .\data\landmarks_new_session.csv `
  --user-id user_01 `
  --session-id user01_new_session
```

Fresh mode refuses an existing output path. Use `--append-existing` only when
intentionally appending to a separately validated CSV.

### Optional training and audit workflows

The five-class baseline workflow is:

```powershell
.\.venv\Scripts\python.exe train_model.py --audit-only
.\.venv\Scripts\python.exe train_model.py
```

For the current eight-gesture session workflow, audit and combine sessions 04
and 05 without changing the source files:

```powershell
.\.venv\Scripts\python.exe prepare_dataset.py --eight-gestures --audit-only
.\.venv\Scripts\python.exe prepare_dataset.py --eight-gestures
```

The supplemental two-class session 06 workflow is deliberately separate:

```powershell
.\.venv\Scripts\python.exe train_session_model.py `
  --eight-gestures `
  --supplemental-data .\data\landmarks_fresh_session06_callme_thumbsup.csv `
  --audit-supplemental-only
```

Candidate comparison:

```powershell
.\.venv\Scripts\python.exe train_session_model.py `
  --eight-gestures `
  --train-data .\data\landmarks_fresh_session04.csv `
  --supplemental-data .\data\landmarks_fresh_session06_callme_thumbsup.csv `
  --test-data .\data\landmarks_fresh_session05.csv
```

Final-model training, which uses sessions 04, 05, and supplemental session 06:

```powershell
.\.venv\Scripts\python.exe train_session_model.py `
  --eight-gestures `
  --final-only `
  --final-data .\data\landmarks_combined_v2_eight_gestures.csv `
  --supplemental-data .\data\landmarks_fresh_session06_callme_thumbsup.csv
```

Training commands use versioned outputs and refuse to overwrite existing
artifacts. Do not run them merely to launch the application.

## How the system works

```text
Webcam frame
    ↓
MediaPipe Hand Landmarker
    ↓
21 hand landmarks per detected hand
    ↓
Wrist-centred, scale-normalized 63-value feature vector
    ↓
Random Forest prediction and probability estimate
    ↓
Short temporal majority smoothing
    ↓
Displayed gesture, confidence estimate, overlay, and FPS/timing data
```

The same normalization function is used by collection and realtime inference.
Metadata such as `label`, `session_id`, `user_id`, and `handedness` is validated
for datasets but is not passed as a classifier feature.

## Dataset and privacy

`collect_data.py` writes CSV rows containing:

- Gesture label.
- Session and user identifiers.
- Detected handedness.
- 63 normalized landmark coordinates.

The collector does not save raw webcam images and does not upload collected
frames. Local CSV files are project data and may contain user identifiers; do
not publish them without reviewing consent, privacy, and repository settings.
The project does not claim that every local dataset is public or included in a
public GitHub repository.

## Evaluation and limitations

### Current model

The current final model is:

```text
models\gesture_rf_v4_eight_gestures_session06.joblib
```

It was trained using sessions 04 and 05 plus the focused supplemental session
06. All available sessions belong to the same user, `user_01`. Because the
final model uses all available sessions, it has no independently verified final
model accuracy.

### Comparative diagnostic

A previous candidate was trained on session 04 plus supplemental session 06 and
evaluated on session 05:

| Metric | Measured result |
|---|---:|
| Accuracy | 96.50% |
| Macro F1 | 96.49% |
| `call_me` recall | 82.67% |

Session 05 had already been examined during development. This result is
therefore a comparative diagnostic, not independent final-model accuracy, and
must not be presented as the final model's independently verified performance.
The detailed report is:

```text
reports\metrics\gesture_rf_session04_plus_session06_eval_eight_v1.json
```

The final-model metadata report is:

```text
reports\metrics\gesture_rf_v4_eight_gestures_session06_final.json
```

Known limitations:

- Independent final-model accuracy is unverified.
- Performance on unseen users is unverified; all available sessions use `user_01`.
- The application expects exactly one detected hand for classification.
- A second visible hand produces `Show one hand only` rather than a gesture label.
- Displayed probabilities are classifier estimates, not guarantees of correctness.
- Live performance depends on camera resolution, lighting, hand position,
  orientation, background, and MediaPipe detection confidence.
- The application classifies eight static gestures; it does not translate
  continuous sign language or sentence-level signing.

## Troubleshooting

### `hand_landmarker.task` is missing

Place the official task bundle at:

```text
models\hand_landmarker.task
```

Use the [MediaPipe model setup](#mediapipe-model-setup) command, then rerun the
environment check.

### The camera cannot be opened

1. Close other applications using the webcam.
2. Check Windows camera privacy permissions.
3. Try another index:

   ```powershell
   .\.venv\Scripts\python.exe predict_webcam.py `
     --model .\models\gesture_rf_v4_eight_gestures_session06.joblib `
     --hand-model .\models\hand_landmarker.task `
     --camera-index 1
   ```

4. Test the device with `webcam_test.py` or the Windows Camera application.

### `No hand detected`

Improve lighting, keep the whole hand inside the frame, and move it closer to
the camera. The detector requires a usable hand with finite landmarks.

### `Show one hand only`

Move the second hand out of view. The detector may see up to two hands, but the
gesture classifier intentionally predicts only when exactly one hand is usable.

### The preview is small or the actual resolution is lower than requested

The application reports the requested and actual camera dimensions. Camera
drivers may return a lower resolution than requested. A larger preview does not
increase the camera's actual detail or recognition resolution.

### The prediction display flickers

The application uses a seven-prediction majority smoother by default. If the
display feels too delayed, try `--smoothing-window 5`; if it is unstable, a
larger window may be steadier. This is a live usability trade-off and is not
inferred from the saved evaluation score.

### The model is rejected before startup

Ensure that `--model` points to a Joblib bundle created by this project. The
inference script validates the saved label mapping, estimator classes, 63-value
feature count, and exact feature-column order before opening the webcam.

## Future improvements

- Collect representative sessions from additional users.
- Reserve a genuinely unseen user/session for final evaluation.
- Improve preview sizing and camera-resolution fallback handling.
- Evaluate confidence calibration and low-confidence rejection.
- Compare additional classifiers while preserving session-aware splits.
- Add a user interface suitable for a broader demonstration.

## License

No license is specified in this repository. Add a license file before
redistributing the project if you want to define reuse terms.
