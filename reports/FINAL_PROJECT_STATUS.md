# Final Project Status

**Project:** Real-Time Hand Sign Recognition System  
**Workspace:** `C:\Users\Lenovo\hand-sign-recognition`  
**Status:** Release-ready for focused manual use, with documented limitations  
**Scope of this pass:** Release-readiness verification only; no webcam was
opened, no data was collected, and no model was retrained.

## 1. Architecture

The application has four main stages:

1. `collect_data.py` uses MediaPipe Hand Landmarker to detect exactly one hand
   and stores 21 wrist-centred, scale-normalized landmarks plus metadata in a
   67-column CSV.
2. `train_model.py`, `prepare_dataset.py`, and `train_session_model.py` validate
   CSV contracts and train versioned scikit-learn Random Forest bundles using
   the 63 landmark features. Metadata is audited but excluded from the model.
3. `predict_webcam.py` loads a saved Joblib bundle, validates its actual label
   mapping and 63-feature order, detects a hand in the webcam stream, and
   predicts the loaded model's labels.
4. `PredictionSmoother` uses a short majority window to reduce display flicker.
   The live application requires exactly one detected hand for a prediction.

The existing five-class default remains supported. The current eight-gesture
final model is selected explicitly through `--model`.

## 2. Supported labels

The configured eight labels are:

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

The labels are defined in `gesture_config.py`. The final model bundle stores
the same eight-label mapping, and `predict_webcam.py` validates the loaded
bundle's actual mapping instead of assuming only the historical five labels.

## 3. Data files

### Original eight-gesture sessions

```text
data\landmarks_fresh_session04.csv
data\landmarks_fresh_session05.csv
```

Each contains 1,200 validated rows: 150 per gesture, with the canonical
67-column schema and 63 numeric features.

### Combined eight-gesture dataset

```text
data\landmarks_combined_v2_eight_gestures.csv
```

Contains 2,400 rows from sessions 04 and 05, with 300 rows per gesture.

### Focused supplemental dataset

```text
data\landmarks_fresh_session06_callme_thumbsup.csv
```

Contains 300 validated rows:

```text
call_me:    150
thumbs_up:  150
```

The supplemental source contains both left- and right-handed samples for both
focused labels. It remains a separate source file and was not used to modify
the original CSVs.

## 4. Model and report files

### Current eight-gesture final model

```text
models\gesture_rf_v4_eight_gestures_session06.joblib
reports\metrics\gesture_rf_v4_eight_gestures_session06_final.json
```

The final model was trained using sessions 04, 05, and 06. It has the eight
configured labels, 63 features, and probability support. It does **not** have
an independently verified final-model accuracy because all available sessions
were used for final training.

### Comparative candidate

```text
models\gesture_rf_session04_plus_session06_eval_eight_v1.joblib
reports\metrics\gesture_rf_session04_plus_session06_eval_eight_v1.json
reports\metrics\gesture_rf_session04_plus_session06_eval_eight_v1_confusion_matrix.csv
```

The candidate was trained on session 04 plus the focused supplemental session
06 and evaluated on the full session 05. Session 05 was already examined
during earlier development, so this is a comparative diagnostic rather than a
pristine final test.

Earlier model artifacts, including `gesture_rf_v3_eight_gestures.joblib` and
the five-class default, are preserved.

## 5. Measured candidate evaluation

The candidate evaluation used:

```text
Training: session 04 + supplemental session 06
Testing:  session 05
Training rows: 1500
Test rows:     1200
```

Measured candidate metrics:

| Metric | Candidate |
|---|---:|
| Accuracy | 96.50% |
| Macro precision | 96.78% |
| Macro recall | 96.50% |
| Macro F1 | 96.49% |
| `call_me` precision | 97.64% |
| `call_me` recall | 82.67% |
| `call_me` F1 | 89.53% |
| `thumbs_up` precision | 85.23% |
| `thumbs_up` recall | 100.00% |
| `thumbs_up` F1 | 92.02% |

For the focused confusion:

```text
Actual call_me predicted as call_me:    124/150
Actual call_me predicted as thumbs_up:   26/150
Actual thumbs_up predicted as thumbs_up: 150/150
Actual thumbs_up predicted as call_me:     0/150
```

These are measured session-05 comparative results. They are not a guarantee
of performance for new users, future sessions, or different cameras and
lighting conditions.

## 6. Manual webcam observations

The latest manual test reported:

```text
Camera opened: yes
All eight gestures tested: yes
Gestures recognized incorrectly: no
Prediction stability: flickering
Approximate FPS: 20
Other observation: mostly recognizing the collector's hand, not other hands
```

Interpretation:

- Camera startup and an eight-gesture smoke test succeeded according to the
  supplied observation.
- The approximately 20 FPS value is an informal manual observation, not a
  recorded benchmark from the timing logger.
- Flickering is a real usability observation, but it is not by itself evidence
  of a classifier or data defect. The application already applies a seven-frame
  majority smoother; lighting, hand movement, detection confidence, and
  borderline frame predictions can still affect the display.
- Recognition of the collector's hand more readily than other hands is
  consistent with the available data: all training users are `user_01`.
  It is a generalization limitation, not proof of a code defect. No unseen-user
  evaluation has been performed.
- The application intentionally requires exactly one detected hand. Multiple
  hands produce a `Show one hand only` status rather than a prediction.

No retraining or inference change was justified by these observations alone.

## 7. Known limitations

- All available training and evaluation sessions belong to `user_01`.
- The final model has no independent test accuracy claim.
- The 96.50% candidate result is a comparative diagnostic on session 05,
  which was examined earlier during development.
- Performance for unseen users is unverified.
- The live application expects one visible hand at a time.
- Display flicker may remain in borderline frames despite smoothing.
- The reported 20 FPS was not collected as a controlled benchmark.
- Live behavior can change with lighting, camera placement, hand orientation,
  distance, and MediaPipe detection confidence.

## 8. Verified checks

The complete automated suite was run without opening the webcam or collecting
data:

```powershell
cd C:\Users\Lenovo\hand-sign-recognition
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Result:

```text
Ran 31 tests
OK
```

Additional release checks passed:

- The current final model loads successfully.
- The final model exposes all eight configured labels.
- The final model uses the expected 63-feature order.
- Probability prediction support is available.
- The candidate model exposes the same eight-label mapping and feature count.
- Python syntax compilation passed for the changed training, inference,
  preparation, and test files.

## 9. Startup instructions

From PowerShell:

```powershell
cd C:\Users\Lenovo\hand-sign-recognition

.\.venv\Scripts\python.exe predict_webcam.py `
  --model .\models\gesture_rf_v4_eight_gestures_session06.joblib `
  --hand-model .\models\hand_landmarker.task
```

Press `q` or `Esc` while the OpenCV window is focused to stop.

Optional troubleshooting variants:

```powershell
# Try another camera index.
.\.venv\Scripts\python.exe predict_webcam.py `
  --model .\models\gesture_rf_v4_eight_gestures_session06.joblib `
  --hand-model .\models\hand_landmarker.task `
  --camera-index 1

# Disable timing output.
.\.venv\Scripts\python.exe predict_webcam.py `
  --model .\models\gesture_rf_v4_eight_gestures_session06.joblib `
  --hand-model .\models\hand_landmarker.task `
  --stats-interval 0

# Use a shorter smoothing window if the display feels too delayed.
.\.venv\Scripts\python.exe predict_webcam.py `
  --model .\models\gesture_rf_v4_eight_gestures_session06.joblib `
  --hand-model .\models\hand_landmarker.task `
  --smoothing-window 5
```

If the hand is not detected, improve lighting and keep the entire hand inside
the frame. If the status says `Show one hand only`, move the second hand out of
the camera view. If another person's hand is not recognized reliably, treat it
as an unverified unseen-user limitation rather than changing the final model
without new representative data.

## 10. Release conclusion

The project passes the focused release-readiness checks for the current
eight-gesture final artifact. The documented startup command now selects
`gesture_rf_v4_eight_gestures_session06.joblib`. The measured candidate result
supports the supplemental-data change for the `call_me`/`thumbs_up` confusion,
while the final model's independent accuracy and unseen-user performance remain
unverified.
