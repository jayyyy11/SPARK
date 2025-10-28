# Parkinson's Voice Detector

This repository contains a small project to extract voice features from audio recordings and train machine learning models to detect Parkinson's disease from voice data. It includes: feature extraction using Praat via the `praat-parselmouth` package, model training (`build_model.py`), and a Flask web app (`app.py`) to upload and analyze new audio samples.

## Repository structure

- `build_model.py`  – script to extract features from WAV files, train multiple models, run grid search, and save the best model and scaler (`best_model.pkl`, `scaler.pkl`).
- `helper.py`       – feature extraction utilities using `praat-parselmouth`.
- `app.py`          – Flask web application for uploading audio and running predictions.
- `Figshare_Audio_Data/PD` – Parkinson's audio samples (WAV).
- `Figshare_Audio_Data/HC` – Healthy control audio samples (WAV).
- `voice_features_dataset.csv` – generated features CSV (produced by `build_model.py`).
- `requirements.txt` – Python dependencies.
- `best_model.pkl`, `scaler.pkl` – saved model + scaler (created after training).

## Goals and expected results

- Extract acoustic features (pitch, jitter, shimmer, HNR, etc.) from WAV files.
- Train and compare multiple models (SVM, RandomForest, XGBoost, GradientBoosting, etc.).
- Save the best performing model and use it in the web app to classify uploaded audio.

## Quick setup (Windows PowerShell)

Open PowerShell in the project root (`c:\Users\jahna\OneDrive\Desktop\parkinsons_detector`). These commands assume you want a local virtual environment named `venv`.

1) Create and activate a virtual environment

```powershell
python -m venv venv
# Activate (PowerShell)
.\venv\Scripts\Activate.ps1
# If you prefer cmd.exe:
# .\venv\Scripts\activate.bat
```

2) Upgrade pip and install requirements

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Notes:
- `requirements.txt` includes `praat-parselmouth` which exposes Praat functionality via Python. On Windows this generally works via the package, but if you encounter Praat-related errors you may need to install the standalone Praat binary (see Troubleshooting below).

## How to run (training)

To extract features and train models (this will generate `voice_features_dataset.csv`, `best_model.pkl`, and `scaler.pkl`):

```powershell
# (venv should be activated)
python build_model.py
```

The script will:
- Search `Figshare_Audio_Data/PD/*.wav` and `Figshare_Audio_Data/HC/*.wav` for audio files
- Extract features using `helper.extract_voice_features`
- Perform grid search + cross-validation for several models
- Save the best model and scaler to the project root

If you want to re-run training and try to improve accuracy, you can edit `build_model.py` to adjust the model search spaces, add data augmentation, or provide more samples.

## How to run (Flask web app)

The web app (`app.py`) serves a simple UI for uploading an audio file and seeing a prediction. Ensure the `best_model.pkl` and `scaler.pkl` are present.

```powershell
# Set environment variables (PowerShell)
$env:FLASK_APP = "app.py"
$env:FLASK_ENV = "development"
# Run the flask app
python -m flask run --host=127.0.0.1 --port=5000
```

Then open http://127.0.0.1:5000 in your browser.

## Prediction from script (optional)

You can also create a small script that loads `best_model.pkl` and `scaler.pkl` and calls `helper.extract_voice_features(path)` then `model.predict()` on the scaled features. Example snippet (informational):

```python
import joblib
from helper import extract_voice_features

model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

features = extract_voice_features('path/to/file.wav')
X = scaler.transform([features.flatten()])
print(model.predict(X))
```

## Troubleshooting

- "No audio files were processed":
  - Confirm WAV files exist under `Figshare_Audio_Data/PD` and `Figshare_Audio_Data/HC`.
  - Check the `AUDIO_DATA_PATH`, `PD_FOLDER_NAME`, and `HC_FOLDER_NAME` values in `build_model.py`.

- Praat/Parselmouth errors (e.g. shimmer or pitch commands not available):
  - Ensure `praat-parselmouth` is installed in the active venv.
  - Some Praat commands require a valid PointProcess/pulses object; the project `helper.py` was updated to use a `pulses` object for shimmer calculations. If you still see errors for certain audio files, those files may be too short or silent—check the file or filter out very short audio.
  - If you see errors mentioning Praat itself, install the Praat binary from https://www.fon.hum.uva.nl/praat/ and ensure it is on your PATH.

- If pip fails building a dependency (old packages):
  - Upgrade pip first (`python -m pip install --upgrade pip`) then reinstall requirements.
  - If a package still fails, try installing it individually to read the error and search for platform-specific fixes.

## Improving accuracy (ideas)

- Add more labeled audio data.
- Clean or filter out noisy / too-short recordings.
- Try feature selection or additional acoustic features.
- Try data augmentation (speed, pitch shift, noise) to increase dataset size.
- Use more robust CV or nested CV and try more model types/hyperparameters.
- Consider ensembling top models.

## Files saved by training

- `voice_features_dataset.csv` – CSV of extracted features.
- `best_model.pkl` – Saved trained sklearn pipeline (scaler + classifier).
- `scaler.pkl` – Saved scaler (if not included in pipeline).

## Licensing & attribution

This project uses publicly available audio datasets and open source packages. Include your preferred license file if you plan to publish (e.g., `MIT` or `Apache-2.0`).

## Next steps

- Add README to the repository and push to GitHub.
- Optionally add a small `requirements-dev.txt` for testing or CI.
- Add tests for `helper.extract_voice_features` on a few known WAV files.

If you want, I can also:
- Create a minimal `predict.py` script for CLI predictions.
- Add a GitHub Actions workflow to run lint/tests and optionally retrain on push.

---

