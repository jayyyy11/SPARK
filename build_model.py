import os
import glob
import pandas as pd
import numpy as np
import joblib
from helper import extract_voice_features  # <-- Importing our helper

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# --- Part 1: Build Our Dataset ---
print("--- Part 1: Building dataset from .wav files ---")

# --- SETTINGS: Change these to match your folder names ---
AUDIO_DATA_PATH = "Figshare_Audio_Data/"
PD_FOLDER_NAME = "PD"  # The name of your folder with Parkinson's audio
HC_FOLDER_NAME = "HC"  # The name of your folder with Healthy Control audio
# --------------------------------------------------------

feature_data = []

# 1. Process all Parkinson's (PD) files
pd_path = os.path.join(AUDIO_DATA_PATH, PD_FOLDER_NAME, "*.wav")
pd_files = glob.glob(pd_path)
print(f"Found {len(pd_files)} files in '{PD_FOLDER_NAME}' folder...")

for filepath in pd_files:
    label = 1  # 1 = Parkinson's
    features = extract_voice_features(filepath)
    if features is not None:
        feature_row = list(features.flatten())
        feature_row.append(label)
        feature_data.append(feature_row)

# 2. Process all Healthy Control (HC) files
hc_path = os.path.join(AUDIO_DATA_PATH, HC_FOLDER_NAME, "*.wav")
hc_files = glob.glob(hc_path)
print(f"Found {len(hc_files)} files in '{HC_FOLDER_NAME}' folder...")

for filepath in hc_files:
    label = 0  # 0 = Healthy
    features = extract_voice_features(filepath)
    if features is not None:
        feature_row = list(features.flatten())
        feature_row.append(label)
        feature_data.append(feature_row)

# 3. Check if we found data and create DataFrame
if len(feature_data) == 0:
    print("\n!!! ERROR: No audio files were processed. !!!")
    print("Please check your folder paths. I looked for:")
    print(f"1. {pd_path}")
    print(f"2. {hc_path}")
    print("Aborting script. Please fix the paths in the 'SETTINGS' section.")
else:
    # --- Create a DataFrame ---
    print(f"\nSuccessfully processed {len(feature_data)} total audio files.")
    num_features = len(feature_data[0]) - 1 # Subtract 1 for the label column
    feature_names = [f"feature_{i+1}" for i in range(num_features)]
    column_names = feature_names + ["label"]

    df = pd.DataFrame(feature_data, columns=column_names)

    # Save the dataset to a CSV
    df.to_csv("voice_features_dataset.csv", index=False)
    print(f"Successfully built and saved 'voice_features_dataset.csv'.")


    # --- Part 2: Compare Models & Save the Best One ---
    print("\n--- Part 2: Training and comparing models ---")

    # 1. Load Data
    data = pd.read_csv('voice_features_dataset.csv')
    X = data.drop('label', axis=1).values
    y = data['label']

    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Scale Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Define Models
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(probability=True),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    best_model = None
    best_accuracy = 0.0

    # 5. Compare Models
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{name}: Accuracy = {accuracy:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    print("------------------------")
    print(f"Best Model: {type(best_model).__name__} with Accuracy: {best_accuracy:.4f}")

    # 6. Save the Best Model and Scaler
    joblib.dump(best_model, 'best_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    print("Best model and scaler saved as 'best_model.pkl' and 'scaler.pkl'")