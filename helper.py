import parselmouth
from parselmouth.praat import call
import numpy as np

# This is the function we will use for BOTH training and prediction.
def extract_voice_features(audio_file_path):
    """
    Extracts a set of acoustic features from an audio file using Parselmouth.
    
    We extract:
    1.  Mean, Min, Max Pitch (Fundamental Frequency Fo)
    2.  Jitter (local, local_abs, rap, ppq5, ddp) - 5 features
    3.  Shimmer (local, local_db, apq3, apq5, apq11, dda) - 6 features
    4.  Harmonicity (HNR)
    
    Total Features: 3 + 5 + 6 + 1 = 15 features
    """
    try:
        snd = parselmouth.Sound(audio_file_path)
        
        # Extract pitch (Fo)
        pitch = call(snd, "To Pitch", 0.0, 75, 600)
        mean_fo = call(pitch, "Get mean", 0, 0, "Hertz")
        min_fo = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")
        max_fo = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")

        # Extract jitter
        point_process = call(snd, "To PointProcess (periodic, cc)", 75, 600)
        jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        jitter_local_abs = call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
        jitter_rap = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        jitter_ppq5 = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
        jitter_ddp = call(point_process, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)

        # Extract shimmer - using the correct method for Praat-Parselmouth
        pulses = call(snd, "To PointProcess (periodic, cc)", 75, 600)
        shimmer_local = call([snd, pulses], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_local_db = call([snd, pulses], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_apq3 = call([snd, pulses], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_apq5 = call([snd, pulses], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_apq11 = call([snd, pulses], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_dda = call([snd, pulses], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

        # Extract HNR
        harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)

        # Compile all features into a list
        feature_list = [
            mean_fo, min_fo, max_fo,
            jitter_local, jitter_local_abs, jitter_rap, jitter_ppq5, jitter_ddp,
            shimmer_local, shimmer_local_db, shimmer_apq3, shimmer_apq5, shimmer_apq11, shimmer_dda,
            hnr
        ]
        
        # Ensure all are floats and return as a 2D numpy array for the scaler
        features = np.array([float(f) for f in feature_list]).reshape(1, -1)
        
        return features

    except Exception as e:
        print(f"Error extracting features from {audio_file_path}: {e}")
        return None