import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, lfilter
import os
import time

start_time = time.time()

# -------------------------------
# FOLDER PATHS
# -------------------------------
DATA_FOLDER = "D:\\FINAL_YEAR_RESEARCH\\AUDIO_AND_TRANSCRIPTS"            # Folder containing audio + transcripts
OUTPUT_FOLDER = "D:\\FINAL_YEAR_RESEARCH\\Cleaned_Audio" # Folder where cleaned audio will be saved

TARGET_SR = 16000

# Create output folder if it does not exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -------------------------------
# Pre-emphasis filter
# -------------------------------
def pre_emphasis(signal, alpha=0.97):
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])

# -------------------------------
# Bandpass filter
# -------------------------------
def bandpass_filter(data, sr, low=100, high=7500, order=5):
    nyquist = 0.5 * sr
    low = low / nyquist
    high = high / nyquist

    b, a = butter(order, [low, high], btype="band")
    return lfilter(b, a, data)

# -------------------------------
# Find all audio files
# -------------------------------
audio_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith("_AUDIO.wav")]

print("Total audio files found:", len(audio_files))

# -------------------------------
# Process each audio file
# -------------------------------
for file in audio_files:
    print("Processing:", file)

    audio_path = os.path.join(DATA_FOLDER, file)

    # Extract base name (example: 001 from 001_AUDIO.wav)
    base_name = file.replace("_AUDIO.wav", "")

    transcript_path = os.path.join(DATA_FOLDER, f"{base_name}_TRANSCRIPT.csv")
    output_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_final_clean.wav")

    # -------------------------------
    # Load audio
    # -------------------------------
    audio, sr = librosa.load(audio_path, sr=None)
    audio = librosa.to_mono(audio)

    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    # -------------------------------
    # Load transcript
    # -------------------------------
    df = pd.read_csv(transcript_path, sep="\t")

    segments = []

    for row in df.itertuples():
        if row.speaker == "Participant":
            start_sample = int(row.start_time * sr)
            end_sample = int(row.stop_time * sr)
            segments.append(audio[start_sample:end_sample])

    if len(segments) == 0:
        print("No participant speech found in", file)
        continue

    participant_audio = np.concatenate(segments)

    # -------------------------------
    # Pre-emphasis
    # -------------------------------
    participant_audio = pre_emphasis(participant_audio)

    # -------------------------------
    # Bandpass filter
    # -------------------------------
    clean_audio = bandpass_filter(participant_audio, sr)

    # -------------------------------
    # Normalize
    # -------------------------------
    max_val = np.max(np.abs(clean_audio))
    if max_val > 0:
        clean_audio = clean_audio / max_val

    # -------------------------------
    # Save output
    # -------------------------------
    sf.write(output_path, clean_audio, sr)

    print("Saved:", output_path)

# -------------------------------
# Total processing time
# -------------------------------
end_time = time.time()

print("Processing completed")
print("Total time:", round(end_time - start_time, 2), "seconds")