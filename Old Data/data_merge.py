import pandas as pd
import os

# =========================
# 1️⃣ Get Correct Base Path
# =========================

# Current working directory
current_dir = os.getcwd()
print("Current Working Directory:", current_dir)

# If running from Training/, go one level up
if os.path.basename(current_dir) == "Training":
    BASE_DIR = os.path.dirname(current_dir)
else:
    BASE_DIR = current_dir

print("Base Directory:", BASE_DIR)

# =========================
# 2️⃣ Build File Paths
# =========================
audio_path = os.path.join(BASE_DIR, "data", "audio_features_full.csv")
text_path = os.path.join(BASE_DIR, "data", "text_features_full.csv")

print("Audio Path:", audio_path)
print("Text Path:", text_path)

# =========================
# 3️⃣ Load Datasets
# =========================
audio_df = pd.read_csv(audio_path)
text_df = pd.read_csv(text_path)

print("Audio shape:", audio_df.shape)
print("Text shape:", text_df.shape)

# =========================
# 4️⃣ Merge on participant_id
# =========================
merged_df = pd.merge(
    audio_df,
    text_df,
    on="participant_id",
    how="inner",
    suffixes=("_audio", "_text")
)

# =========================
# 5️⃣ Handle Duplicate Labels
# =========================
if "label_audio" in merged_df.columns and "label_text" in merged_df.columns:
    
    mismatch = (merged_df["label_audio"] != merged_df["label_text"]).sum()
    print("Label mismatches:", mismatch)

    merged_df["label"] = merged_df["label_audio"]
    merged_df = merged_df.drop(columns=["label_audio", "label_text"])

# =========================
# 6️⃣ Save Final Dataset
# =========================
output_path = os.path.join(BASE_DIR, "data", "multimodal_features.csv")
merged_df.to_csv(output_path, index=False)

print("Merged dataset saved at:", output_path)
print("Final shape:", merged_df.shape)

# =========================
# 7️⃣ Quick Check
# =========================
print("\nPreview:")
print(merged_df.head())