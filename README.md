# 🧠 Multimodal Depression Detection from Speech and Linguistic Analysis Across Conversational Phases

> Depression Detection using Multimodal Audio & Text Analysis — A Final Year Research Project (FYRP)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Folder Structure](#-folder-structure)
3. [Prerequisites & Installation](#-prerequisites--installation)
4. [Dataset Requirements](#-dataset-requirements)
5. [Step 1 – Audio Preprocessing](#step-1--audio-preprocessing-cleaned_audiopy)
6. [Step 2 – Transcript Preprocessing](#step-2--transcript-preprocessing-transcript_preprocessingipynb)
7. [Step 3 – Audio Feature Extraction](#step-3--audio-feature-extraction-audio_feature_extractionipynb)
8. [Step 4 – Text Feature Extraction](#step-4--text-feature-extraction-text_feature_extractionipynb)
9. [Step 5 – Non-Phasewise Model Training](#step-5--non-phasewise-model-training)
10. [Step 6 – Phasewise Model Training](#step-6--phasewise-model-training)
11. [Data Files Reference](#-data-files-reference)
12. [Troubleshooting & Notes](#-troubleshooting--notes)
13. [Execution Pipeline Summary](#-execution-pipeline-summary)

---

## 🔬 Project Overview

This Final Year Research Project (FYRP) focuses on **detecting depression from clinical interview data** using a multimodal machine learning approach. The system analyzes both **audio recordings** and **text transcripts** from the **DAIC-WOZ dataset** to classify participants as *depressed* or *non-depressed* based on the **PHQ-8** (Patient Health Questionnaire-8) binary score.

### Pipeline Stages

| Stage | Description |
|-------|-------------|
| **1. Audio Preprocessing** | Cleaning raw audio, isolating participant speech, filtering, and normalizing |
| **2. Transcript Preprocessing** | Cleaning text transcripts by removing tags, filler words, and normalizing |
| **3. Feature Extraction** | Extracting audio features (MFCCs, pitch, jitter, shimmer, etc.) and text features (SBERT embeddings, sentiment, linguistic markers) |
| **4. Model Training & Evaluation** | Training Random Forest, XGBoost, SVM, and CatBoost classifiers on audio-only, text-only, and multimodal features with phasewise and non-phasewise approaches |

### Dataset Statistics

- **Total interview recordings:** 188
- **Participants with valid PHQ-8 labels:** 141
- **Class distribution:** 99 non-depressed, 42 depressed

---

## 📁 Folder Structure

```
mental-health-fyrp/
├── audio_text_processing/
│   ├── cleaned_audio.py
│   ├── Transcript_preprocessing.ipynb
│   ├── audio_feature_extraction.ipynb
│   └── text_feature_extraction.ipynb
├── data/
│   ├── audio_features_phasewise.csv
│   ├── hubert_phasewise_embeddings.csv
│   ├── merged_audio_hubert_features.csv
│   ├── multimodal_audio_text_features.csv
│   └── text_features_phasewise.csv
├── training_models/
│   ├── non-phasewise training/
│   │   ├── audio_only.ipynb
│   │   ├── multimodal.ipynb
│   │   └── text_only.ipynb
│   └── phasewise training/
│       ├── audio_only_phasewise.ipynb
│       ├── multimodal_phasewise.ipynb
│       └── text_only_phasewise.ipynb
├── FRP Documents/
│   ├── Audio Preprocessing Pipeline.docx
│   ├── Project Proposal Form.docx
│   ├── manuscript_fyrp.docx
│   └── transcript_preprocessing_documentation.docx
├── Graphs/                    # Visualization outputs
├── requirements.txt
├── README.md
└── LICENSE
```

### Folder Descriptions

| Folder | Description |
|--------|-------------|
| `audio_text_processing/` | Scripts and notebooks for audio/transcript preprocessing and feature extraction |
| `data/` | Extracted feature CSV files used for model training |
| `training_models/non-phasewise training/` | Model training notebooks using full-interview (non-phasewise) features |
| `training_models/phasewise training/` | Model training notebooks using interview-phase-segmented features |
| `FRP Documents/` | Project documentation, proposal, and manuscript |
| `Graphs/` | Visualization outputs |

---

## ⚙️ Prerequisites & Installation

### System Requirements

- **Python** 3.10 or higher (tested with Python 3.12.1)
- **Jupyter Notebook** or JupyterLab
- **Google Colab** account (optional, needed for transcript preprocessing)
- Minimum **8 GB RAM** recommended
- **GPU** recommended for faster SBERT embedding generation (not required)

### Installation Steps

**Step 1: Clone the repository**

```bash
git clone https://github.com/<username>/mental-health-fyrp.git
cd mental-health-fyrp
```

**Step 2: Create a virtual environment (recommended)**

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

**Step 3: Install all dependencies**

```bash
pip install -r requirements.txt
```

**Step 4: Download NLTK data**

```bash
python -c "import nltk; nltk.download('vader_lexicon')"
```

**Step 5: Verify installation**

```bash
python -c "import librosa, parselmouth, sklearn, xgboost, sentence_transformers; print('All packages installed successfully!')"
```

### Required Python Packages

| Package | Purpose | Used In |
|---------|---------|---------|
| `numpy` | Numerical computing | All files |
| `pandas` | Data manipulation & CSV handling | All files |
| `librosa` | Audio loading, resampling, feature extraction | `cleaned_audio.py`, `audio_feature_extraction.ipynb` |
| `soundfile` | Audio file writing (WAV output) | `cleaned_audio.py` |
| `scipy` | Signal processing (Butterworth filter) | `cleaned_audio.py` |
| `parselmouth` | Praat-based pitch, jitter & shimmer analysis | `audio_feature_extraction.ipynb` |
| `tqdm` | Progress bar for loops | Multiple notebooks |
| `nltk` | VADER sentiment analysis | `text_feature_extraction.ipynb` |
| `sentence-transformers` | SBERT text embeddings (`all-MiniLM-L6-v2`) | `text_feature_extraction.ipynb` |
| `scikit-learn` | ML models, metrics, preprocessing | All training notebooks |
| `xgboost` | XGBoost classifier | All training notebooks |
| `catboost` | CatBoost classifier | Phasewise training notebooks |
| `jupyter` | Running `.ipynb` notebooks | All notebooks |

---

## 📊 Dataset Requirements

This project uses the **DAIC-WOZ Depression Database**. You will need the following raw data files before running the preprocessing scripts:

| File | Format | Description |
|------|--------|-------------|
| `{participant_id}_AUDIO.wav` | WAV | Raw clinical interview audio recordings |
| `{participant_id}_TRANSCRIPT.csv` | Tab-separated CSV | Transcript files with columns: `speaker`, `start_time`, `stop_time`, `value` |
| `merged_labels.csv` | CSV | Contains columns: `Participant_ID`, `PHQ8_Binary` (0 = not depressed, 1 = depressed) |

> ⚠️ **Note:** The DAIC-WOZ dataset must be obtained separately from the **USC Institute for Creative Technologies**. It is not included in this repository due to licensing restrictions.

---

## Step 1 – Audio Preprocessing (`cleaned_audio.py`)

**File:** `audio_text_processing/cleaned_audio.py`

This Python script preprocesses raw audio files by extracting only the participant's speech segments, applying signal processing filters, and producing cleaned WAV files.

### How to Run

1. **Edit the folder paths** in the script:

```python
DATA_FOLDER = "D:\\FINAL_YEAR_RESEARCH\\AUDIO_AND_TRANSCRIPTS"   # Input folder
OUTPUT_FOLDER = "D:\\FINAL_YEAR_RESEARCH\\Cleaned_Audio"           # Output folder
```

2. **Run the script:**

```bash
python audio_text_processing/cleaned_audio.py
```

### What It Does (Step-by-Step)

1. Scans the input folder for all files ending with `_AUDIO.wav`
2. For each audio file, loads it using `librosa` and converts to **mono**
3. **Resamples** the audio to **16 kHz** (target sample rate)
4. Loads the corresponding transcript CSV file (tab-separated) with matching base name (e.g., `001_TRANSCRIPT.csv` for `001_AUDIO.wav`)
5. Filters the transcript to extract only **"Participant"** speaker segments using `start_time` and `stop_time` columns
6. **Concatenates** all participant audio segments (removing interviewer speech)
7. Applies a **pre-emphasis filter** (α = 0.97) to boost high-frequency components
8. Applies a **Butterworth bandpass filter** (100 Hz – 7500 Hz, order 5) to remove noise outside the speech frequency range
9. **Normalizes** the audio amplitude by dividing by the maximum absolute value
10. Saves the cleaned audio as `{id}_final_clean.wav` in the output folder

### I/O Files

| Direction | File | Description |
|-----------|------|-------------|
| **Input** | `{id}_AUDIO.wav` | Raw audio recordings |
| **Input** | `{id}_TRANSCRIPT.csv` | Tab-separated transcripts |
| **Output** | `{id}_final_clean.wav` | Cleaned participant-only audio files |

---

## Step 2 – Transcript Preprocessing (`Transcript_preprocessing.ipynb`)

**File:** `audio_text_processing/Transcript_preprocessing (2).ipynb`

This Jupyter notebook cleans raw transcript CSV files by filtering to participant-only speech and removing noise from the text. Originally designed to run on **Google Colab**.

### How to Run

**Option A – Google Colab (Original):**

1. Upload the notebook to Google Colab
2. Upload your transcript files to Google Drive at: `/content/drive/MyDrive/audio and transcript/`
3. Run all cells. The notebook will mount Google Drive automatically.

**Option B – Local Jupyter:**

1. Change the folder paths from Google Drive paths to local paths
2. Remove or comment out the `google.colab` import and `drive.mount()` line
3. Run the notebook:

```bash
jupyter notebook "audio_text_processing/Transcript_preprocessing (2).ipynb"
```

### What It Does (Step-by-Step)

1. Mounts Google Drive (Colab only)
2. Sets input folder (raw transcripts) and output folder (cleaned transcripts)
3. Defines text cleaning functions:
   - `remove_tags()` – removes `<tag>` and `[tag]` patterns
   - `extract_parentheses()` – extracts text from inside parentheses
   - `remove_fillers()` – removes filler words (`uh`, `um`, `mm`, `hmm`, `erm`, `ah`, `uhh`, `umm`, `mmm`)
   - `normalize_text()` – collapses whitespace
4. Lists all files containing "transcript" in the input folder (188 files found)
5. For each transcript file: reads the CSV (tab-separated), filters to "Participant" rows only, applies the cleaning pipeline to the `value` column, removes empty rows, and saves as a cleaned CSV

### I/O Files

| Direction | File | Description |
|-----------|------|-------------|
| **Input** | `{id}_TRANSCRIPT.csv` | Raw tab-separated transcripts |
| **Output** | `{id}_TRANSCRIPT_cleaned.csv` | Cleaned comma-separated transcripts (188 files) |

---

## Step 3 – Audio Feature Extraction (`audio_feature_extraction.ipynb`)

**File:** `audio_text_processing/audio_feature_extraction.ipynb`

This notebook extracts a comprehensive set of audio/acoustic features from the cleaned audio WAV files for use in depression detection models.

### How to Run

1. **Ensure the following folders/files exist:**
   - `Cleaned_Audio/` — folder with cleaned WAV files from Step 1
   - `cleaned_transcripts/` — folder with cleaned transcripts from Step 2
   - `merged_labels.csv` — participant labels file

2. **Open and run the notebook:**

```bash
jupyter notebook audio_text_processing/audio_feature_extraction.ipynb
```

### What It Does (Step-by-Step)

1. Sets paths for audio folder, transcript folder, labels file, and output CSV
2. Defines an articulation speed function that reads cleaned transcripts, counts words, and divides by speech duration
3. Defines the main feature extraction function that computes for each audio file:
   - **Speech/Silence analysis** — using `librosa.effects.split(top_db=30)`
   - **RMS Energy** — mean and standard deviation
   - **Zero-Crossing Rate (ZCR)** — mean and standard deviation
   - **13 MFCCs** — mean and standard deviation for each (26 features total)
   - **Spectral features** — spectral centroid, bandwidth, rolloff (mean & std)
   - **Pitch (F0)** — via `librosa.pyin()` in C2–C7 range
   - **Speaking rate** — number of speech intervals / total duration
   - **Pause duration** — mean gap between consecutive speech intervals
   - **Jitter & Shimmer** — via `parselmouth` (Praat)
4. Loads `merged_labels.csv` to filter to 141 valid participants
5. Processes all 188 WAV files, skipping those without valid labels
6. Combines all features into a DataFrame and saves to `features_full.csv`

### Features Extracted (~44+ features)

| Feature Group | Features | Count |
|---------------|----------|-------|
| Speech/Silence | `speech_duration`, `silence_duration`, `speech_to_silence_ratio`, `silence_ratio` | 4 |
| RMS Energy | `rms_mean`, `rms_std`, `energy_mean`, `energy_std` | 4 |
| Zero-Crossing Rate | `zcr_mean`, `zcr_std` | 2 |
| MFCCs | `mfcc_1_mean` through `mfcc_13_mean`, `mfcc_1_std` through `mfcc_13_std` | 26 |
| Spectral | `spectral_centroid_mean/std`, `spectral_bandwidth_mean/std`, `spectral_rolloff_mean/std` | 6 |
| Pitch (F0) | `pitch_mean`, `pitch_std` | 2 |
| Speaking Rate | `speaking_rate` | 1 |
| Pause Duration | `pause_duration` | 1 |
| Jitter | `jitter` | 1 |
| Shimmer | `shimmer` | 1 |
| Articulation | `articulation_speed`, `word_count` | 2 |

### I/O Files

| Direction | File | Description |
|-----------|------|-------------|
| **Input** | `Cleaned_Audio/{id}_final_clean.wav` | Cleaned audio files |
| **Input** | `cleaned_transcripts/{id}_TRANSCRIPT_cleaned.csv` | Cleaned transcripts |
| **Input** | `merged_labels.csv` | Participant labels |
| **Output** | `features_full.csv` | All audio features with participant IDs and labels |

---

## Step 4 – Text Feature Extraction (`text_feature_extraction.ipynb`)

**File:** `audio_text_processing/text_feature_extraction.ipynb`

This notebook extracts NLP/text features from cleaned transcripts. It contains **two versions** of the extraction pipeline.

### How to Run

1. **Ensure the following files exist:**
   - `cleaned_transcripts/` — folder with cleaned transcript CSVs
   - `merged_labels.csv` — participant labels file

2. **Run the notebook:**

```bash
jupyter notebook audio_text_processing/text_feature_extraction.ipynb
```

### Version 1 (Cell 1) → `text_features2.csv`

- Uses **VADER sentiment analysis** + **SBERT embeddings**
- Downloads VADER lexicon via `nltk`
- Loads SBERT model (`all-MiniLM-L6-v2`, 384 dimensions)
- Defines word lists: 28 pronouns, 4 negation words, 6 uncertainty words
- For each transcript: concatenates all text, splits into words/sentences
- Computes: `word_count`, `sentence_count`, `avg_sentence_length`, `pronoun_ratio`, `negation_count`, `uncertainty_count`, `type_token_ratio`
- Runs VADER sentiment → `positive`, `negative`, `neutral`, `compound` scores
- Encodes full text with SBERT → 384 embedding dimensions
- Saves all features to `text_features2.csv`

### Version 2 (Cell 2) → `text_features_updated.csv`

- Updated version **without VADER**, adds **pause-based features** and **per-sentence SBERT embeddings**
- Loads SBERT model (`all-MiniLM-L6-v2`)
- Uses reduced first-person pronoun list (5 words: `i`, `me`, `my`, `mine`, `myself`)
- For each transcript DataFrame: filters sentences with >3 words
- Computes linguistic features: `word_count`, `sentence_count`, `avg_sentence_length`, `first_person_ratio`, `negation_count`, `uncertainty_count`, `type_token_ratio`
- Computes pause features from `start_time`/`stop_time` columns → `mean_pause`, `max_pause`
- Encodes each sentence individually with SBERT, then computes mean and std across all sentence embeddings (384×2 = 768 SBERT features)
- Saves to `text_features_updated.csv`

### I/O Files

| Direction | File | Description |
|-----------|------|-------------|
| **Input** | `cleaned_transcripts/{id}_TRANSCRIPT_cleaned.csv` | Cleaned transcripts |
| **Input** | `merged_labels.csv` | Participant labels |
| **Output** | `text_features2.csv` | Features with VADER sentiment + SBERT (Cell 1) |
| **Output** | `text_features_updated.csv` | Features with pause analysis + per-sentence SBERT (Cell 2) |

---

## Step 5 – Non-Phasewise Model Training

These notebooks train and evaluate machine learning classifiers using features extracted from **full (non-segmented) interviews**. Each notebook tests three classifiers — **Random Forest**, **XGBoost**, and **SVM** — with three evaluation strategies.

### Common Configuration

| Parameter | Value |
|-----------|-------|
| Dataset size | 141 samples (99 non-depressed, 42 depressed) |
| Train/Test split | 80/20 stratified (for 0-fold) |
| Random state | 42 |
| Classification threshold | 0.40 |
| Scaling | StandardScaler (z-score normalization) |
| Class imbalance handling | `class_weight="balanced"` (RF, SVM) or `scale_pos_weight` (XGBoost) |

### 5a. Audio-Only Model (`audio_only.ipynb`)

**File:** `training_models/non-phasewise training/audio_only.ipynb`

```bash
jupyter notebook "training_models/non-phasewise training/audio_only.ipynb"
```

**Input:** `../data/audio_features_updated.csv` (141 rows × 132 features)

| Cell | Classifier | Evaluation |
|------|------------|------------|
| 1 | Random Forest | 0-fold (80/20 split), `n_estimators=400`, `class_weight=balanced` |
| 2 | Random Forest | 5-fold StratifiedKFold |
| 3 | Random Forest | 10-fold StratifiedKFold |
| 4 | XGBoost | 0-fold, `n_estimators=300`, `max_depth=4`, `lr=0.05` |
| 5 | XGBoost | 5-fold, `n_estimators=500`, `max_depth=5` |
| 6 | XGBoost | 10-fold StratifiedKFold |
| 7 | SVM | 0-fold, `kernel=rbf`, `C=1.0`, `gamma=scale` |
| 8 | SVM | 5-fold StratifiedKFold |
| 9 | SVM | 10-fold StratifiedKFold |

### 5b. Text-Only Model (`text_only.ipynb`)

**File:** `training_models/non-phasewise training/text_only.ipynb`

```bash
jupyter notebook "training_models/non-phasewise training/text_only.ipynb"
```

**Input:** Uses two data files:
- `../data/selected_features_dataset.csv` (141×501) for RF 0-fold and RF 5-fold cells
- `../data/text_features_updated.csv` (141×778) for all other cells

Contains 9 code cells with the same structure as `audio_only.ipynb`.

### 5c. Multimodal Model (`multimodal.ipynb`)

**File:** `training_models/non-phasewise training/multimodal.ipynb`

```bash
jupyter notebook "training_models/non-phasewise training/multimodal.ipynb"
```

**Input:** `../data/multimodal_features.csv` (141 rows × 445 features)

Contains 9 code cells combining audio and text features. Same structure as above with RF, XGBoost, and SVM classifiers across 0-fold, 5-fold, and 10-fold evaluations.

### Evaluation Metrics (All Non-Phasewise Notebooks)

- Classification Report (precision, recall, F1-score per class)
- ROC-AUC Score
- Per-fold accuracy (for cross-validation)
- Mean ± Std deviation across folds

---

## Step 6 – Phasewise Model Training

These notebooks train and evaluate classifiers using features that have been **segmented by interview phases** (e.g., introduction, questions, free response), providing temporal granularity to the analysis. These notebooks add a fourth classifier — **CatBoost** — in addition to Random Forest, XGBoost, and SVM.

> **CatBoost** (Categorical Boosting) is a gradient boosting algorithm that handles categorical features natively and is known for robust performance with limited data.

### 6a. Audio-Only Phasewise (`audio_only_phasewise.ipynb`)

**File:** `training_models/phasewise training/audio_only_phasewise.ipynb`

```bash
jupyter notebook "training_models/phasewise training/audio_only_phasewise.ipynb"
```

**Input:** `../data/audio_features_phasewise.csv` and/or `../data/merged_audio_hubert_features.csv`

Trains RF, XGBoost, SVM, and CatBoost on phasewise audio features with 0-fold, 5-fold, and 10-fold stratified cross-validation (12 code cells total).

### 6b. Text-Only Phasewise (`text_only_phasewise.ipynb`)

**File:** `training_models/phasewise training/text_only_phasewise.ipynb`

```bash
jupyter notebook "training_models/phasewise training/text_only_phasewise.ipynb"
```

**Input:** `../data/text_features_phasewise.csv`

Trains RF, XGBoost, SVM, and CatBoost on phasewise text features with 0-fold, 5-fold, and 10-fold stratified cross-validation (12 code cells total).

### 6c. Multimodal Phasewise (`multimodal_phasewise.ipynb`)

**File:** `training_models/phasewise training/multimodal_phasewise.ipynb`

```bash
jupyter notebook "training_models/phasewise training/multimodal_phasewise.ipynb"
```

**Input:** `../data/multimodal_audio_text_features.csv`

Trains RF, XGBoost, SVM, and CatBoost on combined phasewise audio+text features with 0-fold, 5-fold, and 10-fold stratified cross-validation (12 code cells total).

### Phasewise Notebook Structure (Common to All Three)

| Cell | Classifier | Evaluation |
|------|------------|------------|
| 1 | Random Forest | 0-fold |
| 2 | Random Forest | 5-fold |
| 3 | Random Forest | 10-fold |
| 4 | XGBoost | 0-fold |
| 5 | XGBoost | 5-fold |
| 6 | XGBoost | 10-fold |
| 7 | SVM | 0-fold |
| 8 | SVM | 5-fold |
| 9 | SVM | 10-fold |
| 10 | CatBoost | 0-fold |
| 11 | CatBoost | 5-fold |
| 12 | CatBoost | 10-fold |

### Additional Imports (Phasewise)

- `catboost` — `CatBoostClassifier`
- All imports from non-phasewise notebooks also apply

---

## 📂 Data Files Reference

### Files in `data/` Directory

| File | Description | Size |
|------|-------------|------|
| `audio_features_phasewise.csv` | Phasewise audio features | ~694 KB |
| `hubert_phasewise_embeddings.csv` | HuBERT model embeddings (phasewise) | ~3.7 MB |
| `merged_audio_hubert_features.csv` | Combined audio + HuBERT features | ~4.4 MB |
| `multimodal_audio_text_features.csv` | Combined phasewise multimodal features | ~4.5 MB |
| `text_features_phasewise.csv` | Phasewise text features | ~3.8 MB |

### Additional Data Files (Referenced by Notebooks)

| File | Description |
|------|-------------|
| `audio_features_updated.csv` | Non-phasewise audio features (132 columns) |
| `text_features_updated.csv` | Non-phasewise text features (778 columns) |
| `selected_features_dataset.csv` | Selected text features subset (501 columns) |
| `multimodal_features.csv` | Non-phasewise combined features (445 columns) |
| `merged_labels.csv` | Ground truth labels (`Participant_ID`, `PHQ8_Binary`) |

---

## 🛠 Troubleshooting & Notes

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'parselmouth'` | Install with: `pip install praat-parselmouth` |
| `NLTK resource vader_lexicon not found` | Run: `python -c "import nltk; nltk.download('vader_lexicon')"` |
| Google Colab drive mount fails | Ensure you are logged into the correct Google account. For local execution, modify paths and remove the `google.colab` import. |
| Memory errors during SBERT encoding | Process transcripts in smaller batches or use a machine with more RAM. The `all-MiniLM-L6-v2` model is lightweight but encoding many long transcripts can still require significant memory. |

### ⚠️ Important Notes

- **Execution order matters:** Run audio preprocessing (Step 1) and transcript preprocessing (Step 2) before feature extraction (Steps 3–4), and complete feature extraction before model training (Steps 5–6).
- **File paths** in the scripts may need to be updated to match your local directory structure. The scripts were originally written with absolute Windows paths.
- The **Transcript_preprocessing** notebook was designed for Google Colab. To run locally, remove the `google.colab` import and update folder paths.
- All training notebooks produce **console output only** (classification reports, ROC-AUC scores). No trained models are saved to disk.
- The **classification threshold** is set to **0.40** (instead of the default 0.50) across all models to account for class imbalance and improve recall for the depressed class.

---

## 🔄 Execution Pipeline Summary

```
Raw Audio + Transcripts
    ↓  (cleaned_audio.py)
Cleaned Audio WAVs
    ↓  (audio_feature_extraction.ipynb)
Audio Feature CSVs

Raw Transcripts
    ↓  (Transcript_preprocessing.ipynb)
Cleaned Transcripts
    ↓  (text_feature_extraction.ipynb)
Text Feature CSVs

Audio Features + Text Features
    ↓  (Merged into multimodal features)
Model Training (non-phasewise / phasewise)
    ↓
Evaluation Results (Classification Reports, ROC-AUC)
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
