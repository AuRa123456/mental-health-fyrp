import pandas as pd
import numpy as np

np.random.seed(42)

n_samples = 600

data = []

for i in range(n_samples):

    depressed = np.random.choice([0, 1], p=[0.65, 0.35])

    # overlapping distributions
    mean_pitch = np.random.normal(135 - depressed*10, 20)
    pitch_var = np.random.normal(12 - depressed*3, 6)
    energy = np.random.normal(0.4 - depressed*0.05, 0.15)
    pause_duration = np.random.normal(1.5 + depressed*0.5, 0.7)

    mfcc_features = np.random.normal(0, 1.5, 5)

    text_features = np.random.normal(0 - depressed*0.3, 1.2, 10)

    row = [
        mean_pitch,
        pitch_var,
        energy,
        pause_duration,
        *mfcc_features,
        *text_features,
        depressed
    ]

    data.append(row)

columns = (
    ["mean_pitch", "pitch_var", "energy", "pause_duration"] +
    [f"mfcc_{i}" for i in range(5)] +
    [f"text_feat_{i}" for i in range(10)] +
    ["label"]
)

df = pd.DataFrame(data, columns=columns)

df.to_csv("synthetic_depression_dataset.csv", index=False)

print("Realistic dataset created!")