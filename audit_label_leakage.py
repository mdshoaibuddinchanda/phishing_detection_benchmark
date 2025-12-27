import pandas as pd
import numpy as np

# Load test data
test_df = pd.read_csv('data/processed/test.csv')
np.random.seed(42)
sample_indices = np.random.choice(len(test_df), 20, replace=False)

for i, idx in enumerate(sample_indices, 1):
    row = test_df.iloc[idx]
    print(f'\n--- Sample {i} (Index {idx}) ---')
    print(f'Label: {row["label"]}')
    print(f'Text (first 200 chars): {row["text"][:200]}...')
    print(f'Contains "phishing": {"phishing" in row["text"].lower()}')
    print(f'Contains "spam": {"spam" in row["text"].lower()}')
    print(f'Contains "malicious": {"malicious" in row["text"].lower()}')
    print(f'Contains "fraud": {"fraud" in row["text"].lower()}')
