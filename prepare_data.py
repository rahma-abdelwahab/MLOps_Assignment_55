"""
prepare_data.py - Generate breast cancer dataset and save to CSV
"""

import os
import pandas as pd
from sklearn.datasets import load_breast_cancer

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Load the breast cancer dataset
cancer_data = load_breast_cancer()
df = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
df['target'] = cancer_data.target

# Save to CSV
df.to_csv("data/breast_cancer.csv", index=False)
print("✅ Dataset prepared and saved to data/breast_cancer.csv")
