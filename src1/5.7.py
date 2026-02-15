import pandas as pd
import numpy as np
import os
from sklearn.feature_selection import mutual_info_regression

# --- CAU HINH ---
INPUT_TRAIN = "train_data_balanced.csv"

POTENTIAL_QUESTIONS = [
    'MEN_HotFlash', 'MEN_Memory', 'MEN_Sleep', 'MEN_Headache', 'MEN_MuscleAche',
    'MEN_Fatigue', 'MEN_Weight', 'MEN_Skin', 'MEN_Depressed', 'MEN_Impatient',
    'MEN_Urine', 'MEN_Libido', 'PSS_1', 'PSS_2', 'PSS_3', 'PSS_4', 'PSS_5',
    'PSS_6', 'PSS_7', 'PSS_8', 'PSS_9', 'PSS_10', 'Supp_Calcium', 'Supp_Omega3'
]

def run_mi_selection():
    if not os.path.exists(INPUT_TRAIN):
        print("Khong tim thay file train_data_balanced.csv")
        return

    df = pd.read_csv(INPUT_TRAIN)
    targets = ['PSS_Score', 'MENQOL_Score']

    print("--- GIAI DOAN 5.7: CHON CAU HOI VANG BANG MUTUAL INFORMATION ---")

    for target in targets:
        print(f"\nDUA TREN MUTUAL INFO CHO: {target}")
        for c_id in sorted(df['Cluster'].unique()):
            c_data = df[df['Cluster'] == c_id]
            X = c_data[POTENTIAL_QUESTIONS]
            y = c_data[target]

            mi_scores = mutual_info_regression(X, y, discrete_features=True, random_state=42)
            mi_results = pd.Series(mi_scores, index=POTENTIAL_QUESTIONS)
            gold_qs_mi = mi_results.nlargest(5).index.tolist()
            print(f"   Cluster {c_id}: {gold_qs_mi}")

if __name__ == "__main__":
    run_mi_selection()