import pandas as pd
import numpy as np
import os
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler

# --- CAU HINH ---
INPUT_TRAIN = "train_data_balanced.csv"

# Dong bo voi code 5.5: Loai bo Chronic_Disease khoi day vi da dua vao Demo Feats
POTENTIAL_QUESTIONS = [
    'MEN_HotFlash', 'MEN_Memory', 'MEN_Sleep', 'MEN_Headache', 'MEN_MuscleAche',
    'MEN_Fatigue', 'MEN_Weight', 'MEN_Skin', 'MEN_Depressed', 'MEN_Impatient',
    'MEN_Urine', 'MEN_Libido', 'PSS_1', 'PSS_2', 'PSS_3', 'PSS_4', 'PSS_5',
    'PSS_6', 'PSS_7', 'PSS_8', 'PSS_9', 'PSS_10', 'Supp_Calcium', 'Supp_Omega3'
]

def run_rfe_selection():
    if not os.path.exists(INPUT_TRAIN):
        print("Khong tim thay file train_data_balanced.csv")
        return

    df = pd.read_csv(INPUT_TRAIN)
    targets = ['PSS_Score', 'MENQOL_Score']
    scaler = StandardScaler()

    print("--- GIAI DOAN 5.6: CHON CAU HOI VANG BANG RFE ---")

    for target in targets:
        print(f"\nDUA TREN RFE CHO: {target}")
        for c_id in sorted(df['Cluster'].unique()):
            c_data = df[df['Cluster'] == c_id]
            X = c_data[POTENTIAL_QUESTIONS]
            y = c_data[target]
            X_scaled = scaler.fit_transform(X)

            estimator = ExtraTreesRegressor(n_estimators=100, random_state=42)
            selector = RFE(estimator, n_features_to_select=5, step=1)
            selector = selector.fit(X_scaled, y)

            gold_qs_rfe = np.array(POTENTIAL_QUESTIONS)[selector.support_].tolist()
            print(f"   Cluster {c_id}: {gold_qs_rfe}")

if __name__ == "__main__":
    run_rfe_selection()