import pandas as pd
import numpy as np
import os
import joblib
from collections import Counter
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import RFE, mutual_info_regression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# --- CAU HINH ---
INPUT_TRAIN = "train_data_balanced.csv"
INPUT_TEST = "test_data.csv"

POTENTIAL_QUESTIONS = [
    'MEN_HotFlash', 'MEN_Memory', 'MEN_Sleep', 'MEN_Headache', 'MEN_MuscleAche',
    'MEN_Fatigue', 'MEN_Weight', 'MEN_Skin', 'MEN_Depressed', 'MEN_Impatient',
    'MEN_Urine', 'MEN_Libido', 'PSS_1', 'PSS_2', 'PSS_3', 'PSS_4', 'PSS_5',
    'PSS_6', 'PSS_7', 'PSS_8', 'PSS_9', 'PSS_10', 'Supp_Calcium', 'Supp_Omega3'
]

DEMO_FEATS = ['Age', 'BMI', 'Education_Code', 'Income_Code', 'Job_Code', 'Marital_Code',
              'Meno_Duration_New', 'Meno_Status', 'Chronic_Disease']


def get_ensemble_gold_questions(X, y):
    et = ExtraTreesRegressor(n_estimators=100, random_state=42)
    et.fit(X, y)
    et_feats = pd.Series(et.feature_importances_, index=X.columns).nlargest(5).index.tolist()

    selector = RFE(ExtraTreesRegressor(n_estimators=100, random_state=42), n_features_to_select=5)
    selector.fit(X, y)
    rfe_feats = X.columns[selector.support_].tolist()

    mi_scores = mutual_info_regression(X, y, discrete_features=True, random_state=42)
    mi_feats = pd.Series(mi_scores, index=X.columns).nlargest(5).index.tolist()

    all_selected = et_feats + rfe_feats + mi_feats
    counts = Counter(all_selected)
    ensemble_gold = [item for item, count in counts.most_common(5)]

    return ensemble_gold, {"ExtraTrees": et_feats, "RFE": rfe_feats, "MutualInfo": mi_feats}


def run_ensemble_pipeline():
    if not os.path.exists(INPUT_TRAIN) or not os.path.exists(INPUT_TEST):
        print("Loi: Khong tim thay file du lieu.")
        return

    train_df = pd.read_csv(INPUT_TRAIN)
    test_df = pd.read_csv(INPUT_TEST)
    targets = ['PSS_Score', 'MENQOL_Score']
    scaler = StandardScaler()

    print("--- GIAI DOAN 5.8: ENSEMBLE SELECTION & FINE-TUNING ---")

    for target in targets:
        print(f"\n============================================================")
        print(f"TOI UU HOA CHO: {target}")
        print("============================================================")

        for c_id in sorted(train_df['Cluster'].unique()):
            c_train = train_df[train_df['Cluster'] == c_id]
            c_test = test_df[test_df['Cluster'] == c_id]

            X_train_raw = c_train[POTENTIAL_QUESTIONS]
            y_train = c_train[target]

            gold_qs, details = get_ensemble_gold_questions(X_train_raw, y_train)

            print(f"\nCluster {c_id} (Train: {len(c_train)} | Test: {len(c_test)}):")
            print(f"   + ExtraTrees: {details['ExtraTrees']}")
            print(f"   + RFE       : {details['RFE']}")
            print(f"   + MutualInfo: {details['MutualInfo']}")
            print(f"   => KET QUA BAU CHON (GOLD): {gold_qs}")

            final_features = DEMO_FEATS + gold_qs
            X_train_final = c_train[final_features]
            X_train_scaled = scaler.fit_transform(X_train_final)

            param_grid = {
                'n_estimators': [100, 300],
                'max_depth': [6, 8, None],
                'min_samples_leaf': [2, 4],
                'max_features': ['sqrt', None]
            }

            grid = GridSearchCV(ExtraTreesRegressor(random_state=42), param_grid, cv=3, scoring='r2')
            grid.fit(X_train_scaled, y_train)
            best_model = grid.best_estimator_

            if len(c_test) > 0:
                X_test_scaled = scaler.transform(c_test[final_features])
                y_test = c_test[target]
                y_pred = best_model.predict(X_test_scaled)

                # Logic hien thi chi tiet tuong tu code 5.5
                if len(y_test) > 1:
                    r2 = r2_score(y_test, y_pred)
                    print(f"   Chỉ số R2 TEST: {r2:.4f}")
                else:
                    print(f"   Thông báo: Chỉ có 1 mẫu test. Thực tế={y_test.values[0]:.2f}, Dự báo={y_pred[0]:.2f}")

                mae = mean_absolute_error(y_test, y_pred)
                print(f"   Chỉ số MAE TEST: {mae:.4f}")


if __name__ == "__main__":
    run_ensemble_pipeline()