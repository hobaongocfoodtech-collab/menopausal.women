import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

# --- CẤU HÌNH ---
INPUT_TRAIN = "train_data.csv"
INPUT_TEST = "test_data.csv"

# Danh sách 26 câu hỏi tiềm năng (Symptoms & Lifestyle)
POTENTIAL_QUESTIONS = [
    'MEN_HotFlash', 'MEN_Memory', 'MEN_Sleep', 'MEN_Headache', 'MEN_MuscleAche',
    'MEN_Fatigue', 'MEN_Weight', 'MEN_Skin', 'MEN_Depressed', 'MEN_Impatient',
    'MEN_Urine', 'MEN_Libido', 'PSS_1', 'PSS_2', 'PSS_3', 'PSS_4', 'PSS_5',
    'PSS_6', 'PSS_7', 'PSS_8', 'PSS_9', 'PSS_10', 'Chronic_Disease',
    'Supp_Calcium', 'Supp_Omega3'
]

DEMO_FEATS = ['Age', 'BMI', 'Education_Code', 'Income_Code', 'Job_Code', 'Marital_Code',
              'Meno_Duration_New', 'Meno_Status']


def run_fine_tuning():
    train_df = pd.read_csv(INPUT_TRAIN)
    test_df = pd.read_csv(INPUT_TEST)

    targets = ['PSS_Score', 'MENQOL_Score']

    print("--- GIAI ĐOẠN 5: TÌM CÂU HỎI VÀNG & FINE-TUNING ---")

    for target in targets:
        print(f"\n🚀 TỐI ƯU HÓA CHO: {target}")

        for c_id in sorted(train_df['Cluster'].unique()):
            c_train = train_df[train_df['Cluster'] == c_id]
            if len(c_train) < 5: continue

            # --- BƯỚC 1: TÌM CÂU HỎI VÀNG (FEATURE IMPORTANCE) ---
            # Dùng ExtraTrees để "quét" xem câu nào ảnh hưởng nhất đến Target
            selector = ExtraTreesRegressor(n_estimators=100, random_state=42)
            # Lọc các cột hiện có trong data
            valid_potential = [q for q in POTENTIAL_QUESTIONS if q in c_train.columns]

            selector.fit(c_train[valid_potential], c_train[target])
            importances = pd.Series(selector.feature_importances_, index=valid_potential)
            gold_qs = importances.nlargest(5).index.tolist()

            print(f"\n   📍 Cluster {c_id}:")
            print(f"      -> 5 Câu hỏi vàng: {gold_qs}")

            # --- BƯỚC 2: FINE-TUNING VỚI CÂU HỎI VÀNG ---
            # Đầu vào mới = Nhân khẩu học + 5 câu hỏi vàng
            final_features = DEMO_FEATS + gold_qs
            X_train = c_train[final_features]
            y_train = c_train[target]

            # Lưới tham số để GridSearch
            param_grid = {
                'n_estimators': [100, 300],
                'max_depth': [4, 6, None],
                'min_samples_leaf': [2, 4]
            }

            grid = GridSearchCV(ExtraTreesRegressor(random_state=42), param_grid, cv=3, scoring='r2')
            grid.fit(X_train, y_train)

            best_model = grid.best_estimator_

            # --- BƯỚC 3: KIỂM CHỨNG TRÊN TẬP TEST ---
            c_test = test_df[test_df['Cluster'] == c_id]
            if len(c_test) > 0:
                X_test = c_test[final_features]
                y_test = c_test[target]
                y_pred = best_model.predict(X_test)
                score = r2_score(y_test, y_pred)
                print(f"      ✅ R2 trên tập TEST: {score:.4f}")


if __name__ == "__main__":
    run_fine_tuning()