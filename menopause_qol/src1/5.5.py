import pandas as pd
import numpy as np
import os
import warnings
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# --- CẤU HÌNH ---
# Sử dụng tập Train đã được cân bằng bằng SMOTE
INPUT_TRAIN = "train_data_balanced.csv"
INPUT_TEST = "test_data.csv"

# Danh sách 25 câu hỏi tiềm năng (Đã loại bỏ Chronic_Disease vì đưa vào Demo Feats)
POTENTIAL_QUESTIONS = [
    'MEN_HotFlash', 'MEN_Memory', 'MEN_Sleep', 'MEN_Headache', 'MEN_MuscleAche',
    'MEN_Fatigue', 'MEN_Weight', 'MEN_Skin', 'MEN_Depressed', 'MEN_Impatient',
    'MEN_Urine', 'MEN_Libido', 'PSS_1', 'PSS_2', 'PSS_3', 'PSS_4', 'PSS_5',
    'PSS_6', 'PSS_7', 'PSS_8', 'PSS_9', 'PSS_10', 'Supp_Calcium', 'Supp_Omega3'
]

# Thêm Chronic_Disease vào nhóm đặc trưng nền tảng
DEMO_FEATS = ['Age', 'BMI', 'Education_Code', 'Income_Code', 'Job_Code', 'Marital_Code',
              'Meno_Duration_New', 'Meno_Status', 'Chronic_Disease']


def run_fine_tuning():
    if not os.path.exists(INPUT_TRAIN):
        print(f"❌ LỖI: Không tìm thấy file '{INPUT_TRAIN}'. Hãy chạy bước 3.5 trước.")
        return

    train_df = pd.read_csv(INPUT_TRAIN)
    test_df = pd.read_csv(INPUT_TEST)

    targets = ['PSS_Score', 'MENQOL_Score']
    scaler = StandardScaler()

    print("--- GIAI ĐOẠN 5: TÌM CÂU HỎI VÀNG & FINE-TUNING (SMOTE ENHANCED) ---")

    for target in targets:
        print(f"\n" + "=" * 60)
        print(f"🚀 TỐI ƯU HÓA MỤC TIÊU: {target}")
        print("=" * 60)

        for c_id in sorted(train_df['Cluster'].unique()):
            c_train = train_df[train_df['Cluster'] == c_id]
            c_test = test_df[test_df['Cluster'] == c_id]

            # --- BƯỚC 1: TÌM CÂU HỎI VÀNG (FEATURE IMPORTANCE) ---
            # Sử dụng tập Train đã cân bằng giúp việc tìm feature importance chính xác hơn
            selector = ExtraTreesRegressor(n_estimators=100, random_state=42)
            valid_potential = [q for q in POTENTIAL_QUESTIONS if q in c_train.columns]

            selector.fit(c_train[valid_potential], c_train[target])
            importances = pd.Series(selector.feature_importances_, index=valid_potential)
            gold_qs = importances.nlargest(5).index.tolist()

            print(f"\n📍 Cluster {c_id} (Train: {len(c_train)} mẫu | Test: {len(c_test)} mẫu):")
            print(f"   -> 5 Câu hỏi vàng: {gold_qs}")

            # --- BƯỚC 2: FINE-TUNING VỚI GRID SEARCH ---
            final_features = DEMO_FEATS + gold_qs
            X_train = c_train[final_features]
            y_train = c_train[target]

            # Chuẩn hóa dữ liệu trước khi train
            X_train_scaled = scaler.fit_transform(X_train)

            param_grid = {
                'n_estimators': [100, 300],
                'max_depth': [4, 6, 8, None],
                'min_samples_leaf': [2, 4],
                'max_features': ['sqrt', 'log2', None]
            }

            grid = GridSearchCV(ExtraTreesRegressor(random_state=42), param_grid, cv=3, scoring='r2')
            grid.fit(X_train_scaled, y_train)

            best_model = grid.best_estimator_
            print(f"   -> Best Params: {grid.best_params_}")

            # --- BƯỚC 3: ĐÁNH GIÁ TRÊN TẬP TEST THẬT ---
            if len(c_test) > 0:
                X_test_scaled = scaler.transform(c_test[final_features])
                y_test = c_test[target]
                y_pred = best_model.predict(X_test_scaled)

                # Tính R2 và MAE
                # Nếu chỉ có 1 mẫu, R2 sẽ là nan, ta cần check để tránh in ra nan
                if len(y_test) > 1:
                    r2 = r2_score(y_test, y_pred)
                    print(f"   ✅ R2 trên tập TEST: {r2:.4f}")
                else:
                    print(f"   ⚠️ Chỉ có 1 mẫu test: Thực tế={y_test.values[0]:.2f}, Dự báo={y_pred[0]:.2f}")

                mae = mean_absolute_error(y_test, y_pred)
                print(f"   ✅ MAE trên tập TEST: {mae:.4f}")
            else:
                print("   ⚠️ Không có dữ liệu test cho cụm này.")


if __name__ == "__main__":
    run_fine_tuning()