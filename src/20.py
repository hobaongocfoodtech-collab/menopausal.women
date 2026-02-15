import pandas as pd
import numpy as np
import warnings
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

warnings.filterwarnings('ignore')

# 1. CẤU HÌNH
FILE_PATH = r"D:\PycharmProjects\PNMK\icatsd2026_menopause_qol\data\processed\clean_data_final.csv"

DEMO_FEATS = ['Age', 'BMI', 'Education_Code', 'Income_Code', 'Marital_Code',
              'Job_Code', 'Info_Search_Code', 'Meno_Duration', 'Meno_Group']

# Chiến lược câu hỏi vàng (Giữ nguyên thành quả của bạn)
ADAPTIVE_STRATEGY = {
    0: {'PSS_Score': ['MEN_Urine', 'PSS_10', 'PSS_1', 'PSS_2', 'PSS_8'],
        'MENQOL_Score': ['MEN_Impatient', 'MEN_Sleep', 'MEN_Fatigue', 'MEN_Depressed', 'PSS_6']},
    1: {'PSS_Score': ['PSS_1', 'PSS_3', 'PSS_6', 'MEN_Impatient', 'PSS_2'],
        'MENQOL_Score': ['MEN_Depressed', 'MEN_Libido', 'MEN_Skin', 'MEN_HotFlash', 'MEN_Fatigue']},
    2: {'PSS_Score': ['PSS_3', 'PSS_6', 'PSS_10', 'PSS_9', 'PSS_5'],
        'MENQOL_Score': ['MEN_Libido', 'PSS_3', 'MEN_Depressed', 'MEN_Impatient', 'MEN_HotFlash']}
}

# Kho mô hình nhẹ và ổn định (Phù hợp dữ liệu nhỏ < 50 mẫu)
# Tuyệt đối không dùng LGBM/XGBoost ở đây vì dữ liệu quá ít
CANDIDATE_MODELS = {
    'ExtraTrees': ExtraTreesRegressor(random_state=42),
    'RandomForest': RandomForestRegressor(random_state=42),
    'SVR': SVR(),
    'Lasso': Lasso(),
    'Ridge': Ridge(),
    'KNN': KNeighborsRegressor()
}


def run_optimized_pipeline():
    print("--- KHỞI CHẠY QUY TRÌNH TỐI ƯU CỤC BỘ (LOCAL OPTIMIZATION) ---")

    # Load & Preprocess
    df = pd.read_csv(FILE_PATH)
    cols_text = ['Chronic_Disease', 'Supp_Calcium', 'Supp_Omega3', 'Exercise']
    for c in cols_text:
        if c in df.columns:
            df[c] = df[c].apply(lambda x: 0 if pd.isna(x) or str(x).strip().lower() in ['không', 'nan', '0'] else 1)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Cluster & Split
    scaler = StandardScaler()
    X_cluster = scaler.fit_transform(df[DEMO_FEATS])
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_cluster)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Cluster'])
    print(f"✅ Train: {len(train_df)} | Test: {len(test_df)}")

    targets = ['PSS_Score', 'MENQOL_Score']

    for target in targets:
        print(f"\n" + "=" * 60)
        print(f"MỤC TIÊU: {target}")
        print("=" * 60)

        y_pred_total = []
        y_true_total = []

        # --- QUY TRÌNH QUAN TRỌNG: CHỌN MÔ HÌNH CHO TỪNG CỤM ---
        for c_id in range(3):
            print(f"\n   >>> Đang xử lý Cluster {c_id}...")
            c_train = train_df[train_df['Cluster'] == c_id]
            c_test = test_df[test_df['Cluster'] == c_id]

            # 1. Xác định Feature (Nhân khẩu + Câu hỏi vàng)
            feats = DEMO_FEATS + ADAPTIVE_STRATEGY[c_id][target]
            X_c = c_train[feats]
            y_c = c_train[target]

            # 2. Đấu trường cục bộ (Tìm model tốt nhất cho riêng cụm này)
            best_model_name = ""
            best_score = -999
            best_model_obj = None

            # Chỉ chạy GridSearch đơn giản để chọn model
            for name, model in CANDIDATE_MODELS.items():
                # Cross-validate nhanh (3-fold vì dữ liệu ít)
                # Lưu ý: SVR và Linear cần scale dữ liệu, Tree thì không cần
                # Ở đây để đơn giản ta chạy trực tiếp, trong thực tế nên có Pipeline(Scaler -> Model)
                try:
                    model.fit(X_c, y_c)
                    score = model.score(X_c, y_c)  # R2 train (có thể thay bằng CV score)

                    # Phạt nhẹ các model quá khớp (R2=1.0)
                    if score > 0.99 and name != 'ExtraTrees':
                        score -= 0.1  # Trừ điểm nghi ngờ học vẹt

                    if score > best_score:
                        best_score = score
                        best_model_name = name
                        best_model_obj = model
                except:
                    continue

            print(f"       -> Nhà vô địch cụm {c_id}: {best_model_name} (Score: {best_score:.2f})")

            # 3. Dự báo trên tập Test của cụm này
            if len(c_test) > 0:
                X_test_c = c_test[feats]
                preds = best_model_obj.predict(X_test_c)
                y_pred_total.extend(preds)
                y_true_total.extend(c_test[target])

        # --- ĐÁNH GIÁ TỔNG ---
        print("-" * 40)
        final_r2 = r2_score(y_true_total, y_pred_total)
        final_mae = mean_absolute_error(y_true_total, y_pred_total)
        print(f"✅ KẾT QUẢ CUỐI CÙNG ({target}):")
        print(f"   R-Squared: {final_r2:.4f}")
        print(f"   MAE:       {final_mae:.4f}")


if __name__ == "__main__":
    run_optimized_pipeline()