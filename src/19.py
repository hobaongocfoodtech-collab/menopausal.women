import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor

# --- IMPORT CÁC MÔ HÌNH VÔ ĐỊCH (TỪ KẾT QUẢ LAZYPREDICT) ---
from sklearn.linear_model import Lasso
from lightgbm import LGBMRegressor
import warnings
import os
import logging

warnings.filterwarnings('ignore')
logging.getLogger("lightgbm").setLevel(logging.ERROR)  # Tắt log rác của LightGBM

# 1. CẤU HÌNH & DỮ LIỆU
FILE_PATH = r"D:\PycharmProjects\PNMK\icatsd2026_menopause_qol\data\processed\clean_data_final.csv"

# Các biến nhân khẩu học cố định
DEMO_FEATS = ['Age', 'BMI', 'Education_Code', 'Income_Code', 'Marital_Code',
              'Job_Code', 'Info_Search_Code', 'Meno_Duration', 'Meno_Group']

# CHIẾN LƯỢC CÂU HỎI VÀNG (Giữ nguyên)
ADAPTIVE_STRATEGY = {
    0: {
        'PSS_Score': ['MEN_Urine', 'PSS_10', 'PSS_1', 'PSS_2', 'PSS_8'],
        'MENQOL_Score': ['MEN_Impatient', 'MEN_Sleep', 'MEN_Fatigue', 'MEN_Depressed', 'PSS_6']
    },
    1: {
        'PSS_Score': ['PSS_1', 'PSS_3', 'PSS_6', 'MEN_Impatient', 'PSS_2'],
        'MENQOL_Score': ['MEN_Depressed', 'MEN_Libido', 'MEN_Skin', 'MEN_HotFlash', 'MEN_Fatigue']
    },
    2: {
        'PSS_Score': ['PSS_3', 'PSS_6', 'PSS_10', 'PSS_9', 'PSS_5'],
        'MENQOL_Score': ['MEN_Libido', 'PSS_3', 'MEN_Depressed', 'MEN_Impatient', 'MEN_HotFlash']
    }
}

# --- CẤU HÌNH MÔ HÌNH CHIẾN THẮNG (BEST MODELS) ---
# Dựa trên kết quả GridSearch ở bước trước (src/16.py)
BEST_MODELS = {
    'PSS_Score': Lasso(alpha=0.1, random_state=42),  # Champion cho PSS
    'MENQOL_Score': LGBMRegressor(learning_rate=0.05, max_depth=3, n_estimators=100, num_leaves=15, verbose=-1,
                                  random_state=42)  # Champion cho MENQOL
}


def validate_system_corrected():
    if not os.path.exists(FILE_PATH):
        print(f"❌ Lỗi: Không tìm thấy file {FILE_PATH}")
        return

    print("--- ĐANG TẢI DỮ LIỆU VÀ CHUẨN BỊ MÔI TRƯỜNG KIỂM THỬ ---")
    df = pd.read_csv(FILE_PATH)

    # 1. Xử lý sơ bộ
    cols_text = ['Chronic_Disease', 'Supp_Calcium', 'Supp_Omega3', 'Exercise']
    for c in cols_text:
        if c in df.columns:
            df[c] = df[c].apply(lambda x: 0 if pd.isna(x) or str(x).strip().lower() in ['không', 'nan', '0'] else 1)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 2. Tái tạo Phân cụm & Chia Train/Test
    scaler = StandardScaler()
    X_cluster = scaler.fit_transform(df[DEMO_FEATS])
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_cluster)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Cluster'])

    print(f"✅ Dữ liệu Test: {len(test_df)} mẫu (được giấu kín hoàn toàn).")

    # ==========================================================================
    # QUY TRÌNH KIỂM THỬ (VALIDATION LOOP)
    # ==========================================================================

    results = []
    targets = ['PSS_Score', 'MENQOL_Score']

    for target in targets:
        print(f"\n>>> ĐANG KIỂM TRA MỤC TIÊU: {target}")

        # Lấy mô hình vô địch tương ứng
        champion_model = BEST_MODELS[target]
        model_name = type(champion_model).__name__
        print(f"    (Sử dụng mô hình vô địch: {model_name})")

        # --- MODEL A: BASELINE (CHỈ DÙNG 9 CÂU NHÂN KHẨU) ---
        # Huấn luyện mô hình vô địch trên tập Train (chỉ nhân khẩu học)
        champion_model.fit(train_df[DEMO_FEATS], train_df[target])

        y_pred_base = champion_model.predict(test_df[DEMO_FEATS])
        r2_base = r2_score(test_df[target], y_pred_base)

        # --- MODEL B: ADAPTIVE AI (NHÂN KHẨU + 5 CÂU VÀNG) ---
        y_pred_adaptive = []
        y_true_adaptive = []

        sub_models = {}
        for c_id in range(3):
            c_train = train_df[train_df['Cluster'] == c_id]
            feats = DEMO_FEATS + ADAPTIVE_STRATEGY[c_id][target]

            # Khởi tạo lại mô hình vô địch mới cho từng cụm
            # Lưu ý: Với LGBM và Lasso, ta cần fit lại trên dữ liệu mới (nhiều cột hơn)
            if target == 'PSS_Score':
                m = Lasso(alpha=0.1, random_state=42)
            else:
                m = LGBMRegressor(learning_rate=0.05, max_depth=3, n_estimators=100, num_leaves=15, verbose=-1,
                                  random_state=42)

            m.fit(c_train[feats], c_train[target])
            sub_models[c_id] = m

        # Dự báo từng dòng trong tập Test
        for idx, row in test_df.iterrows():
            c_id = int(row['Cluster'])
            feats = DEMO_FEATS + ADAPTIVE_STRATEGY[c_id][target]

            input_vec = row[feats].values.reshape(1, -1)
            pred = sub_models[c_id].predict(input_vec)[0]

            y_pred_adaptive.append(pred)
            y_true_adaptive.append(row[target])

        r2_adapt = r2_score(y_true_adaptive, y_pred_adaptive)
        mae_adapt = mean_absolute_error(y_true_adaptive, y_pred_adaptive)
        rmse_adapt = np.sqrt(mean_squared_error(y_true_adaptive, y_pred_adaptive))

        # Lưu kết quả
        results.append({
            'Mục tiêu': target,
            'Model': model_name,
            'R2 Baseline (9 câu)': r2_base,
            'R2 Adaptive (14 câu)': r2_adapt,
            'Cải thiện (%)': (r2_adapt - r2_base) * 100,
            'MAE': mae_adapt
        })

        # --- VẼ BIỂU ĐỒ SCATTER ---
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        sns.scatterplot(x=test_df[target], y=y_pred_base, color='gray', alpha=0.5)
        plt.plot([test_df[target].min(), test_df[target].max()],
                 [test_df[target].min(), test_df[target].max()], 'r--')
        plt.title(f"{target} - Baseline ({model_name})\nR2={r2_base:.2f}")
        plt.xlabel("Thực tế")
        plt.ylabel("Dự báo")

        plt.subplot(1, 2, 2)
        sns.scatterplot(x=y_true_adaptive, y=y_pred_adaptive, color='green', s=80)
        plt.plot([min(y_true_adaptive), max(y_true_adaptive)],
                 [min(y_true_adaptive), max(y_true_adaptive)], 'r--', linewidth=2)
        plt.title(f"{target} - Adaptive AI ({model_name})\nR2={r2_adapt:.2f}")
        plt.xlabel("Thực tế")

        plt.tight_layout()
        plt.show()

    # ==========================================================================
    # TỔNG KẾT
    # ==========================================================================
    print("\n" + "=" * 80)
    print("BẢNG TỔNG HỢP HIỆU NĂNG (SỬ DỤNG TOP MODELS)")
    print("=" * 80)
    res_df = pd.DataFrame(results)
    print(res_df.round(4))


if __name__ == "__main__":
    validate_system_corrected()