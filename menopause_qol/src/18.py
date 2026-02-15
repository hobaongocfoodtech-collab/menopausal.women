import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
import os

warnings.filterwarnings('ignore')

# 1. CẤU HÌNH & DỮ LIỆU
FILE_PATH = r"/menopause_qol\data\processed\clean_data_final.csv"

# Các biến nhân khẩu học cố định
DEMO_FEATS = ['Age', 'BMI', 'Education_Code', 'Income_Code', 'Marital_Code',
              'Job_Code', 'Info_Search_Code', 'Meno_Duration', 'Meno_Group']

# CHIẾN LƯỢC CÂU HỎI VÀNG (Lấy từ kết quả chạy src/17.py của bạn)
ADAPTIVE_STRATEGY = {
    0: {  # Cluster 0
        'PSS_Score': ['MEN_Urine', 'PSS_10', 'PSS_1', 'PSS_2', 'PSS_8'],
        'MENQOL_Score': ['MEN_Impatient', 'MEN_Sleep', 'MEN_Fatigue', 'MEN_Depressed', 'PSS_6']
    },
    1: {  # Cluster 1
        'PSS_Score': ['PSS_1', 'PSS_3', 'PSS_6', 'MEN_Impatient', 'PSS_2'],
        'MENQOL_Score': ['MEN_Depressed', 'MEN_Libido', 'MEN_Skin', 'MEN_HotFlash', 'MEN_Fatigue']
    },
    2: {  # Cluster 2
        'PSS_Score': ['PSS_3', 'PSS_6', 'PSS_10', 'PSS_9', 'PSS_5'],
        'MENQOL_Score': ['MEN_Libido', 'PSS_3', 'MEN_Depressed', 'MEN_Impatient', 'MEN_HotFlash']
    }
}


def validate_system():
    if not os.path.exists(FILE_PATH):
        print(f"❌ Lỗi: Không tìm thấy file {FILE_PATH}")
        return

    print("--- ĐANG TẢI DỮ LIỆU VÀ CHUẨN BỊ MÔI TRƯỜNG KIỂM THỬ ---")
    df = pd.read_csv(FILE_PATH)

    # 1. Xử lý sơ bộ (Pre-processing) giống hệt lúc train
    cols_text = ['Chronic_Disease', 'Supp_Calcium', 'Supp_Omega3', 'Exercise']
    for c in cols_text:
        if c in df.columns:
            df[c] = df[c].apply(lambda x: 0 if pd.isna(x) or str(x).strip().lower() in ['không', 'nan', '0'] else 1)

    # Ép kiểu số toàn bộ để tránh lỗi
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 2. Tái tạo Phân cụm & Chia Train/Test
    # Lưu ý: Phải chia y hệt như lúc tìm ra câu hỏi vàng để đảm bảo tính khách quan
    scaler = StandardScaler()
    X_cluster = scaler.fit_transform(df[DEMO_FEATS])
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_cluster)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Cluster'])

    print(f"✅ Dữ liệu Test: {len(test_df)} mẫu (được giấu kín hoàn toàn).")

    # ==========================================================================
    # QUY TRÌNH KIỂM THỬ (VALIDATION LOOP)
    # ==========================================================================

    results = []  # Lưu kết quả để báo cáo
    targets = ['PSS_Score', 'MENQOL_Score']

    for target in targets:
        print(f"\n>>> ĐANG KIỂM TRA MỤC TIÊU: {target}")

        # --- MODEL A: BASELINE (CHỈ DÙNG 9 CÂU NHÂN KHẨU) ---
        base_model = ExtraTreesRegressor(n_estimators=300, random_state=42)
        base_model.fit(train_df[DEMO_FEATS], train_df[target])

        y_pred_base = base_model.predict(test_df[DEMO_FEATS])
        r2_base = r2_score(test_df[target], y_pred_base)

        # --- MODEL B: ADAPTIVE AI (NHÂN KHẨU + 5 CÂU VÀNG THEO CỤM) ---
        y_pred_adaptive = []
        y_true_adaptive = []

        # Để dự báo cho tập Test, ta phải đi từng mẫu, xem nó thuộc cụm nào, rồi dùng đúng model của cụm đó
        # (Ở đây ta giả lập việc train 3 model con trên tập train trước)

        sub_models = {}
        for c_id in range(3):
            # Lấy data train của cụm này
            c_train = train_df[train_df['Cluster'] == c_id]
            # Lấy features vàng
            feats = DEMO_FEATS + ADAPTIVE_STRATEGY[c_id][target]
            # Train model con
            m = ExtraTreesRegressor(n_estimators=300, max_depth=5, random_state=42)
            m.fit(c_train[feats], c_train[target])
            sub_models[c_id] = m

        # Dự báo từng dòng trong tập Test
        for idx, row in test_df.iterrows():
            c_id = int(row['Cluster'])
            feats = DEMO_FEATS + ADAPTIVE_STRATEGY[c_id][target]

            # Chuẩn bị input (reshape thành 2D array)
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
            'R2 Baseline (9 câu)': r2_base,
            'R2 Adaptive (14 câu)': r2_adapt,
            'Cải thiện (%)': (r2_adapt - r2_base) * 100,
            'MAE (Sai số TB)': mae_adapt,
            'RMSE (Sai số bình phương)': rmse_adapt
        })

        # --- VẼ BIỂU ĐỒ SCATTER (THỰC TẾ vs DỰ BÁO) ---
        plt.figure(figsize=(10, 5))

        # Subplot 1: Baseline
        plt.subplot(1, 2, 1)
        sns.scatterplot(x=test_df[target], y=y_pred_base, color='gray', alpha=0.5)
        plt.plot([test_df[target].min(), test_df[target].max()],
                 [test_df[target].min(), test_df[target].max()], 'r--')
        plt.title(f"{target} - Baseline\n(R2={r2_base:.2f})")
        plt.xlabel("Thực tế")
        plt.ylabel("Dự báo")

        # Subplot 2: Adaptive
        plt.subplot(1, 2, 2)
        sns.scatterplot(x=y_true_adaptive, y=y_pred_adaptive, color='blue', s=80)
        plt.plot([min(y_true_adaptive), max(y_true_adaptive)],
                 [min(y_true_adaptive), max(y_true_adaptive)], 'r--', linewidth=2)
        plt.title(f"{target} - Adaptive AI\n(R2={r2_adapt:.2f})")
        plt.xlabel("Thực tế")

        plt.tight_layout()
        plt.show()

    # ==========================================================================
    # TỔNG KẾT BÁO CÁO
    # ==========================================================================
    print("\n" + "=" * 80)
    print("BẢNG TỔNG HỢP HIỆU NĂNG TRÊN TẬP TEST (23 MẪU)")
    print("=" * 80)
    res_df = pd.DataFrame(results)
    print(res_df.round(4))

    print("\n💡 NHẬN XÉT CỦA CHUYÊN GIA:")
    for i, row in res_df.iterrows():
        target = row['Mục tiêu']
        baseline = row['R2 Baseline (9 câu)']
        adapt = row['R2 Adaptive (14 câu)']

        if adapt > 0.7:
            status = "XUẤT SẮC"
        elif adapt > 0.5:
            status = "KHÁ TỐT"
        else:
            status = "CẦN CẢI THIỆN"

        print(f"- Với {target}: Hệ thống giúp tăng độ chính xác từ {baseline:.2f} lên {adapt:.2f}.")
        print(f"  -> Đánh giá: {status}.")


if __name__ == "__main__":
    validate_system()