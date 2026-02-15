import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # Thư viện để lưu model
import os
import warnings
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings('ignore')

# --- 1. CẤU HÌNH HỆ THỐNG ---
FILE_PATH = r"D:\PycharmProjects\PNMK\icatsd2026_menopause_qol\data\processed\clean_data_final.csv"
MODEL_DIR = r"D:\PycharmProjects\PNMK\icatsd2026_menopause_qol\models"

# Tạo thư mục lưu model nếu chưa có
os.makedirs(MODEL_DIR, exist_ok=True)

DEMO_FEATS = ['Age', 'BMI', 'Education_Code', 'Income_Code', 'Marital_Code',
              'Job_Code', 'Info_Search_Code', 'Meno_Duration', 'Meno_Group']

# CHIẾN LƯỢC CÂU HỎI VÀNG (Đã được kiểm chứng)
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


def load_and_clean_data():
    print("🔄 Đang tải và làm sạch dữ liệu...")
    df = pd.read_csv(FILE_PATH)

    # Xử lý các cột dạng chữ thành số
    cols_text = ['Chronic_Disease', 'Supp_Calcium', 'Supp_Omega3', 'Exercise']
    for c in cols_text:
        if c in df.columns:
            df[c] = df[c].apply(lambda x: 0 if pd.isna(x) or str(x).strip().lower() in ['không', 'nan', '0'] else 1)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    return df


def train_and_save_system():
    df = load_and_clean_data()

    print("🚀 Bắt đầu huấn luyện hệ thống AI toàn diện...")

    # 1. Huấn luyện Phân cụm (K-Means)
    print("   [1/3] Training K-Means Clusterer...")
    scaler = StandardScaler()
    X_cluster = scaler.fit_transform(df[DEMO_FEATS])
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_cluster)

    # Lưu Scaler và KMeans
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    joblib.dump(kmeans, os.path.join(MODEL_DIR, 'kmeans.pkl'))

    # Chia tập Train/Test để đánh giá lần cuối
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Cluster'])

    # 2. Huấn luyện các Model chuyên biệt (Local Experts)
    targets = ['PSS_Score', 'MENQOL_Score']
    model_performance = []

    for target in targets:
        print(f"   [2/3] Training Models cho mục tiêu: {target}...")

        y_true_all = []
        y_pred_all = []

        for c_id in range(3):
            # Lấy dữ liệu train của cụm
            c_train = train_df[train_df['Cluster'] == c_id]
            feats = DEMO_FEATS + ADAPTIVE_STRATEGY[c_id][target]

            # Khởi tạo ExtraTrees (Vua của dữ liệu nhỏ)
            # Tinh chỉnh nhẹ: min_samples_leaf=2 để tránh học vẹt quá mức
            model = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, random_state=42)
            model.fit(c_train[feats], c_train[target])

            # Lưu model con
            joblib.dump(model, os.path.join(MODEL_DIR, f'extratrees_c{c_id}_{target}.pkl'))

            # Dự báo trên tập Test (Validation)
            c_test = test_df[test_df['Cluster'] == c_id]
            if len(c_test) > 0:
                preds = model.predict(c_test[feats])
                y_true_all.extend(c_test[target])
                y_pred_all.extend(preds)

        # Đánh giá tổng hợp
        r2 = r2_score(y_true_all, y_pred_all)
        mae = mean_absolute_error(y_true_all, y_pred_all)
        model_performance.append({'Target': target, 'R2': r2, 'MAE': mae})

        # 3. Vẽ biểu đồ báo cáo (Final Chart)
        plt.figure(figsize=(7, 6))
        sns.scatterplot(x=y_true_all, y=y_pred_all, hue=test_df['Cluster'], palette='viridis', s=100, alpha=0.8)

        # Vẽ đường tham chiếu lý tưởng
        min_val, max_val = min(y_true_all), max(y_true_all)
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

        plt.title(f"Đánh giá mô hình {target}\n(R2 = {r2:.4f})", fontsize=14, fontweight='bold')
        plt.xlabel("Giá trị Thực tế")
        plt.ylabel("Giá trị Dự báo AI")
        plt.legend(title='Nhóm người dùng')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    print("\n✅ HUẤN LUYỆN HOÀN TẤT! HỆ THỐNG ĐÃ SẴN SÀNG.")
    print("📊 Bảng kết quả cuối cùng:")
    print(pd.DataFrame(model_performance))
    print(f"\n📂 Các file model đã được lưu tại: {MODEL_DIR}")


if __name__ == "__main__":
    train_and_save_system()