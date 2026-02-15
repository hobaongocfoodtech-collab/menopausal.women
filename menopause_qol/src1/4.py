import pandas as pd
import numpy as np
import os
import warnings
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler  # Thêm để chuẩn hóa dữ liệu
# Import các mô hình
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

warnings.filterwarnings('ignore')

# --- CẤU HÌNH ---
# Sử dụng file đã được cân bằng dữ liệu từ bước 3.5
INPUT_TRAIN = "train_data_balanced.csv"
OUTPUT_REPORT = "model_selection_report.csv"

if not os.path.exists(INPUT_TRAIN):
    print(f"❌ LỖI: Không tìm thấy file '{INPUT_TRAIN}'")
    print("👉 Hãy chạy file 'src1/3.5.py' trước để tạo dữ liệu cân bằng.")
    exit()

print("--- BẮT ĐẦU GIAI ĐOẠN 4: LAZY PREDICT (LOCAL SELECTION) ---")
train_df = pd.read_csv(INPUT_TRAIN)
print(f"✅ Đã tải tập Train cân bằng: {train_df.shape}")

# Danh sách các đặc trưng đầu vào (Chỉ lấy các cột số đã encode)
features = ['Age', 'BMI', 'Education_Code', 'Income_Code', 'Job_Code', 'Marital_Code',
            'Meno_Duration_New', 'Meno_Status', 'Chronic_Disease']

targets = ['PSS_Score', 'MENQOL_Score']


def get_models():
    """Trả về danh sách mô hình ứng viên"""
    return {
        'Lasso': Lasso(random_state=42),
        'Ridge': Ridge(random_state=42),
        'ElasticNet': ElasticNet(random_state=42),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(),
        'KNN': KNeighborsRegressor(),
        'GradientBoosting': GradientBoostingRegressor(random_state=42)
    }


# ==============================================================================
# CHẠY SÀNG LỌC (SCREENING LOOP)
# ==============================================================================
results = []

# Khởi tạo scaler để chuẩn hóa dữ liệu đầu vào
scaler = StandardScaler()

for c_id in sorted(train_df['Cluster'].unique()):
    print(f"\n" + "=" * 50)
    print(f"🔰 CLUSTER {c_id}")

    c_data = train_df[train_df['Cluster'] == c_id]
    n_samples = len(c_data)
    print(f"   -> Số lượng mẫu sau Oversampling: {n_samples}")

    candidate_models = get_models()

    for target in targets:
        print(f"\n   🎯 Mục tiêu: {target}")
        best_score = -np.inf
        best_model_name = "None"

        # Chuẩn bị X, y
        X = c_data[features]
        y = c_data[target]

        # CHUẨN HÓA DỮ LIỆU CỤC BỘ (Quan trọng cho SVR/KNN)
        X_scaled = scaler.fit_transform(X)

        print(f"      |{'Model':<20}|{'R2 (CV)':<10}|{'MAE':<10}|")
        print(f"      |{'-' * 20}|{'-' * 10}|{'-' * 10}|")

        for name, model in candidate_models.items():
            try:
                # Dùng CV=5 vì dữ liệu bây giờ đã đủ lớn nhờ SMOTE
                cv = KFold(n_splits=5, shuffle=True, random_state=42)

                cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='r2')
                r2 = cv_scores.mean()

                mae_scores = -cross_val_score(model, X_scaled, y, cv=cv, scoring='neg_mean_absolute_error')
                mae = mae_scores.mean()

                print(f"      |{name:<20}|{r2:<10.4f}|{mae:<10.4f}|")

                if r2 > best_score:
                    best_score = r2
                    best_model_name = name
                    best_mae = mae

            except Exception as e:
                pass

        print(f"      🏆 VÔ ĐỊCH: {best_model_name} (R2={best_score:.4f})")

        results.append({
            'Cluster': c_id,
            'Target': target,
            'Best_Model': best_model_name,
            'Best_R2': best_score,
            'Best_MAE': best_mae,
            'Samples': n_samples
        })

# ==============================================================================
# TỔNG HỢP BÁO CÁO
# ==============================================================================
print("\n" + "=" * 60)
print("📊 BẢNG TỔNG SẮP MÔ HÌNH TỐT NHẤT (LEADERBOARD)")
print("=" * 60)
report_df = pd.DataFrame(results)
print(report_df)

report_df.to_csv(OUTPUT_REPORT, index=False)
print(f"\n✅ Đã lưu báo cáo tại: {OUTPUT_REPORT}")