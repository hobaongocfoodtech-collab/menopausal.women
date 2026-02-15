import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
import os

warnings.filterwarnings('ignore')

# 1. CẤU HÌNH DỮ LIỆU
FILE_PATH = r"/menopause_qol\data\processed\clean_data_final.csv"

# --- ĐỊNH NGHĨA NHÓM BIẾN ---
DEMO_FEATS = ['Age', 'BMI', 'Education_Code', 'Income_Code', 'Marital_Code',
              'Job_Code', 'Info_Search_Code', 'Meno_Duration', 'Meno_Group']

# Danh sách các câu hỏi tiềm năng
POTENTIAL_FEATS = [
    'MEN_HotFlash', 'MEN_Memory', 'MEN_Sleep', 'MEN_Headache', 'MEN_MuscleAche',
    'MEN_Fatigue', 'MEN_Weight', 'MEN_Skin', 'MEN_Depressed', 'MEN_Impatient',
    'MEN_Urine', 'MEN_Libido',
    'PSS_1', 'PSS_2', 'PSS_3', 'PSS_4', 'PSS_5', 'PSS_6', 'PSS_7', 'PSS_8', 'PSS_9', 'PSS_10',
    'Chronic_Disease', 'Supp_Calcium', 'Supp_Omega3', 'Exercise'
]

def run_adaptive_pipeline():
    if not os.path.exists(FILE_PATH):
        print(f"❌ LỖI: Không tìm thấy file {FILE_PATH}")
        return

    df = pd.read_csv(FILE_PATH)

    # --- BƯỚC SỬA LỖI: MÃ HÓA DỮ LIỆU DẠNG CHỮ SANG SỐ ---
    print("\n[PRE-PROCESSING] Đang xử lý dữ liệu dạng chữ...")

    # 1. Xử lý Bệnh mãn tính (Chronic_Disease)
    # Nếu ô chứa chữ (khác nan/trống) -> 1 (Có bệnh), ngược lại -> 0 (Không)
    if 'Chronic_Disease' in df.columns:
        df['Chronic_Disease'] = df['Chronic_Disease'].apply(
            lambda x: 0 if pd.isna(x) or str(x).strip().lower() in ['không', 'nan', '0'] else 1
        )
        print("-> Đã mã hóa 'Chronic_Disease' sang 0/1.")

    # 2. Xử lý Thực phẩm chức năng & Tập thể dục (Supp_..., Exercise)
    # Giả định dữ liệu là 'Có'/'Không' hoặc text -> chuyển sang 1/0
    cols_to_binary = ['Supp_Calcium', 'Supp_Omega3', 'Exercise']
    for c in cols_to_binary:
        if c in df.columns:
            df[c] = df[c].apply(
                lambda x: 1 if str(x).strip().lower() in ['có', 'yes', '1'] else 0
            )
            print(f"-> Đã mã hóa '{c}' sang 0/1.")

    # Kiểm tra lại xem còn cột nào trong POTENTIAL_FEATS là dạng chữ không
    # Nếu còn, ép sang số hoặc loại bỏ
    valid_potential = []
    for c in POTENTIAL_FEATS:
        if c in df.columns:
            # Ép kiểu sang số, nếu lỗi biến thành NaN rồi điền 0
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            valid_potential.append(c)

    print(f"✅ Đã làm sạch số liệu. Số lượng biến tiềm năng hợp lệ: {len(valid_potential)}")

    # ==============================================================================
    # BƯỚC 1: CLUSTERING & SPLIT DATA
    # ==============================================================================
    print("\n" + "="*80)
    print("BƯỚC 1: PHÂN CỤM & CHIA TẬP DỮ LIỆU (STRATIFIED SPLIT)")
    print("="*80)

    scaler = StandardScaler()
    X_cluster = scaler.fit_transform(df[DEMO_FEATS].fillna(0))

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_cluster)

    print("\n📊 HỒ SƠ ĐẶC TRƯNG CỦA TỪNG NHÓM (CLUSTER PROFILE):")
    profile = df.groupby('Cluster')[DEMO_FEATS].mean().T
    print(profile.round(2))
    print("-" * 60)
    print("-> GIẢI THÍCH NHANH:")
    for i in range(3):
        age = profile.loc['Age', i]
        income = profile.loc['Income_Code', i]
        print(f"   + Cluster {i}: Độ tuổi TB {age:.1f}, Thu nhập TB mức {income:.1f}")

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Cluster'])
    print(f"\n✅ Đã chia dữ liệu: Train ({len(train_df)}) - Test ({len(test_df)})")

    # ==============================================================================
    # BƯỚC 2 & 3: TÌM 'CÂU HỎI VÀNG' RIÊNG BIỆT CHO TỪNG NHÓM
    # ==============================================================================
    print("\n" + "="*80)
    print("BƯỚC 2 & 3: TÌM 'CÂU HỎI VÀNG' RIÊNG BIỆT CHO TỪNG NHÓM")
    print("="*80)

    targets = ['PSS_Score', 'MENQOL_Score']

    for target in targets:
        print(f"\n>>> ĐANG XÂY DỰNG CHIẾN LƯỢC CHO MỤC TIÊU: {target}")

        y_test_all = []
        y_pred_all = []

        for cluster_id in range(3):
            print(f"\n   [CLUSTER {cluster_id}] Phân tích đặc trưng...")

            c_train = train_df[train_df['Cluster'] == cluster_id]
            c_test = test_df[test_df['Cluster'] == cluster_id]

            if len(c_train) < 5:
                print("      ⚠️ Quá ít dữ liệu, bỏ qua.")
                continue

            # Feature Selection
            selector = ExtraTreesRegressor(n_estimators=100, random_state=42)
            X_select = c_train[DEMO_FEATS + valid_potential].fillna(0)
            y_select = c_train[target]

            selector.fit(X_select, y_select)

            importances = pd.Series(selector.feature_importances_, index=X_select.columns)
            potential_only = importances[importances.index.isin(valid_potential)]
            top_5_questions = potential_only.nlargest(5).index.tolist()

            print(f"      -> 5 Câu hỏi vàng: {top_5_questions}")

            # ==========================================================================
            # BƯỚC 4: FINE-TUNING
            # ==========================================================================
            final_features = DEMO_FEATS + top_5_questions
            X_train_fold = c_train[final_features].fillna(0)
            y_train_fold = c_train[target]

            # Tinh chỉnh nhẹ nhàng để tránh overfit trên tập con
            param_grid = {
                'n_estimators': [100],
                'max_depth': [3, 5],
                'min_samples_leaf': [2, 4]
            }

            grid = GridSearchCV(ExtraTreesRegressor(random_state=42), param_grid, cv=3, n_jobs=-1)
            grid.fit(X_train_fold, y_train_fold)

            best_model = grid.best_estimator_
            # print(f"      -> Best Params: {grid.best_params_}") # Tắt bớt log cho gọn

            # DỰ BÁO
            X_test_fold = c_test[final_features].fillna(0)
            y_test_fold = c_test[target]
            y_pred = best_model.predict(X_test_fold)

            y_test_all.extend(y_test_fold)
            y_pred_all.extend(y_pred)

        # ==============================================================================
        # BƯỚC 5: ĐÁNH GIÁ
        # ==============================================================================
        print("\n" + "-"*60)
        print(f"KẾT QUẢ CUỐI CÙNG (MỤC TIÊU: {target})")
        r2_final = r2_score(y_test_all, y_pred_all)
        mae_final = mean_absolute_error(y_test_all, y_pred_all)

        print(f"✅ R-Squared (Độ chính xác): {r2_final:.4f}")
        print(f"✅ MAE (Sai số trung bình): {mae_final:.4f}")

        # Vẽ biểu đồ
        plt.figure(figsize=(6, 6))
        sns.scatterplot(x=y_test_all, y=y_pred_all, color='blue', alpha=0.6)
        plt.plot([min(y_test_all), max(y_test_all)], [min(y_test_all), max(y_test_all)], 'r--')
        plt.title(f'{target}: Thực tế vs Dự báo (Adaptive AI)')
        plt.xlabel('Thực tế')
        plt.ylabel('Dự báo')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    run_adaptive_pipeline()