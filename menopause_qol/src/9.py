import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings('ignore')

# 1. Cấu hình
FILE_PATH = r"C:\Users\Admin\PycharmProjects\PNMK\icatsd2026_menopause_qol\data\processed\clean_data_refined.csv"

try:
    df = pd.read_csv(FILE_PATH)

    # Định nghĩa các bộ câu hỏi
    demographics = ['Age', 'BMI', 'Education_Code', 'Income_Code', 'Marital_Code', 'Job_Code', 'Info_Search_Code',
                    'Meno_Duration', 'Meno_Group']
    pool_pss = [c for c in df.columns if c.startswith('PSS_') and c != 'PSS_Score']
    pool_men = [c for c in df.columns if c.startswith('MEN_') and c != 'MENQOL_Score']

    # --- BƯỚC 1: PHÂN CỤM DỰA TRÊN 9 CÂU NHÂN KHẨU HỌC ---
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(df[demographics].fillna(0))


    # --- BƯỚC 2: HÀM FINE-TUNING VỚI m, n CÂU HỎI VÀNG ---
    def adaptive_fine_tuning(target_name, models_dict):
        print(f"\n--- TINH CHỈNH THÍCH NGHI CHO: {target_name} ---")

        # Tìm m=2 câu PSS và n=2 câu MENQOL quan trọng nhất toàn cục (để demo)
        # Trong thực tế, bạn có thể làm điều này cho từng cụm
        selector = ExtraTreesRegressor(n_estimators=100, random_state=42)
        all_pool = pool_pss + pool_men
        selector.fit(df[all_pool], df[target_name])

        feat_imp = pd.Series(selector.feature_importances_, index=all_pool)
        gold_questions = feat_imp.nlargest(4).index.tolist()  # m+n = 4

        print(f"Các câu hỏi vàng được máy chọn thêm: {gold_questions}")

        # Cập nhật bộ Features
        X_new = df[demographics + gold_questions]
        y = df[target_name]

        # Chia lại Train/Test để kiểm chứng
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)

        for name, model in models_dict.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            print(f"Model {name:25} -> R2 sau Fine-tuning: {r2:.4f}")


    # 4 Mô hình chiến lược
    selected_models = {
        'KNeighbors': KNeighborsRegressor(n_neighbors=5),
        'OMP_CV': OrthogonalMatchingPursuitCV(),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(random_state=42)
    }

    # Chạy Fine-tuning
    adaptive_fine_tuning('PSS_Score', selected_models)
    adaptive_fine_tuning('MENQOL_Score', selected_models)

except Exception as e:
    print(f"Lỗi: {e}")