import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from lazypredict.Supervised import LazyRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import lightgbm as lgb
import warnings
import logging
import os

# --- 1. TẮT CÁC CẢNH BÁO RÁC ---
warnings.filterwarnings('ignore')
# Tắt thông báo spam của LightGBM
logging.getLogger("lightgbm").setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- CẤU HÌNH ---
FILE_PATH = r"/menopause_qol\data\processed\clean_data_final.csv"

FEATURES = ['Age', 'BMI', 'Education_Code', 'Income_Code', 'Marital_Code',
            'Job_Code', 'Info_Search_Code', 'Meno_Duration', 'Meno_Group']

# Lưới tham số (Đã tối ưu để chạy nhanh hơn)
PARAM_GRIDS = {
    'ExtraTreesRegressor': {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'min_samples_leaf': [2, 4],
        'min_samples_split': [2, 5]
    },
    'RandomForestRegressor': {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'min_samples_leaf': [2, 4],
        'max_features': ['sqrt', 'log2']
    },
    'GradientBoostingRegressor': {
        'n_estimators': [100],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 4],
        'subsample': [0.8]
    },
    'XGBRegressor': {
        'n_estimators': [100],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5],
        'subsample': [0.8]
    },
    'LGBMRegressor': {
        'n_estimators': [100],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5],
        'num_leaves': [15, 31],
        'verbose': [-1]  # Tắt log
    },
    'SVR': {'C': [1, 10], 'gamma': ['scale', 0.1], 'kernel': ['rbf']},
    'Ridge': {'alpha': [0.1, 1.0, 10.0]},
    'Lasso': {'alpha': [0.001, 0.01, 0.1]},
    'ElasticNet': {'alpha': [0.01, 0.1], 'l1_ratio': [0.5]},
    'KNeighborsRegressor': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
}

MODEL_MAP = {
    'ExtraTreesRegressor': ExtraTreesRegressor(random_state=42),
    'RandomForestRegressor': RandomForestRegressor(random_state=42),
    'GradientBoostingRegressor': GradientBoostingRegressor(random_state=42),
    'XGBRegressor': xgb.XGBRegressor(random_state=42, verbosity=0),
    'LGBMRegressor': lgb.LGBMRegressor(random_state=42, verbose=-1),
    'SVR': SVR(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'ElasticNet': ElasticNet(),
    'KNeighborsRegressor': KNeighborsRegressor()
}


def run_full_workflow():
    if not os.path.exists(FILE_PATH):
        print(f"❌ LỖI: Không tìm thấy file {FILE_PATH}")
        return

    df = pd.read_csv(FILE_PATH)
    print(f"✅ Đã tải dữ liệu: {df.shape}")

    targets = ['PSS_Score', 'MENQOL_Score']

    for target in targets:
        print("\n" + "#" * 80)
        print(f"🚀 BẮT ĐẦU QUY TRÌNH CHO MỤC TIÊU: {target}")
        print("#" * 80)

        # --- BƯỚC 0: SPLIT DATA ---
        X = df[FEATURES]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"[Bước 0] Split Data: Train={len(X_train)}, Test={len(X_test)}")

        # --- BƯỚC 1: LAZY PREDICT ---
        print("\n[Bước 1] Sàng lọc mô hình với LazyPredict (Mặc định)...")
        reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
        models_lazy, predictions = reg.fit(X_train, X_test, y_train, y_test)

        # Lọc ra các model có trong MODEL_MAP
        top_models_df = models_lazy[models_lazy.index.isin(MODEL_MAP.keys())].head(3)
        top_3_names = top_models_df.index.tolist()

        print(f"\n🏆 Top 3 ứng cử viên sáng giá cho {target}:")
        # --- SỬA LỖI Ở ĐÂY: Đổi 'Time' thành 'Time Taken' ---
        cols_to_show = ['R-Squared', 'RMSE', 'Time Taken']
        # Kiểm tra xem cột nào tồn tại thì in ra
        valid_cols = [c for c in cols_to_show if c in top_models_df.columns]
        print(top_models_df[valid_cols])

        print(f"-> Chọn: {top_3_names}")

        # --- BƯỚC 2: FINE-TUNING ---
        print("\n[Bước 2] Tinh chỉnh tham số & Kiểm tra Overfitting (GridSearch CV)...")

        tuned_results = []

        for name in top_3_names:
            print(f"\n   >>> Đang tối ưu hóa: {name}...")
            base_model = MODEL_MAP[name]
            param_grid = PARAM_GRIDS.get(name, {})

            grid = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=5,
                scoring='r2',
                n_jobs=-1,
                verbose=0
            )

            grid.fit(X_train, y_train)

            best_model = grid.best_estimator_
            best_r2_cv = grid.best_score_
            train_r2 = best_model.score(X_train, y_train)
            gap = train_r2 - best_r2_cv

            print(f"       - Best Params: {grid.best_params_}")
            print(f"       - R2 Train: {train_r2:.4f} | R2 CV (Valid): {best_r2_cv:.4f}")
            print(f"       - Overfit Gap: {gap:.4f}")

            tuned_results.append({
                'Model Name': name,
                'Model Instance': best_model,
                'R2 CV': best_r2_cv
            })

        # --- BƯỚC 3: FINAL EVALUATION ---
        print("\n[Bước 3] So sánh đối kháng trên Tập Test...")

        final_metrics = []

        for item in tuned_results:
            model = item['Model Instance']
            name = item['Model Name']

            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)

            final_metrics.append({
                'Mô hình': name,
                'R2 Test': r2,
                'RMSE Test': rmse,
                'MAE Test': mae
            })

        final_df = pd.DataFrame(final_metrics).sort_values(by='R2 Test', ascending=False)
        print("\n📊 BẢNG XẾP HẠNG CUỐI CÙNG (FINAL LEADERBOARD):")
        print(final_df)

        plt.figure(figsize=(10, 5))
        sns.barplot(x='R2 Test', y='Mô hình', data=final_df, palette='viridis')
        plt.title(f'Hiệu năng thực tế trên tập Test - Mục tiêu: {target}')
        plt.xlim(0, 1)
        plt.tight_layout()
        plt.show()

        best_final_model = final_df.iloc[0]['Mô hình']
        print(f"\n✅ KẾT LUẬN: Mô hình triển khai cho {target} là: {best_final_model}")


if __name__ == "__main__":
    run_full_workflow()