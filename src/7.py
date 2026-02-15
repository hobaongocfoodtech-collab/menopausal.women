import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# 1. CẤU HÌNH ĐƯỜNG DẪN
FILE_PATH = r"C:\Users\Admin\PycharmProjects\PNMK\icatsd2026_menopause_qol\data\processed\clean_data_refined.csv"

try:
    # 2. TẢI DỮ LIỆU
    df = pd.read_csv(FILE_PATH)

    # 3. ĐỊNH NGHĨA BIẾN
    demographics = ['Age', 'BMI', 'Education_Code', 'Income_Code', 'Marital_Code',
                    'Job_Code', 'Info_Search_Code', 'Meno_Duration', 'Meno_Group']
    targets = ['PSS_Score', 'MENQOL_Score']

    X = df[demographics]
    y = df[targets]

    # 4. CHIA DỮ LIỆU CÓ PHÂN TẦNG (STRATIFIED SPLIT)
    # Binning PSS_Score để đảm bảo phân bổ nặng/nhẹ đều ở cả 2 tập
    y_bins = pd.qcut(df['PSS_Score'], q=4, labels=False, duplicates='drop')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y_bins
    )

    print(f"--- ĐÃ CHIA DỮ LIỆU (80/20) ---")
    print(f"Tập Train: {len(X_train)} mẫu | Tập Test: {len(X_test)} mẫu\n")


    def analyze_model_errors(target_name, X_tr, X_te, y_tr, y_te):
        # Huấn luyện mô hình Baseline (ExtraTrees)
        model = ExtraTreesRegressor(n_estimators=100, random_state=42)
        model.fit(X_tr, y_tr[target_name])

        y_pred = model.predict(X_te)
        y_actual = y_te[target_name].values

        # --- BẮT ĐẦU TRỰC QUAN HÓA SAI SỐ ---
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        # Biểu đồ 1: Actual vs Predicted (Đường chéo lý tưởng)
        sns.scatterplot(x=y_actual, y=y_pred, ax=axes[0], alpha=0.7, color='teal')
        lims = [min(y_actual.min(), y_pred.min()), max(y_actual.max(), y_pred.max())]
        axes[0].plot(lims, lims, 'r--', lw=2, label='Ideal (y=x)')
        axes[0].set_title(f'{target_name}: Actual vs Predicted')
        axes[0].set_xlabel('Actual Value')
        axes[0].set_ylabel('Predicted Value')
        axes[0].legend()

        # Biểu đồ 2: Residual Plot (Phân phối sai số)
        residuals = y_actual - y_pred
        sns.histplot(residuals, kde=True, ax=axes[1], color='purple')
        axes[1].axvline(0, color='red', linestyle='--')
        axes[1].set_title(f'{target_name}: Error Distribution (Residuals)')
        axes[1].set_xlabel('Error (Actual - Predicted)')

        # Biểu đồ 3: Error Gap per Sample (Theo dõi từng người)
        indices = np.arange(len(y_actual))
        axes[2].plot(indices, y_actual, label='Actual', marker='o', alpha=0.8, color='#1f77b4')
        axes[2].plot(indices, y_pred, label='Predicted', marker='x', ls='--', color='#ff7f0e')
        axes[2].fill_between(indices, y_actual, y_pred, color='gray', alpha=0.2, label='Error Gap')
        axes[2].set_title(f'{target_name}: Sample-wise Error Gap')
        axes[2].set_xlabel('Sample Index (Test Set)')
        axes[2].legend()

        plt.suptitle(f"BASELINE ERROR ANALYSIS FOR {target_name.upper()}", fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        # In các chỉ số cơ bản
        print(f"[{target_name}] R-Squared: {r2_score(y_actual, y_pred):.4f}")
        print(f"[{target_name}] RMSE: {np.sqrt(mean_squared_error(y_actual, y_pred)):.4f}")


    # Thực hiện phân tích cho cả 2 chỉ số
    analyze_model_errors('PSS_Score', X_train, X_test, y_train, y_test)
    analyze_model_errors('MENQOL_Score', X_train, X_test, y_train, y_test)

except Exception as e:
    print(f"Lỗi: {e}")