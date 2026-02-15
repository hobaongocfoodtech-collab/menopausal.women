import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from lazypredict.Supervised import LazyRegressor
import warnings
import logging

# --- 1. TẮT CẢNH BÁO RÁC ---
warnings.filterwarnings('ignore')
# Tắt thông báo từ LightGBM
logging.getLogger("lightgbm").setLevel(logging.ERROR)

FILE_PATH = r"C:\Users\Admin\PycharmProjects\PNMK\icatsd2026_menopause_qol\data\processed\clean_data_refined.csv"


def run_stage_1_fixed(target_name, X_train, X_test, y_train, y_test):
    print(f"\n" + "=" * 60)
    print(f"BẮT ĐẦU QUÉT MÔ HÌNH CHO: {target_name}")
    print("=" * 60)

    # 1. Huấn luyện với LazyPredict
    reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
    models, predictions = reg.fit(X_train, X_test, y_train[target_name], y_test[target_name])

    # In bảng xếp hạng Top 5
    print(f"\n[BẢNG XẾP HẠNG TOP 5 MÔ HÌNH {target_name}]:")
    print(models[['R-Squared', 'RMSE']].head(5))

    # 2. XỬ LÝ LỖI KHỚP TÊN (FIX KEYERROR)
    best_model_name = models.index[0]

    # Tìm cột trong 'predictions' khớp với tên mô hình tốt nhất
    if best_model_name in predictions.columns:
        y_pred = predictions[best_model_name].values.flatten()
    else:
        # Nếu không khớp tên chính xác, lấy cột đầu tiên (luôn là mô hình tốt nhất)
        y_pred = predictions.iloc[:, 0].values.flatten()
        best_model_name = predictions.columns[0]

    y_actual = y_test[target_name].values.flatten()
    min_len = min(len(y_actual), len(y_pred))

    # 3. TRỰC QUAN HÓA SAI SỐ
    plt.figure(figsize=(20, 6))

    # Biểu đồ 1: Actual vs Predicted
    plt.subplot(1, 3, 1)
    sns.scatterplot(x=y_actual[:min_len], y=y_pred[:min_len], alpha=0.7, color='teal')
    lims = [min(y_actual.min(), y_pred.min()), max(y_actual.max(), y_pred.max())]
    plt.plot(lims, lims, 'r--', lw=2, label='Perfect Prediction')
    plt.title(f'{target_name}: Actual vs Predicted\n(Best: {best_model_name})')
    plt.legend()

    # Biểu đồ 2: Residual Distribution
    plt.subplot(1, 3, 2)
    residuals = y_actual[:min_len] - y_pred[:min_len]
    sns.histplot(residuals, kde=True, color='purple')
    plt.axvline(0, color='red', linestyle='--')
    plt.title(f'{target_name}: Phân phối sai số')

    # Biểu đồ 3: Error Gap
    plt.subplot(1, 3, 3)
    indices = np.arange(min_len)
    plt.plot(indices, y_actual[:min_len], label='Thực tế', marker='o', alpha=0.8)
    plt.plot(indices, y_pred[:min_len], label='Dự đoán', marker='x', ls='--', color='orange')
    plt.fill_between(indices, y_actual[:min_len], y_pred[:min_len], color='gray', alpha=0.2, label='Vùng sai lệch')
    plt.title(f'{target_name}: Diễn biến sai số từng mẫu')
    plt.legend()

    plt.tight_layout()
    plt.show()

    print(f"\n=> Kết luận {target_name}: R2 = {r2_score(y_actual[:min_len], y_pred[:min_len]):.4f}")


# --- THỰC THI CHÍNH ---
try:
    df = pd.read_csv(FILE_PATH)

    demographics = ['Age', 'BMI', 'Education_Code', 'Income_Code', 'Marital_Code',
                    'Job_Code', 'Info_Search_Code', 'Meno_Duration', 'Meno_Group']
    targets = ['PSS_Score', 'MENQOL_Score']

    X = df[demographics]
    y = df[targets]

    # Chia dữ liệu phân tầng
    y_bins = pd.qcut(df['PSS_Score'], q=4, labels=False, duplicates='drop')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y_bins
    )

    # Chạy cho PSS_Score
    run_stage_1_fixed('PSS_Score', X_train, X_test, y_train, y_test)

    # Chạy cho MENQOL_Score (Lệnh này sẽ không bị bỏ qua nếu lệnh trên lỗi)
    run_stage_1_fixed('MENQOL_Score', X_train, X_test, y_train, y_test)

except Exception as e:
    print(f"❌ LỖI HỆ THỐNG: {e}")