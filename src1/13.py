import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
import os

# --- CẤU HÌNH ---
MODEL_PATH = r"D:\PycharmProjects\PNMK\icatsd2026_menopause_qol\src1\final_health_advisor.pkl"
TEST_FILE = "test_data.csv"
OUTPUT_DIR = r"D:\PycharmProjects\PNMK\icatsd2026_menopause_qol\reports\error_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_error_analysis():
    # 1. Tải mô hình và dữ liệu test
    package = joblib.load(MODEL_PATH)
    test_df = pd.read_csv(TEST_FILE)

    scaler = package['scaler']
    kmeans = package['kmeans']
    experts = package['experts']
    strategy = package['strategy']
    demo_feats = package['demo_feats']

    print("--- ĐANG THỰC HIỆN GIAI ĐOẠN 13: PHÂN TÍCH SAI SỐ & TRƯỜNG HỢP BIÊN ---")

    all_results = []

    # 2. Dự báo và tính sai số dư (Residuals)
    for _, row in test_df.iterrows():
        user_demo_df = pd.DataFrame([row[demo_feats]])
        user_demo_scaled = scaler.transform(user_demo_df)
        c_id = kmeans.predict(user_demo_scaled)[0]

        res_item = {'Actual_Cluster': row['Cluster'], 'AI_Cluster': c_id}

        for t_type in ['PSS', 'MEN']:
            target_col = 'PSS_Score' if t_type == 'PSS' else 'MENQOL_Score'
            req_feats = demo_feats + strategy[c_id][t_type]

            input_df = pd.DataFrame([row[req_feats]])
            pred = experts[f"expert_{c_id}_{t_type}"].predict(input_df)[0]

            actual = row[target_col]
            residual = actual - pred  # Sai số dư

            res_item[f'{t_type}_Actual'] = actual
            res_item[f'{t_type}_Pred'] = pred
            res_item[f'{t_type}_Error'] = residual
            res_item[f'{t_type}_AbsError'] = abs(residual)

        all_results.append(res_item)

    error_df = pd.DataFrame(all_results)

    # 3. VẼ BIỂU ĐỒ RESIDUAL PLOT
    plt.figure(figsize=(16, 6))

    # Biểu đồ PSS Residuals
    plt.subplot(1, 2, 1)
    sns.scatterplot(x='PSS_Pred', y='PSS_Error', data=error_df, hue='AI_Cluster', palette='viridis', s=100)
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Residual Plot - PSS Score\n(Dưới 0: AI đoán cao hơn thực tế | Trên 0: AI đoán thấp hơn)')
    plt.xlabel('Giá trị Dự báo')
    plt.ylabel('Sai số dư (Actual - Predicted)')

    # Biểu đồ MENQOL Residuals
    plt.subplot(1, 2, 2)
    sns.scatterplot(x='MEN_Pred', y='MEN_Error', data=error_df, hue='AI_Cluster', palette='magma', s=100)
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Residual Plot - MENQOL Score')
    plt.xlabel('Giá trị Dự báo')
    plt.ylabel('Sai số dư')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "residual_plots.png"))
    plt.close()

    # 4. TRÍCH XUẤT CÁC TRƯỜNG HỢP SAI SỐ LỚN NHẤT (TOP OUTLIERS)
    print("\n⚠️ CÁC TRƯỜNG HỢP AI ĐOÁN SAI NHIỀU NHẤT (TOP 5 OUTLIERS - PSS):")
    top_errors_pss = error_df.nlargest(5, 'PSS_AbsError')
    print(top_errors_pss[['Actual_Cluster', 'AI_Cluster', 'PSS_Actual', 'PSS_Pred', 'PSS_Error']])

    # 5. PHÂN TÍCH ĐẶC ĐIỂM CỤM 0 (NHÓM THIỂU SỐ)
    c0_errors = error_df[error_df['Actual_Cluster'] == 0]
    if not c0_errors.empty:
        print("\n📍 PHÂN TÍCH RIÊNG CỤM 0 (Tiền mãn kinh):")
        mae_c0_pss = c0_errors['PSS_AbsError'].mean()
        mae_c0_men = c0_errors['MEN_AbsError'].mean()
        print(f"   - MAE PSS (Cluster 0): {mae_c0_pss:.4f}")
        print(f"   - MAE MEN (Cluster 0): {mae_c0_men:.4f}")

    error_df.to_csv(os.path.join(OUTPUT_DIR, "detailed_error_report.csv"), index=False)
    print(f"\n✅ Hoàn tất! Báo cáo sai số lưu tại: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_error_analysis()