import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error
import os
import warnings

# Tắt các cảnh báo không cần thiết để đầu ra sạch sẽ
warnings.filterwarnings('ignore', category=UserWarning)

# --- CẤU HÌNH ---
MODEL_PATH = "final_health_advisor.pkl"
TEST_FILE = "test_data.csv"

if not os.path.exists(MODEL_PATH) or not os.path.exists(TEST_FILE):
    print("❌ LỖI: Thiếu file model .pkl hoặc file test_data.csv.")
    exit()


def run_visual_test():
    # 1. Tải hệ thống và dữ liệu
    package = joblib.load(MODEL_PATH)
    test_df = pd.read_csv(TEST_FILE)

    scaler = package['scaler']
    kmeans = package['kmeans']
    experts = package['experts']
    strategy = package['strategy']
    demo_feats = package['demo_feats']

    print(f"--- ĐANG KIỂM THỬ HỆ THỐNG TRÊN {len(test_df)} MẪU TEST ---")

    results = []

    # 2. Duyệt qua từng dòng dữ liệu
    for _, row in test_df.iterrows():
        # SỬA LỖI: Chuyển sang DataFrame để giữ tên cột (Feature names)
        user_demo_df = pd.DataFrame([row[demo_feats]])
        user_demo_scaled = scaler.transform(user_demo_df)

        # Bước 1: Xác định Cụm
        c_id = kmeans.predict(user_demo_scaled)[0]

        res_row = {'Cluster_Real': row['Cluster'], 'Cluster_AI': c_id}

        for target_type in ['PSS', 'MEN']:
            target_col = 'PSS_Score' if target_type == 'PSS' else 'MENQOL_Score'
            required_feats = demo_feats + strategy[c_id][target_type]

            # SỬA LỖI: Đưa dữ liệu vào dưới dạng DataFrame có tên cột
            input_df = pd.DataFrame([row[required_feats]])

            # Dự báo
            pred = experts[f"expert_{c_id}_{target_type}"].predict(input_df)[0]

            res_row[f'{target_type}_True'] = row[target_col]
            res_row[f'{target_type}_Pred'] = pred

        results.append(res_row)

    res_df = pd.DataFrame(results)

    # ==============================================================================
    # TRỰC QUAN HÓA
    # ==============================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # PSS_Score
    r2_pss = r2_score(res_df['PSS_True'], res_df['PSS_Pred'])
    mae_pss = mean_absolute_error(res_df['PSS_True'], res_df['PSS_Pred'])

    sns.regplot(x='PSS_True', y='PSS_Pred', data=res_df, ax=ax1,
                scatter_kws={'s': 100, 'alpha': 0.6, 'color': 'teal'},
                line_kws={'color': 'red', 'label': f'R2 = {r2_pss:.4f}'})
    ax1.set_title(f'PSS Score: Thực tế vs AI Dự báo\n(MAE: {mae_pss:.2f})', fontsize=12)
    ax1.set_xlabel('Giá trị thực tế')
    ax1.set_ylabel('Giá trị AI dự báo')
    ax1.legend()

    # MENQOL_Score
    r2_men = r2_score(res_df['MEN_True'], res_df['MEN_Pred'])
    mae_men = mean_absolute_error(res_df['MEN_True'], res_df['MEN_Pred'])

    sns.regplot(x='MEN_True', y='MEN_Pred', data=res_df, ax=ax2,
                scatter_kws={'s': 100, 'alpha': 0.6, 'color': 'orange'},
                line_kws={'color': 'red', 'label': f'R2 = {r2_men:.4f}'})
    ax2.set_title(f'MENQOL Score: Thực tế vs AI Dự báo\n(MAE: {mae_men:.2f})', fontsize=12)
    ax2.set_xlabel('Giá trị thực tế')
    ax2.legend()

    plt.suptitle('KIỂM THỬ HỆ THỐNG ADAPTIVE HEALTH ADVISOR - KẾT QUẢ NGHIỆM THU', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    print("\n📊 BẢNG SO SÁNH CHI TIẾT (5 mẫu đầu tiên):")
    print(res_df[['PSS_True', 'PSS_Pred', 'MEN_True', 'MEN_Pred']].head().round(2))

    print(f"\n✅ TỔNG KẾT TOÀN DIỆN:")
    print(f"   * PSS Score   - R2: {r2_pss:.4f}, MAE: {mae_pss:.4f}")
    print(f"   * MENQOL Score - R2: {r2_men:.4f}, MAE: {mae_men:.4f}")


if __name__ == "__main__":
    run_visual_test()