import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore')

# --- CẤU HÌNH ---
MODEL_PATH = r"/menopause_qol\src1\final_health_advisor.pkl"
TEST_FILE = "test_data.csv"
OUTPUT_DIR = r"/menopause_qol\reports\xai_shap"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Bộ từ điển để hiển thị tên câu hỏi tiếng Việt trên biểu đồ
QUESTION_MAP = {
    'PSS_1': 'Kho khan doi mat van de',
    'PSS_2': 'Mat kiem soat cuoc song',
    'PSS_3': 'Cang thang lo lang',
    'PSS_10': 'Kho khan tich tu qua muc',
    'MEN_HotFlash': 'Boc hoa',
    'MEN_Memory': 'Giam tri nho',
    'MEN_Sleep': 'Mat ngu',
    'MEN_Depressed': 'Lo au tram cam',
    'MEN_Impatient': 'Mat kien nhan',
    'MEN_Libido': 'Giam ham muon',
    'MEN_Fatigue': 'Met moi',
    'Age': 'Tuoi',
    'BMI': 'Chi so BMI'
}


def run_shap_analysis():
    if not os.path.exists(MODEL_PATH):
        print("Khong tim thay file model!")
        return

    package = joblib.load(MODEL_PATH)
    test_df = pd.read_csv(TEST_FILE)
    experts = package['experts']
    strategy = package['strategy']
    demo_feats = package['demo_feats']

    print("--- DANG THUC HIEN GIAI DOAN 12: GIAI THICH MO HINH (SHAP) ---")

    # Chung ta se phan tich Cluster 1 (Nhom co mau thu lon nhat) de lam vi du dien hinh
    c_id = 1
    target_types = ['PSS', 'MEN']

    for t_type in target_types:
        print(f"\n[+] Dang phan tich SHAP cho: {t_type} (Cluster {c_id})")

        # 1. Lay model va du lieu tuong ung
        model = experts[f"expert_{c_id}_{t_type}"]
        features = demo_feats + strategy[c_id][t_type]

        X_test = test_df[test_df['Cluster'] == c_id][features]

        # Doi ten cot sang tieng Viet de bieu do dep hon
        X_test_named = X_test.rename(columns=QUESTION_MAP)

        # 2. Khoi tao SHAP Explainer (TreeExplainer dung cho ExtraTrees)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_named)

        # 3. VE BIEU DO TONG QUAN (Summary Plot)
        # Bieu do nay thay the Feature Importance cua Sklearn
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test_named, show=False)
        plt.title(f"SHAP Summary Plot - {t_type} (Cluster {c_id})")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"shap_summary_{t_type}_c{c_id}.png"))
        plt.close()

        # 4. VE BIEU DO CHO 1 BENH NHAN CU THE (Waterfall Plot)
        # Chon benh nhan dau tien trong tap test cua cluster 1
        plt.figure(figsize=(10, 6))
        # Base value la gia tri du bao trung binh
        # shap_values[0] la dong gop cua tung bien cho benh nhan nay
        explanation = shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=X_test_named.iloc[0],
            feature_names=X_test_named.columns
        )
        shap.plots.waterfall(explanation, show=False)
        plt.title(f"SHAP Waterfall - Phan tich ca benh cu the ({t_type})")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"shap_waterfall_{t_type}_c{c_id}_user0.png"))
        plt.close()

    print(f"\nHoan tat! Bieu do SHAP da duoc luu tai: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_shap_analysis()