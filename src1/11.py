import pandas as pd
import numpy as np
import pingouin as pg
import os
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Tắt cảnh báo
warnings.filterwarnings('ignore')

# --- CẤU HÌNH ---
FILE_PATH = r"D:\PycharmProjects\PNMK\icatsd2026_menopause_qol\data\processed\clean_data_final.csv"
REPORT_PATH = r"D:\PycharmProjects\PNMK\icatsd2026_menopause_qol\reports\statistical_pingouin_report.xlsx"

# Tạo thư mục báo cáo nếu chưa có
os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)


def run_pingouin_validation():
    if not os.path.exists(FILE_PATH):
        print(f"❌ Lỗi: Không tìm thấy file {FILE_PATH}")
        return

    # 1. Đọc dữ liệu và kiểm tra cột
    df = pd.read_csv(FILE_PATH)
    print(f"✅ Đã tải dữ liệu: {df.shape}")

    # Tự động tìm tên cột đúng (tránh lỗi KeyError)
    potential_meno_cols = [c for c in df.columns if 'Meno_Duration' in c]
    meno_col = potential_meno_cols[0] if potential_meno_cols else None

    # Danh sách các cột cần thiết cho thống kê mô tả
    target_cols = ['Age', 'BMI', 'PSS_Score', 'MENQOL_Score']
    if meno_col: target_cols.append(meno_col)

    # 2. Tự động tái lập Cluster (vì file final.csv chưa có nhãn Cluster)
    print("🔄 Đang tái lập phân cụm để phân tích thống kê...")
    demo_feats = ['Age', 'BMI', 'Education_Code', 'Income_Code', 'Job_Code', 'Marital_Code']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[demo_feats])
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # 3. Thực hiện phân tích và lưu vào Excel
    try:
        with pd.ExcelWriter(REPORT_PATH, engine='openpyxl') as writer:
            # --- Sheet 1: Thống kê mô tả ---
            desc = df[target_cols].describe().T
            desc.to_excel(writer, sheet_name='Descriptive_Stats')

            # --- Sheet 2: ANOVA cho MENQOL ---
            aov_men = pg.anova(data=df, dv='MENQOL_Score', between='Cluster', detailed=True)
            aov_men.to_excel(writer, sheet_name='ANOVA_MENQOL')

            # --- Sheet 3: Post-hoc (So sánh cặp giữa các cụm) ---
            posthoc = pg.pairwise_tukey(data=df, dv='MENQOL_Score', between='Cluster')
            posthoc.to_excel(writer, sheet_name='PostHoc_Tukey_MENQOL')

            # --- Sheet 4: Tương quan đa biến (Spearman) ---
            # Chỉ lấy các cột số
            num_df = df.select_dtypes(include=[np.number])
            corrs = pg.pairwise_corr(num_df, columns=['Age', 'BMI', 'PSS_Score', 'MENQOL_Score'], method='spearman')
            corrs.to_excel(writer, sheet_name='Correlation_Analysis')

            print(f"✅ HOÀN TẤT! Báo cáo đã lưu tại: {REPORT_PATH}")

    except Exception as e:
        print(f"❌ Lỗi trong quá trình xử lý: {e}")


if __name__ == "__main__":
    run_pingouin_validation()