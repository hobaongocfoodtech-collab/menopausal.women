import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import ExtraTreesRegressor
import warnings

warnings.filterwarnings('ignore')

# 1. NẠP DỮ LIỆU (Sửa lỗi NameError)
FILE_PATH = r"D:\PycharmProjects\PNMK\icatsd2026_menopause_qol\data\processed\clean_data_final.csv"


def run_diagnostic():
    try:
        df = pd.read_csv(FILE_PATH)

        # Định nghĩa các biến cần thiết
        demographics = ['Age', 'BMI', 'Education_Code', 'Income_Code', 'Marital_Code',
                        'Job_Code', 'Info_Search_Code', 'Meno_Duration', 'Meno_Group']

        # Xác định lại Gold Questions (Giả sử lấy 4 câu quan trọng nhất)
        pool_pss = [c for c in df.columns if c.startswith('PSS_') and c != 'PSS_Score']
        pool_men = [c for c in df.columns if c.startswith('MEN_') and c != 'MENQOL_Score']

        print("-" * 60)
        print("BẮT ĐẦU CHẨN ĐOÁN OVERFITTING")
        print("-" * 60)

        def check_model(target_name):
            # Tìm 4 câu hỏi vàng cho mục tiêu này
            selector = ExtraTreesRegressor(n_estimators=100, random_state=42)
            selector.fit(df[pool_pss + pool_men], df[target_name])
            gold_q = pd.Series(selector.feature_importances_, index=pool_pss + pool_men).nlargest(4).index.tolist()

            X = df[demographics + gold_q]
            y = df[target_name]

            # --- MÔ HÌNH 1: EXTRA TREES MẶC ĐỊNH (Dễ Overfit) ---
            model_overfit = ExtraTreesRegressor(n_estimators=100, random_state=42)

            # --- MÔ HÌNH 2: EXTRA TREES ĐÃ CẮT TỈA (Chống Overfit) ---
            # max_depth=5: Giới hạn độ sâu của cây
            # min_samples_leaf=5: Mỗi lá phải có ít nhất 5 mẫu mới được tách
            model_tuned = ExtraTreesRegressor(n_estimators=100, max_depth=5, min_samples_leaf=5, random_state=42)

            def get_scores(m):
                m.fit(X, y)
                train_r2 = m.score(X, y)
                cv = KFold(n_splits=5, shuffle=True, random_state=42)
                test_r2 = cross_val_score(m, X, y, cv=cv).mean()
                return train_r2, test_r2

            tr_o, te_o = get_scores(model_overfit)
            tr_t, te_t = get_scores(model_tuned)

            print(f"\n[MỤC TIÊU: {target_name}]")
            print(f"{'Thông số':<25} | {'Mô hình Gốc':<15} | {'Mô hình Tinh chỉnh'}")
            print(f"{'-' * 60}")
            print(f"{'R2 Train (Học vẹt)':<25} | {tr_o:15.4f} | {tr_t:.4f}")
            print(f"{'R2 CV (Thực tế)':<25} | {te_o:15.4f} | {te_t:.4f}")
            print(f"{'Độ lệch (Gap)':<25} | {tr_o - te_o:15.4f} | {tr_t - te_t:.4f}")

        check_model('PSS_Score')
        check_model('MENQOL_Score')

    except Exception as e:
        print(f"LỖI: {e}")


if __name__ == "__main__":
    run_diagnostic()