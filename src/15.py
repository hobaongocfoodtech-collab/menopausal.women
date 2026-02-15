import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split, KFold
from sklearn.metrics import r2_score
import warnings
import os

warnings.filterwarnings('ignore')

# 1. CẤU HÌNH
FILE_PATH = r"D:\PycharmProjects\PNMK\icatsd2026_menopause_qol\data\processed\clean_data_final.csv"


def reduce_overfitting_advanced():
    if not os.path.exists(FILE_PATH):
        print(f"LỖI: Không tìm thấy file tại {FILE_PATH}")
        return

    df = pd.read_csv(FILE_PATH)

    # Định nghĩa biến đầu vào cơ bản
    demographics = ['Age', 'BMI', 'Education_Code', 'Income_Code', 'Marital_Code',
                    'Job_Code', 'Info_Search_Code', 'Meno_Duration', 'Meno_Group']

    # Lấy thêm 4 câu hỏi vàng (Giả định từ bước trước)
    # Trong thực tế bạn có thể chạy lại bước tìm Feature Importance để lấy chính xác
    # Ở đây tôi lấy ví dụ 4 câu thường gặp nhất để demo
    gold_q = ['PSS_3', 'PSS_10', 'MEN_Sleep', 'MEN_Fatigue']

    # Đảm bảo các cột này tồn tại
    gold_q = [c for c in gold_q if c in df.columns]

    X = df[demographics + gold_q]

    print("=" * 60)
    print("CHIẾN LƯỢC GIẢM OVERFITTING TỰ ĐỘNG")
    print("=" * 60)

    # --- BƯỚC 1: LOẠI BỎ BIẾN TƯƠNG QUAN CAO (MULTICOLLINEARITY) ---
    # Dựa trên insight từ báo cáo HTML: Nếu 2 biến giống nhau > 90%, bỏ 1 biến
    print("\n[1] Kiểm tra Đa cộng tuyến...")
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]

    if to_drop:
        print(f"-> Phát hiện biến thừa (Cor > 0.9): {to_drop}")
        X = X.drop(columns=to_drop)
        print("-> Đã loại bỏ để giảm nhiễu.")
    else:
        print("-> Dữ liệu sạch, không có biến trùng lặp thông tin.")

    # --- BƯỚC 2: TỰ ĐỘNG TỐI ƯU THAM SỐ (HYPERPARAMETER TUNING) ---
    print("\n[2] Chạy tối ưu hóa tham số (Randomized Search)...")
    print("(Máy sẽ thử nghiệm hàng nghìn cấu hình để tìm ra bộ não tốt nhất)")

    def tune_model(target_col):
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Không gian tham số cần tìm kiếm
        # Đây là các "núm vặn" để chỉnh độ phức tạp của mô hình
        param_dist = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5, 6, 7, None],  # Độ sâu cây: Thấp -> Chống Overfit
            'min_samples_split': [2, 5, 10],  # Cần bao nhiêu mẫu để tách nhánh
            'min_samples_leaf': [2, 4, 6, 8],  # Cần bao nhiêu mẫu ở lá (Quan trọng nhất)
            'max_features': ['sqrt', 'log2', None]  # Số đặc trưng tối đa mỗi cây nhìn thấy
        }

        rf = ExtraTreesRegressor(random_state=42)

        # Chạy tìm kiếm ngẫu nhiên (Nhanh hơn GridSearch)
        random_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_dist,
            n_iter=50,  # Thử 50 tổ hợp ngẫu nhiên
            cv=5,  # Kiểm tra chéo 5 lần
            verbose=0,
            random_state=42,
            n_jobs=-1,  # Chạy song song tất cả CPU
            scoring='r2'
        )

        random_search.fit(X_train, y_train)

        best_model = random_search.best_estimator_

        # Đánh giá kết quả
        train_score = best_model.score(X_train, y_train)
        test_score = best_model.score(X_test, y_test)
        gap = train_score - test_score

        print(f"\n--- KẾT QUẢ TỐI ƯU CHO {target_col} ---")
        print(f"Tham số tốt nhất: {random_search.best_params_}")
        print(f"R2 Train: {train_score:.4f}")
        print(f"R2 Test : {test_score:.4f}")
        print(f"Độ lệch (Gap): {gap:.4f}")

        if gap < 0.15:
            print("=> ✅ MÔ HÌNH RẤT TỐT (Gap < 0.15)")
        elif gap < 0.25:
            print("=> ⚠️ CHẤP NHẬN ĐƯỢC (0.15 < Gap < 0.25)")
        else:
            print("=> ❌ VẪN CÒN OVERFITTING (Gap > 0.25). Cần giảm thêm số câu hỏi.")

    # Chạy tối ưu cho cả 2 mục tiêu
    tune_model('PSS_Score')
    tune_model('MENQOL_Score')


if __name__ == "__main__":
    reduce_overfitting_advanced()