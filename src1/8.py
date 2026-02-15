import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, KFold

# --- CẤU HÌNH ---
MODEL_PATH = r"D:\PycharmProjects\PNMK\icatsd2026_menopause_qol\src1\final_health_advisor.pkl"
TRAIN_BALANCED_FILE = "train_data_balanced.csv"  # File dùng để train (đã có SMOTE)
TEST_FILE = "test_data.csv"  # File dữ liệu thật hoàn toàn


def check_model_health():
    if not os.path.exists(MODEL_PATH):
        print("❌ Không tìm thấy file model!")
        return

    package = joblib.load(MODEL_PATH)
    df_train = pd.read_csv(TRAIN_BALANCED_FILE)
    df_test = pd.read_csv(TEST_FILE)

    experts = package['experts']
    strategy = package['strategy']
    demo_feats = package['demo_feats']

    print("--- HỆ THỐNG KIỂM TRA SỨC KHỎE MÔ HÌNH (MODEL HEALTH CHECK) ---")

    for target in ['PSS', 'MEN']:
        print(f"\n🎯 Đánh giá mục tiêu: {target}")

        for c_id in range(3):
            # Lấy model expert cụ thể
            model = experts[f"expert_{c_id}_{target}"]
            feats = demo_feats + strategy[c_id][target]

            # 1. Kiểm tra trên tập Train (Dữ liệu đã học)
            X_train = df_train[df_train['Cluster'] == c_id][feats]
            y_train = df_train[df_train['Cluster'] == c_id]['PSS_Score' if target == 'PSS' else 'MENQOL_Score']
            train_preds = model.predict(X_train)
            train_r2 = r2_score(y_train, train_preds)

            # 2. Kiểm tra trên tập Test (Dữ liệu mới hoàn toàn)
            X_test = df_test[df_test['Cluster'] == c_id][feats]
            y_test = df_test[df_test['Cluster'] == c_id]['PSS_Score' if target == 'PSS' else 'MENQOL_Score']

            if len(y_test) > 0:
                test_preds = model.predict(X_test)
                # Tính R2 tập test (Chỉ tính nếu > 1 mẫu)
                test_r2 = r2_score(y_test, test_preds) if len(y_test) > 1 else np.nan
                test_mae = mean_absolute_error(y_test, test_preds)

                # KHOẢNG CÁCH GIỮA TRAIN VÀ TEST (OVERFITTING GAP)
                gap = train_r2 - test_r2 if not np.isnan(test_r2) else 0

                print(f"\n📍 Cluster {c_id}:")
                print(f"   - R2 Train: {train_r2:.4f}")
                print(f"   - R2 Test : {test_r2:.4f}" if not np.isnan(test_r2) else f"   - R2 Test : N/A (Mẫu quá ít)")
                print(f"   - MAE Test: {test_mae:.4f}")

                # ĐƯA RA CẢNH BÁO
                if gap > 0.2:
                    print("   ⚠️ CẢNH BÁO: Có dấu hiệu Overfitting (Khoảng cách Train-Test quá lớn).")
                elif train_r2 > 0.98 and test_r2 < 0.5:
                    print("   🚨 NGUY HIỂM: Mô hình đang học thuộc lòng (Data Leakage?).")
                else:
                    print("   ✅ Mô hình ổn định (Khả năng tổng quát hóa tốt).")
            else:
                print(f"\n📍 Cluster {c_id}: Không có dữ liệu test để đánh giá.")


if __name__ == "__main__":
    check_model_health()