import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --- CẤU HÌNH ---
REAL_DATA = "clean_data_final.csv"
TRAIN_BALANCED = "train_data_balanced.csv"
MODEL_PATH = "final_health_advisor.pkl"

# 1. CẬP NHẬT CHIẾN LƯỢC TỪ KẾT QUẢ ENSEMBLE 5.8 (GOLD)
STRATEGY = {
    0: {
        'PSS': ['MEN_Memory', 'MEN_Libido', 'PSS_4', 'PSS_5', 'PSS_9'],
        'MEN': ['MEN_Sleep', 'PSS_8', 'PSS_2', 'MEN_HotFlash', 'PSS_1']
    },
    1: {
        'PSS': ['PSS_6', 'PSS_1', 'PSS_3', 'PSS_5', 'PSS_10'],
        'MEN': ['MEN_Depressed', 'MEN_Fatigue', 'MEN_Libido', 'PSS_3', 'PSS_1']
    },
    2: {
        'PSS': ['PSS_3', 'PSS_10', 'PSS_2', 'PSS_8', 'PSS_9'],
        'MEN': ['MEN_Sleep', 'MEN_Depressed', 'PSS_3', 'MEN_Impatient', 'MEN_Weight']
    }
}

# 2. CẬP NHẬT THAM SỐ TỐI ƯU TỪ GRIDSEARCH (FINE-TUNING)
BEST_PARAMS = {
    0: {
        'PSS': {'n_estimators': 300, 'min_samples_leaf': 2, 'max_depth': None, 'max_features': 'sqrt'},
        'MEN': {'n_estimators': 300, 'min_samples_leaf': 2, 'max_depth': 8, 'max_features': None}
    },
    1: {
        'PSS': {'n_estimators': 100, 'min_samples_leaf': 2, 'max_depth': 6, 'max_features': None},
        'MEN': {'n_estimators': 100, 'min_samples_leaf': 2, 'max_depth': 6, 'max_features': None}
    },
    2: {
        'PSS': {'n_estimators': 300, 'min_samples_leaf': 2, 'max_depth': 8, 'max_features': None},
        'MEN': {'n_estimators': 100, 'min_samples_leaf': 2, 'max_depth': 8, 'max_features': None}
    }
}

DEMO_FEATS = ['Age', 'BMI', 'Education_Code', 'Income_Code', 'Job_Code', 'Marital_Code',
              'Meno_Duration_New', 'Meno_Status', 'Chronic_Disease']


def train_deployment_system():
    print("--- GIAI ĐOẠN 6: ĐÓNG GÓI HỆ THỐNG CHUYÊN GIA TỐI ƯU ---")

    if not os.path.exists(REAL_DATA) or not os.path.exists(TRAIN_BALANCED):
        print("❌ Lỗi: Thiếu file dữ liệu cần thiết.")
        return

    df_real = pd.read_csv(REAL_DATA)
    df_balanced = pd.read_csv(TRAIN_BALANCED)

    # 1. Định hình không gian cụm (Dữ liệu THẬT)
    scaler = StandardScaler()
    scaler.fit(df_real[DEMO_FEATS])
    X_real_scaled = scaler.transform(df_real[DEMO_FEATS])

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_real_scaled)

    # 2. Gán nhãn cụm cho dữ liệu huấn luyện
    X_train_scaled = scaler.transform(df_balanced[DEMO_FEATS])
    df_balanced['Cluster'] = kmeans.predict(X_train_scaled)

    # 3. Huấn luyện các Expert Model với tham số tối ưu
    experts = {}
    for c_id in range(3):
        c_data = df_balanced[df_balanced['Cluster'] == c_id]
        if len(c_data) == 0: continue

        for t_type in ['PSS', 'MEN']:
            target_col = 'PSS_Score' if t_type == 'PSS' else 'MENQOL_Score'
            features = DEMO_FEATS + STRATEGY[c_id][t_type]
            params = BEST_PARAMS[c_id][t_type]

            # Khởi tạo model với tham số riêng biệt cho từng Expert
            model = ExtraTreesRegressor(
                n_estimators=params['n_estimators'],
                min_samples_leaf=params['min_samples_leaf'],
                max_depth=params['max_depth'],
                max_features=params['max_features'],
                random_state=42
            )

            model.fit(c_data[features], c_data[target_col])
            experts[f"expert_{c_id}_{t_type}"] = model
            print(f"   ✅ Đã huấn luyện expert_{c_id}_{t_type} thành công.")

    # 4. Đóng gói tài sản
    package = {
        'scaler': scaler,
        'kmeans': kmeans,
        'experts': experts,
        'strategy': STRATEGY,
        'demo_feats': DEMO_FEATS
    }

    joblib.dump(package, MODEL_PATH)
    print(f"\n🚀 HỆ THỐNG ĐÃ SẴN SÀNG: {MODEL_PATH}")


if __name__ == "__main__":
    train_deployment_system()