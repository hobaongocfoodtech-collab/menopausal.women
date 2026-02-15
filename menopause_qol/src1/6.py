import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --- CẤU HÌNH ---
DATA_FILE = "clean_data_final.csv"  # Dùng file gốc sạch nhất để tái lập quy trình
MODEL_PATH = "final_health_advisor.pkl"

# Chiến lược đã tối ưu từ bước 5
STRATEGY = {
    0: {'PSS': ['MEN_Memory', 'MEN_Libido', 'PSS_5', 'PSS_4', 'PSS_9'],
        'MEN': ['PSS_2', 'MEN_Sleep', 'MEN_HotFlash', 'PSS_1', 'PSS_8']},
    1: {'PSS': ['PSS_6', 'PSS_1', 'PSS_3', 'PSS_2', 'PSS_5'],
        'MEN': ['MEN_Libido', 'MEN_Depressed', 'MEN_Fatigue', 'PSS_3', 'PSS_1']},
    2: {'PSS': ['PSS_3', 'PSS_10', 'PSS_2', 'PSS_8', 'PSS_9'],
        'MEN': ['MEN_Sleep', 'MEN_Depressed', 'PSS_3', 'MEN_Impatient', 'MEN_Weight']}
}

DEMO_FEATS = ['Age', 'BMI', 'Education_Code', 'Income_Code', 'Job_Code', 'Marital_Code',
              'Meno_Duration_New', 'Meno_Status', 'Chronic_Disease']


def train_deployment_system():
    print("--- GIAI ĐOẠN 6: ĐÓNG GÓI MÔ HÌNH ---")
    df = pd.read_csv(DATA_FILE)

    # 1. Tái lập K-Means & Scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[DEMO_FEATS])
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # 2. Huấn luyện 6 mô hình chuyên gia (3 cụm x 2 mục tiêu)
    experts = {}
    for c_id in range(3):
        c_data = df[df['Cluster'] == c_id]
        for target_type in ['PSS', 'MEN']:
            target_col = 'PSS_Score' if target_type == 'PSS' else 'MENQOL_Score'
            features = DEMO_FEATS + STRATEGY[c_id][target_type]

            # Sử dụng tham số tốt nhất từ GridSearch bước 5
            model = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, random_state=42)
            model.fit(c_data[features], c_data[target_col])
            experts[f"expert_{c_id}_{target_type}"] = model

    # 3. Đóng gói tất cả vào một file duy nhất
    package = {
        'scaler': scaler,
        'kmeans': kmeans,
        'experts': experts,
        'strategy': STRATEGY,
        'demo_feats': DEMO_FEATS
    }
    joblib.dump(package, MODEL_PATH)
    print(f"✅ Hệ thống đã được đóng gói tại: {MODEL_PATH}")


def predict_health(user_demo_list):
    """Hàm dùng để dự báo cho người dùng mới"""
    package = joblib.load(MODEL_PATH)

    # B1: Xác định cụm
    user_demo_scaled = package['scaler'].transform([user_demo_list])
    c_id = package['kmeans'].predict(user_demo_scaled)[0]

    print(f"\nAI: Bạn thuộc nhóm đối tượng số {c_id}")
    print(f"AI: Để chính xác hơn, vui lòng trả lời thêm các câu hỏi sau: {package['strategy'][c_id]['PSS']}")

    # (Trong thực tế sẽ nhận input tiếp theo ở đây, ở đây ta giả lập)
    return c_id


if __name__ == "__main__":
    train_deployment_system()
    # Thử nghiệm dự báo cho 1 người: 52 tuổi, BMI 24, ĐH, Thu nhập 3, VP, Có gia đình, Mãn kinh 4 năm, Có bệnh nền
    # predict_health([52, 24, 4, 3, 3, 1, 4, 1, 1])