import pandas as pd
import numpy as np
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

# CẤU HÌNH
MODEL_DIR = r"D:\PycharmProjects\PNMK\icatsd2026_menopause_qol\models"

# CHIẾN LƯỢC CÂU HỎI (Phải khớp với lúc train)
ADAPTIVE_STRATEGY = {
    0: {
        'PSS_Score': ['MEN_Urine', 'PSS_10', 'PSS_1', 'PSS_2', 'PSS_8'],
        'MENQOL_Score': ['MEN_Impatient', 'MEN_Sleep', 'MEN_Fatigue', 'MEN_Depressed', 'PSS_6']
    },
    1: {
        'PSS_Score': ['PSS_1', 'PSS_3', 'PSS_6', 'MEN_Impatient', 'PSS_2'],
        'MENQOL_Score': ['MEN_Depressed', 'MEN_Libido', 'MEN_Skin', 'MEN_HotFlash', 'MEN_Fatigue']
    },
    2: {
        'PSS_Score': ['PSS_3', 'PSS_6', 'PSS_10', 'PSS_9', 'PSS_5'],
        'MENQOL_Score': ['MEN_Libido', 'PSS_3', 'MEN_Depressed', 'MEN_Impatient', 'MEN_HotFlash']
    }
}

TEXT_MAP = {
    'MEN_Urine': 'Tiểu đêm/Tiểu nhiều', 'PSS_10': 'Khó khăn chồng chất', 'PSS_1': 'Buồn bực bất ngờ',
    'PSS_2': 'Mất kiểm soát', 'PSS_8': 'Hài lòng (ngược)', 'MEN_Impatient': 'Mất kiên nhẫn',
    'MEN_Sleep': 'Mất ngủ', 'MEN_Fatigue': 'Kiệt sức', 'MEN_Depressed': 'Trầm cảm',
    'PSS_6': 'Quá tải', 'PSS_3': 'Lo âu/Stress', 'MEN_Libido': 'Giảm ham muốn',
    'MEN_Skin': 'Khô da', 'MEN_HotFlash': 'Bốc hỏa', 'PSS_9': 'Cáu gắt', 'PSS_5': 'Mọi việc ổn (ngược)'
}


def run_prediction_app():
    print("⏳ Đang tải 'bộ não' AI từ ổ cứng...")

    try:
        scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
        kmeans = joblib.load(os.path.join(MODEL_DIR, 'kmeans.pkl'))
    except FileNotFoundError:
        print("❌ Lỗi: Chưa tìm thấy file model. Hãy chạy src/21.py trước!")
        return

    print("\n" + "=" * 60)
    print("DEMO HỆ THỐNG TƯ VẤN SỨC KHỎE (CHẠY THỰC TẾ)")
    print("=" * 60)

    # 1. Nhập Nhân khẩu học (Giả lập nhập nhanh)
    print("Nhập thông tin bệnh nhân:")
    # Thứ tự: Age, BMI, Edu, Income, Marital, Job, Info, MenoDur, MenoGroup
    # Ví dụ: Một người 55 tuổi, BMI 23, Thu nhập thấp
    input_demo = []
    input_demo.append(float(input("1. Tuổi: ")))
    input_demo.append(float(input("2. BMI: ")))
    input_demo.append(float(input("3. Học vấn (0-5): ")))
    input_demo.append(float(input("4. Thu nhập (1-4): ")))
    input_demo.append(float(input("5. Hôn nhân (0-1): ")))
    input_demo.append(float(input("6. Nghề nghiệp (1-6): ")))
    input_demo.append(float(input("7. Tìm kiếm TT (0-1): ")))
    input_demo.append(float(input("8. Thời gian mãn kinh (năm): ")))
    input_demo.append(float(input("9. Trạng thái mãn kinh (0-1): ")))

    # 2. Phân loại cụm
    vec = np.array(input_demo).reshape(1, -1)
    vec_scaled = scaler.transform(vec)
    cluster_id = kmeans.predict(vec_scaled)[0]

    print(f"\n✅ AI Phân tích: Bệnh nhân thuộc Nhóm {cluster_id}")

    # 3. Hỏi câu hỏi thích nghi
    answers = {}
    print("\n--- CẦN KHAI THÁC THÊM CÁC TRIỆU CHỨNG SAU ---")

    needed_qs = list(set(ADAPTIVE_STRATEGY[cluster_id]['PSS_Score'] +
                         ADAPTIVE_STRATEGY[cluster_id]['MENQOL_Score']))

    for q in needed_qs:
        name = TEXT_MAP.get(q, q)
        if "PSS" in q:
            range_txt = "(0-4)"
        else:
            range_txt = "(1-6)"
        val = float(input(f"   + {name} {range_txt}: "))
        answers[q] = val

    # 4. Dự báo
    print("\n" + "-" * 50)
    print("KẾT QUẢ DỰ BÁO TỪ AI:")

    for target in ['PSS_Score', 'MENQOL_Score']:
        # Load model chuyên biệt cho cụm này
        model_path = os.path.join(MODEL_DIR, f'extratrees_c{cluster_id}_{target}.pkl')
        model = joblib.load(model_path)

        # Tạo vector input đúng chuẩn
        feats = ADAPTIVE_STRATEGY[cluster_id][target]
        final_input = input_demo.copy()
        for f in feats:
            final_input.append(answers[f])

        pred = model.predict([final_input])[0]

        print(f"   -> {target}: {pred:.2f}")

    print("-" * 50)


if __name__ == "__main__":
    run_prediction_app()