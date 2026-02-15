import pandas as pd
import numpy as np
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

# --- CẤU HÌNH ĐƯỜNG DẪN ---
MODEL_PATH = r"/menopause_qol\src1\final_health_advisor.pkl"

# --- BỘ TỪ ĐIỂN GIẢI THÍCH (MAPPING DATA) ---
DEMO_MAP = {
    'Age': 'Tuổi (Ví dụ: 54)',
    'BMI': 'Chỉ số khối cơ thể (BMI) (Ví dụ: 22.5)',
    'Education_Code': 'Trình độ học vấn (0: Không học - 5: Sau đại học)',
    'Income_Code': 'Mức thu nhập (1: <5tr - 4: >20tr)',
    'Job_Code': 'Nghề nghiệp (1: Nội trợ - 6: Chuyên gia)',
    'Marital_Code': 'Tình trạng hôn nhân (0: Độc thân - 1: Có gia đình)',
    'Meno_Duration_New': 'Thời gian đã mãn kinh (năm)',
    'Meno_Status': 'Trạng thái mãn kinh (0: Chưa mãn kinh - 1: Đã mãn kinh)',
    'Chronic_Disease': 'Tiền sử bệnh mãn tính (0: Không - 1: Có)'
}

QUESTION_MAP = {
    'PSS_1': 'Cảm thấy khó khăn khi đối mặt với những vấn đề dồn dập',
    'PSS_2': 'Cảm thấy mất kiểm soát đối với những việc quan trọng trong cuộc sống',
    'PSS_3': 'Cảm thấy căng thẳng và lo lắng',
    'PSS_4': 'Cảm thấy tự tin vào khả năng xử lý các vấn đề cá nhân',
    'PSS_5': 'Cảm thấy mọi việc đang diễn ra theo ý muốn của mình',
    'PSS_6': 'Cảm thấy không thể làm hết tất cả các việc cần làm',
    'PSS_7': 'Cảm thấy có thể giữ được bình tĩnh trước những khó khăn',
    'PSS_8': 'Cảm thấy làm chủ được tình hình',
    'PSS_9': 'Cảm thấy tức giận vì những việc nằm ngoài tầm kiểm soát',
    'PSS_10': 'Cảm thấy các khó khăn tích tụ vượt mức có thể giải quyết',
    'MEN_HotFlash': 'Bốc hỏa, nóng bừng mặt',
    'MEN_Memory': 'Giảm trí nhớ, hay quên',
    'MEN_Sleep': 'Mất ngủ hoặc khó ngủ',
    'MEN_Headache': 'Nhức đầu hoặc đau nửa đầu',
    'MEN_MuscleAche': 'Đau cơ hoặc khớp',
    'MEN_Fatigue': 'Cảm thấy mệt mỏi, thiếu năng lượng',
    'MEN_Weight': 'Tăng cân nhanh chóng',
    'MEN_Skin': 'Thay đổi cấu trúc da (khô, nhăn)',
    'MEN_Depressed': 'Cảm thấy lo âu hoặc trầm cảm',
    'MEN_Impatient': 'Cảm thấy mất kiên nhẫn, dễ cáu gắt',
    'MEN_Urine': 'Vấn đề về tiểu tiện (tiểu nhiều, tiểu đêm)',
    'MEN_Libido': 'Thay đổi ham muốn tình dục',
    'Supp_Calcium': 'Sử dụng thực phẩm bổ sung Canxi (0: Không - 1: Có)',
    'Supp_Omega3': 'Sử dụng thực phẩm bổ sung Omega-3 (0: Không - 1: Có)'
}


def run_app():
    if not os.path.exists(MODEL_PATH):
        print("❌ Lỗi: Không tìm thấy file model tại đường dẫn đã chỉ định.")
        return

    # 1. Load "Bộ não" AI
    package = joblib.load(MODEL_PATH)
    scaler = package['scaler']
    kmeans = package['kmeans']
    experts = package['experts']
    strategy = package['strategy']
    demo_feats = package['demo_feats']

    print("=" * 60)
    print("🌟 HỆ THỐNG TƯ VẤN SỨC KHỎE THÍCH NGHI (ADAPTIVE AI) 🌟")
    print("=" * 60)

    # 2. Nhập dữ liệu nhân khẩu học
    user_data = {}
    print("\n[BƯỚC 1] VUI LÒNG NHẬP THÔNG TIN CƠ BẢN:")
    for feat in demo_feats:
        label = DEMO_MAP.get(feat, feat)
        val = float(input(f"   + {label}: "))
        user_data[feat] = val

    # 3. Phân loại nhóm người dùng (Clustering)
    user_demo_df = pd.DataFrame([user_data])[demo_feats]
    user_demo_scaled = scaler.transform(user_demo_df)
    c_id = kmeans.predict(user_demo_scaled)[0]

    print(f"\n✅ AI PHÂN TÍCH: Bạn thuộc Nhóm đối tượng {c_id}")
    print("   (Hệ thống đang điều chỉnh các câu hỏi chuyên sâu dành riêng cho bạn...)")

    # 4. Đặt câu hỏi vàng thích nghi (Adaptive Questions)
    # Lấy danh sách các câu hỏi cần thiết cho cả 2 mục tiêu (PSS và MEN)
    gold_qs = list(set(strategy[c_id]['PSS'] + strategy[c_id]['MEN']))

    print("\n[BƯỚC 2] VUI LÒNG ĐÁNH GIÁ CÁC TRIỆU CHỨNG SAU:")
    print("   (Thang điểm PSS: 0-4 | Thang điểm MENQOL: 1-6)")

    for q in gold_qs:
        label = QUESTION_MAP.get(q, q)
        val = float(input(f"   + {label}: "))
        user_data[q] = val

    # 5. Dự báo kết quả cuối cùng
    print("\n" + "=" * 60)
    print("📊 KẾT QUẢ DỰ BÁO SỨC KHỎE TỪ AI")
    print("=" * 60)

    for target_type in ['PSS', 'MEN']:
        target_name = "Mức độ Stress (PSS)" if target_type == 'PSS' else "Chất lượng sống (MENQOL)"
        required_feats = demo_feats + strategy[c_id][target_type]

        # Tạo input đúng định dạng DataFrame cho model
        final_input_df = pd.DataFrame([user_data])[required_feats]

        # Gọi chuyên gia tương ứng
        prediction = experts[f"expert_{c_id}_{target_type}"].predict(final_input_df)[0]

        print(f"   ▶ {target_name}: {prediction:.2f}")

    print("\nLưu ý: Kết quả mang tính tham khảo, vui lòng tham vấn ý kiến bác sĩ.")
    print("=" * 60)


if __name__ == "__main__":
    run_app()