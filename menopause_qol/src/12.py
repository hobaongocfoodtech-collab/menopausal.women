import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesRegressor
import warnings
import os

warnings.filterwarnings('ignore')

# --- 1. KHỞI TẠO DỮ LIỆU & MÔ HÌNH NỀN ---
FILE_PATH = r"/menopause_qol\data\processed\clean_data_final.csv"

# Kiểm tra file tồn tại
if not os.path.exists(FILE_PATH):
    print(f"❌ LỖI: Không tìm thấy file dữ liệu tại {FILE_PATH}")
    exit()

df = pd.read_csv(FILE_PATH)

# Định nghĩa nhóm biến
demographics = ['Age', 'BMI', 'Education_Code', 'Income_Code', 'Marital_Code',
                'Job_Code', 'Info_Search_Code', 'Meno_Duration', 'Meno_Group']
pool_pss = [c for c in df.columns if c.startswith('PSS_') and c != 'PSS_Score']
pool_men = [c for c in df.columns if c.startswith('MEN_') and c != 'MENQOL_Score']

# Huấn luyện K-Means để phân cụm
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(df[demographics].fillna(0))

# Từ điển hiển thị nội dung câu hỏi
QUESTION_MAP = {
    'PSS_1': 'Khó khăn khi đối mặt với những vấn đề dồn dập',
    'PSS_2': 'Mất kiểm soát đối với những việc quan trọng',
    'PSS_3': 'Cảm thấy căng thẳng và lo lắng',
    'PSS_4': 'Tự tin vào khả năng xử lý vấn đề cá nhân',
    'PSS_5': 'Mọi việc đang diễn ra theo ý muốn',
    'PSS_6': 'Không thể làm hết tất cả các việc cần làm',
    'PSS_7': 'Có thể giữ bình tĩnh trước khó khăn',
    'PSS_8': 'Làm chủ được tình hình',
    'PSS_9': 'Tức giận vì việc ngoài tầm kiểm soát',
    'PSS_10': 'Khó khăn tích tụ vượt mức giải quyết',
    'MEN_HotFlash': 'Bốc hỏa, nóng bừng mặt',
    'MEN_Memory': 'Giảm trí nhớ, hay quên',
    'MEN_Sleep': 'Mất ngủ hoặc khó ngủ',
    'MEN_Headache': 'Nhức đầu hoặc đau nửa đầu',
    'MEN_MuscleAche': 'Đau cơ hoặc khớp',
    'MEN_Fatigue': 'Mệt mỏi, thiếu năng lượng',
    'MEN_Weight': 'Tăng cân nhanh chóng',
    'MEN_Skin': 'Da khô, nhăn',
    'MEN_Depressed': 'Lo âu hoặc trầm cảm',
    'MEN_Impatient': 'Mất kiên nhẫn, dễ cáu',
    'MEN_Urine': 'Tiểu nhiều, tiểu đêm',
    'MEN_Libido': 'Giảm ham muốn tình dục'
}

# --- CẤU HÌNH THAM SỐ TỐI ƯU (TỪ BƯỚC TUNING) ---
# Đây là "trái tim" của sự thay đổi
PARAMS_PSS = {
    'n_estimators': 300,
    'min_samples_split': 2,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'max_depth': 4,  # Cây nông để tránh Overfitting cho Stress
    'random_state': 42
}

PARAMS_MEN = {
    'n_estimators': 300,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'max_depth': 7,  # Cây sâu để đạt độ chính xác cao cho CLCS
    'random_state': 42
}


# --- 2. HÀM DỰ BÁO THÍCH NGHI (FINAL VERSION) ---
def get_adaptive_prediction():
    print("\n" + "=" * 70)
    print("HỆ THỐNG DỰ BÁO SỨC KHỎE THÍCH NGHI (OPTIMIZED AI)")
    print("=" * 70)

    # Menu hướng dẫn nhập liệu
    menus = {
        'Education_Code': "0: Không học, 2: THCS, 3: THPT, 4: ĐH-CĐ, 5: Sau ĐH",
        'Income_Code': "1: <5tr, 2: 5-10tr, 3: 10-20tr, 4: >20tr",
        'Marital_Code': "0: Độc thân, 1: Có gia đình",
        'Job_Code': "1: Nội trợ, 2: Công nhân, 3: Văn phòng, 4: Kinh doanh, 5: Hưu trí, 6: Chuyên gia",
        'Info_Search_Code': "0: Không, 1: Có",
        'Meno_Group': "0: Chưa mãn kinh, 1: Đã mãn kinh"
    }

    user_input = []
    print("\n[BƯỚC 1]: NHẬP HỒ SƠ NHÂN KHẨU HỌC")

    for feat in demographics:
        if feat in menus:
            print(f"💡 Gợi ý: {menus[feat]}")
            val = float(input(f"==> Nhập mã số {feat}: "))
        else:
            val = float(input(f"==> Nhập giá trị {feat}: "))
        user_input.append(val)

    # Phân cụm người dùng
    user_df = pd.DataFrame([user_input], columns=demographics)
    cluster_id = kmeans.predict(user_df)[0]
    print(f"\n✅ Hệ thống xác định bạn thuộc Nhóm: {cluster_id}")

    # Lấy dữ liệu của nhóm tương ứng
    cluster_data = df[df['Cluster'] == cluster_id]

    # Hàm tìm câu hỏi vàng (Sử dụng tham số chuẩn)
    def find_gold_questions(target, pool):
        # Chọn bộ tham số dựa trên mục tiêu
        params = PARAMS_PSS if target == 'PSS_Score' else PARAMS_MEN

        selector = ExtraTreesRegressor(**params)
        selector.fit(cluster_data[pool], cluster_data[target])
        importance = pd.Series(selector.feature_importances_, index=pool)
        return importance.nlargest(2).index.tolist()

    gold_q = find_gold_questions('PSS_Score', pool_pss) + find_gold_questions('MENQOL_Score', pool_men)

    print("\n[BƯỚC 2]: TRẢ LỜI CÂU HỎI THÍCH NGHI")
    additional_answers = {}

    for q in gold_q:
        q_text = QUESTION_MAP.get(q, q)
        if "PSS" in q:
            guide = "(0: Không bao giờ -> 4: Rất thường xuyên)"
        else:
            guide = "(1: Không có -> 6: Rất nghiêm trọng)"

        print(f"\n❓ {q_text}")
        val = float(input(f"   {guide} Nhập điểm: "))
        additional_answers[q] = val

    # Hàm dự báo cuối cùng
    def final_predict(target, features):
        # Chọn bộ tham số chuẩn xác nhất cho từng loại dự báo
        params = PARAMS_PSS if target == 'PSS_Score' else PARAMS_MEN

        model = ExtraTreesRegressor(**params)
        model.fit(cluster_data[demographics + features], cluster_data[target])

        full_vec = user_input + [additional_answers[q] for q in features]
        return model.predict([full_vec])[0]

    pss = final_predict('PSS_Score', [q for q in gold_q if "PSS" in q])
    men = final_predict('MENQOL_Score', [q for q in gold_q if "MEN" in q])

    print("\n" + "*" * 60)
    print("BÁO CÁO KẾT QUẢ (Đã tối ưu hóa)")
    print("*" * 60)
    print(f"1. Mức độ Stress (PSS): {pss:.2f} / 40")
    print(f"2. Chất lượng sống (MENQOL): {men:.2f} / 6")
    print("*" * 60)


# --- 3. DÒNG LỆNH QUAN TRỌNG ĐỂ CHẠY CODE ---
if __name__ == "__main__":
    try:
        get_adaptive_prediction()
    except Exception as e:
        print(f"❌ LỖI HỆ THỐNG: {e}")