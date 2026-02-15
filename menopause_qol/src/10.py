import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesRegressor
import warnings
import os

warnings.filterwarnings('ignore')

# --- 1. CẤU HÌNH VÀ KHỞI TẠO DỮ LIỆU (BẮT BUỘC) ---
FILE_PATH = r"/menopause_qol\data\processed\clean_data_final.csv"

# Kiểm tra file trước khi chạy
if not os.path.exists(FILE_PATH):
    print(f"❌ LỖI: Không tìm thấy file tại {FILE_PATH}. Hãy chạy lại bước tiền xử lý dữ liệu.")
    exit()

df = pd.read_csv(FILE_PATH)

# Định nghĩa các nhóm biến để hàm get_adaptive_prediction có thể sử dụng
demographics = ['Age', 'BMI', 'Education_Code', 'Income_Code', 'Marital_Code',
                'Job_Code', 'Info_Search_Code', 'Meno_Duration', 'Meno_Group']
pool_pss = [c for c in df.columns if c.startswith('PSS_') and c != 'PSS_Score']
pool_men = [c for c in df.columns if c.startswith('MEN_') and c != 'MENQOL_Score']

# Huấn luyện bộ não phân cụm (KMeans) dựa trên dữ liệu hiện có
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(df[demographics].fillna(0))

# --- 2. TỪ ĐIỂN NỘI DUNG CÂU HỎI ---
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
    'MEN_Libido': 'Thay đổi ham muốn tình dục'
}

# --- 3. ĐỊNH NGHĨA HÀM DỰ BÁO THÍCH NGHI ---
def get_adaptive_prediction():
    print("\n" + "=" * 70)
    print("HỆ THỐNG DỰ BÁO SỨC KHỎE THÍCH NGHI (ADAPTIVE AI SYSTEM)")
    print("Gợi ý: Vui lòng nhập đúng mã số để máy nhận diện chính xác.")
    print("=" * 70)

    menus = {
        'Education_Code': "0: Không học, 2: THCS, 3: THPT, 4: ĐH-CĐ, 5: Sau ĐH",
        'Income_Code': "1: <5tr, 2: 5-10tr, 3: 10-20tr, 4: >20tr",
        'Marital_Code': "0: Độc thân/Ly hôn, 1: Có gia đình (Sống cùng chồng/con)",
        'Job_Code': "1: Nội trợ, 2: Công nhân, 3: Văn phòng, 4: Kinh doanh, 5: Hưu trí, 6: Chuyên gia",
        'Info_Search_Code': "0: Không tìm hiểu, 1: Có tìm hiểu",
        'Meno_Group': "0: Chưa mãn kinh, 1: Đã mãn kinh"
    }

    user_input = []
    print("\n[BƯỚC 1]: THIẾT LẬP HỒ SƠ NHÂN KHẨU HỌC")

    for feat in demographics:
        if feat in menus:
            print(f"\n💡 Gợi ý cho {feat}: {menus[feat]}")
            val = float(input(f"==> Nhập mã số cho {feat}: "))
        else:
            example = "54" if "Age" in feat else "22.5"
            val = float(input(f"==> Nhập giá trị {feat} (Ví dụ: {example}): "))
        user_input.append(val)

    # Logic phân cụm người dùng mới
    user_df = pd.DataFrame([user_input], columns=demographics)
    cluster_id = kmeans.predict(user_df)[0]

    print(f"\n{'!' * 20}")
    print(f"KẾT QUẢ: Hệ thống xác định bạn thuộc Nhóm đối tượng (Cluster): {cluster_id}")
    print(f"{'!' * 20}")

    cluster_data = df[df['Cluster'] == cluster_id]

    def find_gold_questions(target, pool):
        selector = ExtraTreesRegressor(n_estimators=100, random_state=42)
        selector.fit(cluster_data[pool], cluster_data[target])
        importance = pd.Series(selector.feature_importances_, index=pool)
        return importance.nlargest(2).index.tolist()

    gold_q = find_gold_questions('PSS_Score', pool_pss) + find_gold_questions('MENQOL_Score', pool_men)

    print("\n[BƯỚC 2]: TRẢ LỜI CÁC CÂU HỎI CHỈ BÁO THÍCH NGHI")
    print(f"Dựa trên hồ sơ của bạn, máy gợi ý 4 triệu chứng then chốt sau:")

    additional_answers = {}
    for q in gold_q:
        question_text = QUESTION_MAP.get(q, "Triệu chứng chưa xác định")
        if "PSS" in q:
            guide = "(0: Không bao giờ | 1: Hiếm khi | 2: Thỉnh thoảng | 3: Thường xuyên | 4: Rất thường xuyên)"
        else:
            guide = "(1: Không có triệu chứng -> 6: Triệu chứng rất nghiêm trọng)"

        print(f"\n❓ CÂU HỎI: {question_text}")
        print(f"💡 Gợi ý thang điểm: {guide}")
        val = float(input(f"==> Trả lời của bạn: "))
        additional_answers[q] = val

    def get_final_score(target, gold_features):
        model = ExtraTreesRegressor(n_estimators=100, random_state=42)
        model.fit(cluster_data[demographics + gold_features], cluster_data[target])
        full_vec = user_input + [additional_answers[q] for q in gold_features]
        return model.predict([full_vec])[0]

    pss_final = get_final_score('PSS_Score', [q for q in gold_q if "PSS" in q])
    men_final = get_final_score('MENQOL_Score', [q for q in gold_q if "MEN" in q])

    print("\n" + "*" * 60)
    print("BÁO CÁO PHÂN TÍCH SỨC KHỎE TỔNG QUAN")
    print("*" * 60)
    print(f"- Dự báo mức độ Stress: {pss_final:.2f} / 40 (Càng thấp càng tốt)")
    print(f"- Dự báo Chất lượng sống: {men_final:.2f} / 6 (Càng thấp càng tốt)")
    print("*" * 60)

# --- 4. GỌI HÀM THỰC THI (QUAN TRỌNG NHẤT) ---
if __name__ == "__main__":
    get_adaptive_prediction()