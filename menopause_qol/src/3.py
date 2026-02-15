import pandas as pd
import numpy as np
import os

# --- 1. CẤU HÌNH ĐƯỜNG DẪN ---
INPUT_PATH = r"C:\Users\Admin\PycharmProjects\PNMK\icatsd2026_menopause_qol\data\raw\XLSL.PNMK01.xlsx"
OUTPUT_PATH = r"C:\Users\Admin\PycharmProjects\PNMK\icatsd2026_menopause_qol\data\processed\clean_data_final.csv"


def preprocess_data():
    print("--- BẮT ĐẦU QUY TRÌNH TIỀN XỬ LÝ DỮ LIỆU ---")

    # 1. ĐỌC DỮ LIỆU
    try:
        df = pd.read_excel(INPUT_PATH, sheet_name=1, engine='openpyxl')
    except Exception:
        df = pd.read_excel(INPUT_PATH, sheet_name=0, engine='openpyxl')

    df.columns = df.columns.str.strip()
    print(f"-> Đã tải dữ liệu thành công: {len(df)} dòng.")

    # 2. CHUẨN HÓA CỘT TUỔI MÃN KINH
    if 'Meno_Age' in df.columns:
        df.rename(columns={'Meno_Age': 'Meno_Age_Raw'}, inplace=True)

    if 'Meno_Age_Raw' not in df.columns:
        print("Cảnh báo: Không tìm thấy cột Meno_Age_Raw. Kiểm tra lại header.")
        return

    # 3. PHÂN NHÓM MÃN KINH VÀ TÍNH DURATION
    def classify_menopause(val):
        v = str(val).lower().strip()
        # Nếu chứa chữ "chưa", "không", "tiền" hoặc để trống -> Nhóm 0 (Chưa mãn kinh)
        if any(kw in v for kw in ['chưa', 'tiền', 'không', 'vẫn', 'nan']):
            return 0
        try:
            float(val)
            return 1  # Nhóm 1 (Đã mãn kinh)
        except:
            return 0

    df['Meno_Group'] = df['Meno_Age_Raw'].apply(classify_menopause)
    df['Meno_Age_Numeric'] = pd.to_numeric(df['Meno_Age_Raw'], errors='coerce').fillna(0)
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce').fillna(df['Age'].mean())

    # Tính thời gian mãn kinh (Duration) = Tuổi hiện tại - Tuổi lúc mãn kinh
    df['Meno_Duration'] = (df['Age'] - df['Meno_Age_Numeric']).clip(lower=0)
    # Nếu chưa mãn kinh thì Duration mặc định bằng 0
    df.loc[df['Meno_Group'] == 0, 'Meno_Duration'] = 0

    # 4. TÍNH ĐIỂM STRESS (PSS-10)
    # Các câu hỏi đảo ngược điểm (Reverse scoring): 4, 5, 7, 8
    pss_rev = ['PSS_4', 'PSS_5', 'PSS_7', 'PSS_8']
    pss_norm = ['PSS_1', 'PSS_2', 'PSS_3', 'PSS_6', 'PSS_9', 'PSS_10']

    for c in pss_rev + pss_norm:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    # Công thức: Tổng (câu thuận) + Tổng (4 - câu đảo)
    df['PSS_Score'] = df[pss_norm].sum(axis=1) + (4 * len(pss_rev) - df[pss_rev].sum(axis=1))

    # 5. TÍNH ĐIỂM CHẤT LƯỢNG CUỘC SỐNG (MENQOL)
    # MENQOL Score thường là trung bình cộng của tất cả các triệu chứng lẻ
    men_items = [c for c in df.columns if c.startswith('MEN_')]
    if men_items:
        for c in men_items:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        df['MENQOL_Score'] = df[men_items].mean(axis=1)

    # 6. MAPPING NHÂN KHẨU HỌC SANG MÃ SỐ (CODING)
    # Mapping cho Học vấn
    map_edu = {
        'Không đi học': 0, 'THCS': 2, 'THPT – trung cấp': 3,
        'THPT - trung cấp': 3, 'Đại học – cao đẳng': 4,
        'Đại học - cao đẳng': 4, 'Trên đại học': 5
    }
    # Mapping cho Thu nhập
    map_inc = {
        'Dưới 5 triệu': 1, 'Từ 5 đến 10 triệu': 2,
        'Trên 10 triệu': 3, 'Từ 11 đến 20 triệu': 3, 'Trên 20 triệu': 4
    }
    # Mapping cho Nghề nghiệp
    map_job = {
        'Nội trợ': 1, 'Công nhân': 2, 'Nhân viên văn phòng': 3,
        'Kinh doanh': 4, 'Hưu trí': 5, 'Về hưu': 5
    }

    df['Education_Code'] = df['Education'].map(map_edu).fillna(3)
    df['Income_Code'] = df['Income'].map(map_inc).fillna(2)
    df['Job_Code'] = df['Job'].map(map_job).fillna(0)

    # Mapping Tình trạng hôn nhân (1: Có đôi, 0: Độc lập)
    df['Marital_Code'] = df['Marital_Status'].apply(lambda x: 1 if 'sống cùng' in str(x).lower() else 0)

    # Mapping Tìm kiếm thông tin (1: Có, 0: Không)
    df['Info_Search_Code'] = df['Info_Search'].apply(lambda x: 1 if str(x).strip() == 'Có' else 0)

    # 7. TÍNH CHỈ SỐ BMI
    df['BMI'] = pd.to_numeric(df['Weight_kg'], errors='coerce') / (
                (pd.to_numeric(df['Height_cm'], errors='coerce') / 100) ** 2)
    # Xử lý các giá trị vô cực hoặc lỗi chia cho 0
    df['BMI'] = df['BMI'].replace([np.inf, -np.inf], 0).fillna(df['BMI'].mean())

    # 8. XUẤT FILE SẠCH
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')

    print("-" * 40)
    print(f"HOÀN TẤT! Dữ liệu đã được làm sạch và lưu tại:")
    print(f"-> {OUTPUT_PATH}")
    print(f"Các cột mới đã tạo: PSS_Score, MENQOL_Score, BMI, Meno_Duration, và các cột _Code.")


if __name__ == "__main__":
    try:
        preprocess_data()
    except Exception as e:
        print(f"LỖI THỰC THI: {e}")