import pandas as pd
import numpy as np
import re
import os

# --- CẤU HÌNH ĐƯỜNG DẪN ---
INPUT_PATH = r"C:\Users\Admin\PycharmProjects\PNMK\icatsd2026_menopause_qol\data\raw\XLSL.PNMK01.xlsx"
OUTPUT_PATH = r"C:\Users\Admin\PycharmProjects\PNMK\icatsd2026_menopause_qol\data\processed\clean_data_refined.csv"


def refine_dataset():
    print("--- BẮT ĐẦU HIỆU CHỈNH DỮ LIỆU ---")

    # 1. ĐỌC DỮ LIỆU
    df = pd.read_excel(INPUT_PATH, sheet_name=1, engine='openpyxl')
    df.columns = df.columns.str.strip()

    # Định nghĩa danh sách chuyên gia ở đây để dùng chung
    expert_list = ['bác sĩ', 'luật sư', 'giáo viên', 'giảng viên', 'nha sĩ', 'công an',
                   'nhà báo', 'hộ lí', 'diễn viên', 'huấn luyện', 'yoga', 'aerobic', 'csskcđ']

    # 2. XỬ LÝ NGHỀ NGHIỆP
    def smart_group_job(job_text):
        j = str(job_text).lower().strip()
        if any(kw in j for kw in ['nội trợ']): return 1
        if any(kw in j for kw in ['công nhân', 'thợ', 'lao động']): return 2
        if any(kw in j for kw in ['văn phòng', 'kế toán', 'nhân viên']): return 3
        if any(kw in j for kw in ['kinh doanh', 'buôn bán', 'giám đốc', 'quản lý', 'tiếp viên']): return 4
        if any(kw in j for kw in ['về hưu', 'hưu trí']): return 5
        if any(kw in j for kw in expert_list): return 6
        return 0  # Khác

    df['Job_Code'] = df['Job'].apply(smart_group_job)

    # 3. XỬ LÝ TUỔI MÃN KINH
    def clean_meno_age(val):
        v = str(val).lower().strip()
        if any(kw in v for kw in ['chưa', 'vẫn', 'đang', 'sắp', 'nan']): return 0
        nums = re.findall(r'\d+', v)
        return int(nums[0]) if nums else 0

    df['Meno_Age_Numeric'] = df['Meno_Age'].apply(clean_meno_age)
    df['Meno_Group'] = df['Meno_Age_Numeric'].apply(lambda x: 1 if x > 0 else 0)

    # 4. TÍNH DURATION & BMI
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce').fillna(df['Age'].mean())
    df['Meno_Duration'] = (df['Age'] - df['Meno_Age_Numeric']).clip(lower=0)
    df.loc[df['Meno_Group'] == 0, 'Meno_Duration'] = 0

    df['BMI'] = pd.to_numeric(df['Weight_kg'], errors='coerce') / (
                (pd.to_numeric(df['Height_cm'], errors='coerce') / 100) ** 2)
    df['BMI'] = df['BMI'].replace([np.inf, -np.inf], 0).fillna(df['BMI'].mean())

    # 5. TÍNH ĐIỂM PSS-10
    pss_rev = ['PSS_4', 'PSS_5', 'PSS_7', 'PSS_8']
    pss_norm = ['PSS_1', 'PSS_2', 'PSS_3', 'PSS_6', 'PSS_9', 'PSS_10']
    for c in pss_rev + pss_norm: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    df['PSS_Score'] = df[pss_norm].sum(axis=1) + (4 * len(pss_rev) - df[pss_rev].sum(axis=1))

    # 6. TÍNH ĐIỂM MENQOL
    men_items = [c for c in df.columns if c.startswith('MEN_')]
    for c in men_items: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    df['MENQOL_Score'] = df[men_items].mean(axis=1)

    # 7. MAPPING CÁC BIẾN CÒN LẠI
    map_edu = {'Không đi học': 0, 'THCS': 2, 'THPT – trung cấp': 3, 'THPT - trung cấp': 3, 'Đại học – cao đẳng': 4,
               'Đại học - cao đẳng': 4, 'Trên đại học': 5}
    map_inc = {'Dưới 5 triệu': 1, 'Từ 5 đến 10 triệu': 2, 'Trên 10 triệu': 3, 'Từ 11 đến 20 triệu': 3,
               'Trên 20 triệu': 4}

    df['Education_Code'] = df['Education'].map(map_edu).fillna(3)
    df['Income_Code'] = df['Income'].map(map_inc).fillna(2)
    df['Marital_Code'] = df['Marital_Status'].apply(lambda x: 0 if 'độc thân' in str(x).lower() else 1)
    df['Info_Search_Code'] = df['Info_Search'].apply(lambda x: 1 if str(x).strip() == 'Có' else 0)

    # 8. XUẤT FILE
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')

    print("-" * 30)
    print(f"✅ HIỆU CHỈNH THÀNH CÔNG!")
    print(f"-> File lưu tại: {OUTPUT_PATH}")
    print(f"-> Nhóm nghề nghiệp 'Chuyên gia' mới đã bao gồm {len(expert_list)} từ khóa định danh.")


if __name__ == "__main__":
    refine_dataset()