import pandas as pd
import numpy as np
import os
# Load dữ liệu
# --- CẤU HÌNH ĐƯỜNG DẪN ---
# Sử dụng đường dẫn tuyệt đối để tránh lỗi không tìm thấy file
FILE_PATH = r"/menopause_qol\data\processed\clean_data_refined.csv"

# Kiểm tra xem file có tồn tại không trước khi đọc
if not os.path.exists(FILE_PATH):
    print(f"❌ LỖI: Không tìm thấy file tại {FILE_PATH}")
    print("👉 Bạn hãy kiểm tra lại xem file 'clean_data_refined.csv' đang nằm ở thư mục nào?")
    exit()

# Load dữ liệu
df = pd.read_csv(FILE_PATH)
print("✅ Đã tải dữ liệu thành công!")

# 1. TÍNH TOÁN CHỈ SỐ (FEATURE ENGINEERING)
# BMI: Weight (kg) / Height (m)^2
df['BMI'] = df['Weight_kg'] / ((df['Height_cm'] / 100) ** 2)

# PSS_Score: Tổng điểm (đảo ngược câu 4, 5, 7, 8)
# Giả định thang đo 0-4 (chuẩn PSS-10). Đảo ngược = 4 - điểm cũ.
pss_cols = [f'PSS_{i}' for i in range(1, 11)]
positive_items = ['PSS_4', 'PSS_5', 'PSS_7', 'PSS_8']
# Tạo bản sao để tính toán
df_pss = df[pss_cols].copy()
for col in positive_items:
    df_pss[col] = 4 - df_pss[col]
df['PSS_Score'] = df_pss.sum(axis=1)

# MENQOL_Score: Trung bình cộng các triệu chứng
men_cols = [c for c in df.columns if c.startswith('MEN_') and c not in ['MENQOL_Score', 'Meno_Age', 'Meno_Age_Numeric', 'Meno_Group', 'Meno_Duration']]
df['MENQOL_Score'] = df[men_cols].mean(axis=1)

# 2. MÃ HÓA (ENCODING)
# Education
edu_map = {'Không đi học': 0, 'THCS': 2, 'THPT – trung cấp': 3, 'Đại học – cao đẳng': 4, 'Trên đại học': 5}
df['Education_Code'] = df['Education'].map(edu_map)

# Income (Gộp "Trên 10 triệu" vào nhóm 3: 11-20tr)
income_map = {'Dưới 5 triệu': 1, 'Từ 5 đến 10 triệu': 2, 'Trên 10 triệu': 3, 'Từ 11 đến 20 triệu': 3, 'Trên 20 triệu': 4}
df['Income_Code'] = df['Income'].map(income_map)

# Job
def map_job(job):
    job = str(job).lower()
    if 'nội trợ' in job: return 1
    if 'công nhân' in job: return 2
    if 'văn phòng' in job: return 3
    if 'kinh doanh' in job: return 4
    if 'về hưu' in job: return 5
    return 6 # Chuyên gia/Khác
df['Job_Code'] = df['Job'].apply(map_job)

# Marital Status
df['Marital_Code'] = df['Marital_Status'].apply(lambda x: 0 if 'độc thân' in str(x).lower() else 1)
def clean_meno_age(val):
    val_str = str(val).strip().lower()
    try:
        # Nếu là số -> giữ nguyên
        return float(val_str)
    except ValueError:
        # Nếu là chữ -> kiểm tra từ khóa
        keywords = ['chưa', 'không', 'vẫn', 'đều', 'sắp', 'đang']
        if any(kw in val_str for kw in keywords):
            return 0  # Quy ước 0 là chưa mãn kinh
        return 0 # Mặc định các trường hợp lạ khác về 0

# Áp dụng hàm làm sạch
df['Meno_Age_Clean'] = df['Meno_Age'].apply(clean_meno_age)

# Tạo trạng thái mãn kinh (1: Có, 0: Không)
df['Meno_Status'] = df['Meno_Age_Clean'].apply(lambda x: 1 if x > 0 else 0)

# Tính thời gian mãn kinh (Duration)
df['Meno_Duration_New'] = df.apply(
    lambda row: max(0, row['Age'] - row['Meno_Age_Clean']) if row['Meno_Status'] == 1 else 0,
    axis=1
)
# Lưu file
df.to_csv("clean_data_preprocessed.csv", index=False)