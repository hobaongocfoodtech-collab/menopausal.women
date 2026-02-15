import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

# --- CẤU HÌNH ---
# Đọc file kết quả từ bước trước (đang nằm cùng thư mục script hoặc đường dẫn bạn quy định)
# Giả sử file code này chạy cùng thư mục với file csv vừa tạo
INPUT_FILE = "clean_data_preprocessed.csv"
OUTPUT_FILE = "clean_data_final.csv"

# Kiểm tra file đầu vào
if not os.path.exists(INPUT_FILE):
    print(f"❌ LỖI: Không tìm thấy file '{INPUT_FILE}'. Hãy chạy code Giai đoạn 1 trước!")
    exit()

print("--- BẮT ĐẦU GIAI ĐOẠN 2: LÀM SẠCH & CHỌN LỌC ĐẶC TRƯNG ---")
df = pd.read_csv(INPUT_FILE)
print(f"✅ Đã tải dữ liệu: {df.shape}")

# ==============================================================================
# 1. XỬ LÝ GIÁ TRỊ RỖNG (NULL HANDLING)
# ==============================================================================
print("\n[1] Xử lý giá trị khuyết thiếu (Null)...")

# Danh sách biến số liên tục (Continuous Vars)
numeric_cols = ['Age', 'BMI', 'PSS_Score', 'MENQOL_Score', 'Meno_Age_Clean', 'Meno_Duration_New']
# Danh sách biến phân loại (Categorical Vars) - Đã mã hóa ở bước 1
cat_cols = ['Education_Code', 'Income_Code', 'Job_Code', 'Marital_Code', 'Meno_Status']

# Điền Null cho biến số bằng Median (Trung vị)
for col in numeric_cols:
    if col in df.columns:
        median_val = df[col].median()
        if df[col].isnull().sum() > 0:
            print(f"   -> Điền {df[col].isnull().sum()} dòng Null ở cột '{col}' bằng Median ({median_val:.2f})")
            df[col] = df[col].fillna(median_val)

# Điền Null cho biến phân loại bằng Mode (Giá trị xuất hiện nhiều nhất)
for col in cat_cols:
    if col in df.columns:
        mode_val = df[col].mode()[0]
        if df[col].isnull().sum() > 0:
            print(f"   -> Điền {df[col].isnull().sum()} dòng Null ở cột '{col}' bằng Mode ({mode_val})")
            df[col] = df[col].fillna(mode_val)

# ==============================================================================
# 2. MÃ HÓA NHỊ PHÂN BIẾN VĂN BẢN (TEXT TO BINARY)
# ==============================================================================
print("\n[2] Mã hóa biến văn bản (Text -> 0/1)...")

text_binary_cols = ['Chronic_Disease', 'Supp_Calcium', 'Supp_Omega3', 'Exercise']

for col in text_binary_cols:
    if col in df.columns:
        # Quy tắc: Nếu ô chứa chữ (khác nan/trống/không) -> 1, ngược lại -> 0
        # Chuyển về chữ thường để so sánh
        def encode_binary(val):
            val_str = str(val).strip().lower()
            if val_str in ['nan', '', '0', 'none']: return 0
            if 'không' in val_str: return 0  # Từ khóa phủ định tiếng Việt
            return 1  # Có dữ liệu text (VD: "Xơ gan", "Có", "Thường xuyên") -> 1


        # Tạo tên cột mới (VD: Chronic_Disease_Code) hoặc đè lên cột cũ
        # Ở đây ta đè lên cột cũ để gọn dữ liệu
        df[col] = df[col].apply(encode_binary)
        print(f"   -> Đã mã hóa '{col}': {df[col].value_counts().to_dict()}")

# ==============================================================================
# 3. KIỂM ĐỊNH PHƯƠNG SAI & LOẠI BỎ (LOW VARIANCE FILTER)
# ==============================================================================
print("\n[3] Kiểm định & Loại bỏ biến ít thông tin...")

cols_to_drop = []
THRESHOLD = 0.95  # Ngưỡng 95% (Nếu 95% người giống hệt nhau thì biến này vô dụng)

for col in text_binary_cols:
    if col in df.columns:
        counts = df[col].value_counts(normalize=True)  # Tính tỷ lệ phần trăm
        max_ratio = counts.max()  # Lấy tỷ lệ của giá trị phổ biến nhất

        if max_ratio > THRESHOLD:
            print(f"   ⚠️ Cảnh báo: Cột '{col}' có {max_ratio:.1%} giá trị giống nhau -> LOẠI BỎ.")
            cols_to_drop.append(col)
        else:
            print(f"   ✅ Cột '{col}' có phân bố tốt (Dominant: {max_ratio:.1%}) -> GIỮ LẠI.")

if cols_to_drop:
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"   -> Đã xóa {len(cols_to_drop)} cột: {cols_to_drop}")
else:
    print("   -> Không có cột nào bị loại bỏ.")

# ==============================================================================
# 4. LƯU KẾT QUẢ
# ==============================================================================
# Chỉ giữ lại các cột dạng số đã xử lý để đưa vào Machine Learning
# (Loại bỏ các cột text gốc như 'Job', 'Income'...)
final_cols = [c for c in df.columns if df[c].dtype in ['int64', 'float64', 'int32', 'float32']]

# Hoặc nếu bạn muốn giữ lại tất cả để đối chiếu thì dùng dòng dưới (nhưng cẩn thận khi train model)
df_final = df  # df[final_cols]

df_final.to_csv(OUTPUT_FILE, index=False)
print("\n" + "=" * 60)
print(f"✅ HOÀN TẤT GIAI ĐOẠN 2! File kết quả: {OUTPUT_FILE}")
print(f"📊 Kích thước cuối cùng: {df_final.shape}")
print("=" * 60)

# In thử 5 dòng đầu
print(df_final[
          ['Age', 'BMI', 'PSS_Score', 'MENQOL_Score'] + [c for c in text_binary_cols if c in df_final.columns]].head())