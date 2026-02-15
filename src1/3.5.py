import pandas as pd
import numpy as np
import os
from imblearn.over_sampling import SMOTE
from collections import Counter

# --- CẤU HÌNH ---
INPUT_TRAIN = "train_data.csv"
OUTPUT_TRAIN_BALANCED = "train_data_balanced.csv"

if not os.path.exists(INPUT_TRAIN):
    print(f"❌ LỖI: Không tìm thấy file '{INPUT_TRAIN}'")
    exit()

print("--- GIAI ĐOẠN 3.5: TĂNG CƯỜNG DỮ LIỆU (FIXED) ---")
df_train = pd.read_csv(INPUT_TRAIN)

# --- BƯỚC QUAN TRỌNG: CHỈ LẤY CÁC CỘT DẠNG SỐ ---
# SMOTE không thể xử lý các cột văn bản như 'Độc thân', 'Nội trợ'...
df_numeric = df_train.select_dtypes(include=[np.number])

# Tách đặc trưng (X) và nhãn cụm (y)
X = df_numeric.drop(columns=['Cluster'])
y = df_numeric['Cluster']

print(f"📊 Phân bố cụm trước khi tăng cường: {Counter(y)}")

# Khởi tạo SMOTE
# k_neighbors=2 vì Cluster 0 chỉ có 5 mẫu (k < n_samples)
smote = SMOTE(sampling_strategy='auto', k_neighbors=2, random_state=42)

# Thực hiện sinh dữ liệu ảo
X_resampled, y_resampled = smote.fit_resample(X, y)

print(f"🚀 Phân bố cụm sau khi tăng cường: {Counter(y_resampled)}")

# Gộp lại thành dataframe mới
df_balanced = pd.concat([pd.DataFrame(X_resampled, columns=X.columns),
                         pd.Series(y_resampled, name='Cluster')], axis=1)

# Làm tròn các giá trị mã hóa (vì SMOTE sinh số thực kiểu 1.2, 1.8...)
# Chuyển về số nguyên cho các cột Code và Score
cols_to_round = [c for c in df_balanced.columns if '_Code' in c or 'PSS_Score' in c or 'Meno_Status' in c]
for col in cols_to_round:
    df_balanced[col] = df_balanced[col].round().astype(int)

# Lưu file train mới (chỉ chứa dữ liệu số, sẵn sàng cho ML)
df_balanced.to_csv(OUTPUT_TRAIN_BALANCED, index=False)

print("\n" + "="*60)
print(f"✅ HOÀN TẤT! Đã tạo ra: {OUTPUT_TRAIN_BALANCED}")
print(f"   Dữ liệu Cluster 0 đã được cân bằng lên {Counter(y_resampled)[0]} mẫu.")
print("="*60)
