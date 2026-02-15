import pandas as pd
import numpy as np
import os
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# --- CẤU HÌNH ---
INPUT_FILE = "clean_data_final.csv"
OUTPUT_TRAIN = "train_data.csv"
OUTPUT_TEST = "test_data.csv"

if not os.path.exists(INPUT_FILE):
    print(f"❌ LỖI: Không tìm thấy file '{INPUT_FILE}'.")
    exit()

print("--- BẮT ĐẦU GIAI ĐOẠN 3: PHÂN CỤM & CHIA TẬP DỮ LIỆU ---")
df = pd.read_csv(INPUT_FILE)
print(f"✅ Đã tải dữ liệu: {df.shape}")

# ==============================================================================
# 1. CHUẨN BỊ DỮ LIỆU ĐỂ PHÂN CỤM
# ==============================================================================
# Chúng ta chỉ dùng các biến NHÂN KHẨU HỌC để phân nhóm (Profile)
# Không dùng PSS_Score hay MENQOL_Score để phân cụm (để tránh rò rỉ dữ liệu)
clustering_cols = [
    'Age', 'BMI',
    'Education_Code', 'Income_Code', 'Job_Code', 'Marital_Code',
    'Meno_Duration_New', 'Meno_Status'
]

# Kiểm tra xem các cột này có đủ không
valid_cluster_cols = [c for c in clustering_cols if c in df.columns]
print(f"\n[1] Các đặc trưng dùng để phân cụm ({len(valid_cluster_cols)} biến):")
print(f"   {valid_cluster_cols}")

X_cluster = df[valid_cluster_cols]

# Chuẩn hóa dữ liệu (Scaling)
# Bước này bắt buộc vì 'Age' (40-60) lớn hơn nhiều so với 'Income_Code' (1-4)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# ==============================================================================
# 2. THỰC HIỆN K-MEANS CLUSTERING
# ==============================================================================
print("\n[2] Đang chạy K-Means (k=3)...")

# Chọn k=3 (3 nhóm người tiêu biểu)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Thống kê số lượng mỗi nhóm
counts = df['Cluster'].value_counts().sort_index()
print("   -> Số lượng thành viên mỗi nhóm:")
print(counts)

# ==============================================================================
# 3. ĐỊNH DANH CÁC NHÓM (CLUSTERING PROFILING)
# ==============================================================================
print("\n[3] Đặc điểm trung bình của từng nhóm:")

# Tính giá trị trung bình của các biến nhân khẩu học theo từng cụm
profile = df.groupby('Cluster')[valid_cluster_cols].mean()

# Hiển thị làm tròn 2 chữ số
print(profile.round(2).T)

print("\n   -> NHẬN XÉT SƠ BỘ (Dựa trên số liệu trên):")
for i in range(3):
    age = profile.loc[i, 'Age']
    income = profile.loc[i, 'Income_Code'] if 'Income_Code' in profile.columns else 0
    meno_dur = profile.loc[i, 'Meno_Duration_New']
    print(f"      * Cluster {i}: Tuổi TB ~{age:.1f}, Thu nhập mức ~{income:.1f}, Mãn kinh ~{meno_dur:.1f} năm.")

# ==============================================================================
# 4. CHIA TẬP TRAIN / TEST (STRATIFIED SPLIT)
# ==============================================================================
print("\n[4] Chia tập Train/Test (Tỷ lệ 80/20, Phân tầng theo Cluster)...")

# stratify=df['Cluster'] đảm bảo tập Test có đủ đại diện của cả 3 nhóm
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['Cluster']
)

print(f"   -> Tập Train: {train_df.shape[0]} mẫu")
print(f"   -> Tập Test : {test_df.shape[0]} mẫu")

# Kiểm tra tỷ lệ nhóm trong tập Test
test_counts = test_df['Cluster'].value_counts(normalize=True).sort_index()
print("   -> Tỷ lệ phân bố nhóm trong tập Test (nên tương đương tập gốc):")
print(test_counts.round(2))

# ==============================================================================
# 5. LƯU KẾT QUẢ
# ==============================================================================
train_df.to_csv(OUTPUT_TRAIN, index=False)
test_df.to_csv(OUTPUT_TEST, index=False)

print("\n" + "="*60)
print(f"✅ HOÀN TẤT GIAI ĐOẠN 3!")
print(f"   📂 File Train: {OUTPUT_TRAIN}")
print(f"   📂 File Test : {OUTPUT_TEST}")
print("="*60)