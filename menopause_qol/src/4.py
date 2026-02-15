import pandas as pd
import os

# --- CẤU HÌNH ---
file_path = r"C:\Users\Admin\PycharmProjects\PNMK\icatsd2026_menopause_qol\data\processed\clean_data_final.csv"

try:
    print(f"--- ĐANG KIỂM TRA FILE: {os.path.basename(file_path)} ---")

    # 1. Đọc file CSV
    df = pd.read_csv(file_path)

    # 2. Tổng hợp số lượng Null theo từng cột
    null_counts = df.isnull().sum()

    # Lọc ra các cột có Null ( > 0)
    cols_with_null = null_counts[null_counts > 0]

    print(f"\nTổng số dòng dữ liệu: {len(df)}")

    if len(cols_with_null) == 0:
        print("\n✅ TUYỆT VỜI! File dữ liệu KHÔNG CÒN giá trị Null nào.")
        print("Bạn đã sẵn sàng để chạy mô hình AI.")
    else:
        print(f"\n⚠️ CẢNH BÁO: Vẫn còn {len(cols_with_null)} cột chứa giá trị Null:")
        print("-" * 50)
        print(f"{'Tên cột':<30} | {'Số lượng Null':<15} | {'Tỷ lệ %':<10}")
        print("-" * 50)

        for col, count in cols_with_null.items():
            percent = (count / len(df)) * 100
            print(f"{col:<30} | {count:<15} | {percent:.1f}%")

        print("-" * 50)

        # 3. Phân tích nguyên nhân (Quan trọng)
        print("\n🔍 PHÂN TÍCH NHANH:")
        if 'Meno_Age_Clean' in cols_with_null:
            print("- Cột 'Meno_Age_Clean' bị Null là HỢP LÝ nếu đó là những người CHƯA mãn kinh.")
            print("  (Bạn không cần lo lắng về cột này vì chúng ta đã có cột 'Is_PostMenopause' để thay thế).")

        if 'BMI' in cols_with_null:
            print("- Cột 'BMI' bị Null: Do thiếu Chiều cao hoặc Cân nặng -> Cần kiểm tra lại dữ liệu gốc.")

    # 4. Kiểm tra xem có dòng nào bị Null ở các cột quan trọng không?
    # Các cột này bắt buộc phải có số liệu để chạy AI
    critical_cols = ['Age', 'BMI', 'Education_Code', 'Income_Code', 'PSS_Score', 'MENQOL_Score']
    # Lọc các cột critical có trong df
    existing_critical = [c for c in critical_cols if c in df.columns]

    if df[existing_critical].isnull().any().any():
        print(f"\n❌ LỖI NGHIÊM TRỌNG: Có dòng bị thiếu dữ liệu ở các cột quan trọng ({existing_critical}):")
        bad_rows = df[df[existing_critical].isnull().any(axis=1)]
        print(bad_rows[existing_critical].head())
        print("-> Bạn nên xóa các dòng này hoặc điền trung bình cộng trước khi training.")

except FileNotFoundError:
    print(f"❌ Lỗi: Không tìm thấy file tại {file_path}")
except Exception as e:
    print(f"❌ Lỗi: {e}")