import pandas as pd
import os

# --- CẤU HÌNH ---
file_path = r"C:\Users\Admin\PycharmProjects\PNMK\icatsd2026_menopause_qol\data\processed\clean_data_final.csv"

try:
    print(f"--- KIỂM TRA TÍNH NHẤT QUÁN CỦA DỮ LIỆU ---")
    df = pd.read_csv(file_path)

    # 1. Lấy danh sách các cột dạng Chữ (Object/Text)
    # Loại bỏ các cột Code, ID, và các cột số
    text_cols = df.select_dtypes(include=['object']).columns.tolist()

    print(f"Tìm thấy {len(text_cols)} cột dạng Text: {text_cols}\n")

    for col in text_cols:
        print(f"=== KIỂM TRA CỘT: {col} ===")
        # Đếm số lượng giá trị xuất hiện (value_counts)
        counts = df[col].value_counts()

        # In ra màn hình
        print(counts)
        print("-" * 30)

        # Cảnh báo nếu có quá nhiều giá trị lạ (xuất hiện chỉ 1 lần)
        single_values = counts[counts == 1]
        if not single_values.empty:
            print(f"⚠️ CẢNH BÁO: Có {len(single_values)} giá trị chỉ xuất hiện 1 lần (Nghi ngờ lỗi chính tả):")
            print(single_values.index.tolist())
        print("\n")

    # 2. KIỂM TRA CHEO (CROSS-CHECK) GIỮA TEXT VÀ CODE
    # Kiểm tra xem việc Mapping có bị lỗi về 0 hết không?
    print("=== KIỂM TRA LOGIC MAPPING (TEXT vs CODE) ===")

    pairs_to_check = [
        ('Job', 'Job_Code'),
        ('Education', 'Education_Code'),
        ('Income', 'Income_Code'),
        ('Marital_Status', 'Marital_Code')
    ]

    for text_col, code_col in pairs_to_check:
        if text_col in df.columns and code_col in df.columns:
            print(f"\nKiểm tra cặp: {text_col} -> {code_col}")
            # Group by để xem mỗi Text đang được gán Code mấy
            check = df.groupby([text_col, code_col]).size().reset_index(name='Count')
            print(check)

            # Cảnh báo nếu Code = 0 (tức là không map được)
            zeros = check[check[code_col] == 0]
            if not zeros.empty:
                print(f"⚠️ CẢNH BÁO: Các giá trị sau đây bị gán Code = 0 (Do sai chính tả hoặc thiếu trong từ điển):")
                print(zeros[text_col].tolist())

except Exception as e:
    print(f"Lỗi: {e}")