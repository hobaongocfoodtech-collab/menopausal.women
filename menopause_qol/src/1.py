import pandas as pd
import os

# 1. Định nghĩa đường dẫn đến file
# Dùng r"..." để Python không bị lỗi với dấu gạch chéo ngược (\) trên Windows
file_path = r"C:\Users\Admin\PycharmProjects\PNMK\icatsd2026_menopause_qol\data\raw\XLSL.PNMK01.xlsx"

try:
    # 2. Đọc file Excel
    print(f"Đang đọc file từ: {file_path} ...")

    # engine='openpyxl' là bắt buộc đối với file .xlsx
    df = pd.read_excel(file_path, engine='openpyxl')

    # 3. Kiểm tra dữ liệu
    print("\n--- Đọc thành công! 5 dòng đầu tiên ---")
    print(df.head())

    print("\n--- Thông tin các cột (Kiểm tra xem tên cột đã chuẩn chưa) ---")
    print(df.info())

    # 4. Kiểm tra kích thước dữ liệu
    print(f"\n--- Kích thước dataset: {df.shape[0]} dòng, {df.shape[1]} cột ---")

except FileNotFoundError:
    print(f"\nLỖI: Không tìm thấy file!")
    print(f"Bạn hãy kiểm tra lại xem file 'XLSL.PNMK01.xlsx' có thực sự nằm trong folder đó không.")
except Exception as e:
    print(f"\nCÓ LỖI XẢY RA: {e}")