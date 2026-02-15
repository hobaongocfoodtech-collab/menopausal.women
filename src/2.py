import pandas as pd
import os

# Đường dẫn file
file_path = r"C:\Users\Admin\PycharmProjects\PNMK\icatsd2026_menopause_qol\data\raw\XLSL.PNMK01.xlsx"

try:
    print(f"Đang kiểm tra file: {file_path}")

    # BƯỚC 1: Kiểm tra xem file có những Sheet nào
    excel_file = pd.ExcelFile(file_path, engine='openpyxl')
    sheet_names = excel_file.sheet_names
    print(f"\n--- Tìm thấy {len(sheet_names)} Sheet: {sheet_names} ---")

    # BƯỚC 2: Đọc Sheet bạn muốn
    # Cách 1: Đọc theo tên (Bạn thay 'Sheet2' bằng tên thực tế nếu code chạy sai)
    # target_sheet = 'Sheet2'

    # Cách 2: Đọc theo thứ tự (0 là sheet đầu, 1 là sheet thứ hai)
    # Tôi để mặc định là 1 (tức là sheet thứ 2) vì bạn bảo sheet kia đã sửa
    target_sheet = 1

    print(f"Đang đọc dữ liệu từ Sheet thứ tự: {target_sheet} ...")

    df = pd.read_excel(file_path, sheet_name=target_sheet, engine='openpyxl')

    # BƯỚC 3: Kiểm tra kết quả
    print("\n--- 5 dòng đầu tiên (Xem Header đã chuẩn tiếng Anh chưa) ---")
    print(df.head())

    print("\n--- Danh sách cột ---")
    print(df.columns.tolist())

except IndexError:
    print("\nLỖI: File chỉ có 1 Sheet, không tìm thấy Sheet thứ 2.")
    print("Hãy kiểm tra lại file Excel xem bạn đã lưu Sheet mới chưa.")
except Exception as e:
    print(f"\nCÓ LỖI XẢY RA: {e}")