import pandas as pd
from ydata_profiling import ProfileReport
import os
import warnings

# Tắt các cảnh báo không cần thiết
warnings.filterwarnings('ignore')

# 1. CẤU HÌNH
FILE_PATH = r"/menopause_qol\data\processed\clean_data_final.csv"
OUTPUT_HTML = r"D:\PycharmProjects\PNMK\icatsd2026_menopause_qol\reports\Tong_quan_du_lieu.html"


def generate_auto_report():
    print("--- ĐANG TẠO BÁO CÁO TỰ ĐỘNG (Vui lòng đợi 1-2 phút)... ---")

    try:
        # Kiểm tra file tồn tại
        if not os.path.exists(FILE_PATH):
            print(f"❌ LỖI: Không tìm thấy file dữ liệu tại {FILE_PATH}")
            return

        # Đọc dữ liệu
        df = pd.read_csv(FILE_PATH)
        print(f"-> Đã tải dữ liệu: {df.shape}")

        # Cấu hình báo cáo (ĐÃ SỬA LỖI: Xóa dark_mode=True)
        profile = ProfileReport(
            df,
            title="Báo cáo Dữ liệu Mãn kinh & CLCS (ICATSD 2026)",
            explorative=True  # Chế độ thám hiểm sâu (Deep Analysis)
        )

        # Tạo thư mục nếu chưa có
        os.makedirs(os.path.dirname(OUTPUT_HTML), exist_ok=True)

        # Xuất ra file HTML
        print("-> Đang tính toán thống kê và vẽ biểu đồ...")
        profile.to_file(OUTPUT_HTML)

        print(f"\n✅ ĐÃ XONG! Báo cáo được lưu tại:")
        print(f"-> {OUTPUT_HTML}")
        print("-> Hãy vào thư mục trên và mở file HTML bằng trình duyệt (Chrome/Edge).")

    except Exception as e:
        print(f"❌ LỖI HỆ THỐNG: {e}")


if __name__ == "__main__":
    generate_auto_report()