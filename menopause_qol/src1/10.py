import os
from fpdf import FPDF
import warnings

warnings.filterwarnings('ignore')

# --- CẤU HÌNH ---
SOURCE_DIR = r"/menopause_qol\src1"
OUTPUT_FILE = "Tong_Hop_Ma_Nguon_Luan_An.pdf"
# Đường dẫn font hỗ trợ tiếng Việt (Sử dụng font có sẵn trên Windows)
FONT_PATH = r"C:\Windows\Fonts\arial.ttf"


class CodePDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, "Báo cáo Mã nguồn Hệ thống Adaptive Health Advisor - ICATSD 2026", 0, 1, "R")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Trang {self.page_no()}", 0, 0, "C")


def export_src_to_pdf():
    pdf = CodePDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Đăng ký font hỗ trợ tiếng Việt
    if os.path.exists(FONT_PATH):
        pdf.add_font("Arial", "", FONT_PATH)
        pdf.add_font("Arial", "B", FONT_PATH)
        pdf.set_font("Arial", size=10)
    else:
        print("⚠️ Không tìm thấy font Arial.ttf, kết quả có thể lỗi hiển thị tiếng Việt.")
        pdf.set_font("Courier", size=10)

    # Lấy danh sách file .py và sắp xếp theo tên (1, 2, 3...)
    files = [f for f in os.listdir(SOURCE_DIR) if f.endswith('.py')]
    files.sort(key=lambda x: str(x))  # Sắp xếp theo thứ tự số

    print(f"--- ĐANG TỔNG HỢP {len(files)} FILES SANG PDF ---")

    for filename in files:
        file_path = os.path.join(SOURCE_DIR, filename)

        # Thêm trang mới cho mỗi file
        pdf.add_page()

        # Tiêu đề file
        pdf.set_font("Arial", "B", 14)
        pdf.set_text_color(0, 51, 102)  # Màu xanh thẫm
        pdf.cell(0, 10, f"FILE: {filename}", ln=True)
        pdf.ln(2)

        # Vẽ đường kẻ ngang
        pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + 190, pdf.get_y())
        pdf.ln(5)

        # Đọc nội dung code
        pdf.set_font("Arial", "", 9)
        pdf.set_text_color(0, 0, 0)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                code_content = f.read()

            # Ghi nội dung vào PDF (Xử lý đa dòng)
            pdf.multi_cell(0, 5, code_content)
        except Exception as e:
            pdf.cell(0, 10, f"Lỗi khi đọc file: {str(e)}", ln=True)

        print(f"   + Đã thêm: {filename}")

    # Xuất file
    pdf.output(OUTPUT_FILE)
    print("\n" + "=" * 50)
    print(f"✅ HOÀN TẤT! File PDF đã được tạo: {OUTPUT_FILE}")
    print("=" * 50)


if __name__ == "__main__":
    export_src_to_pdf()