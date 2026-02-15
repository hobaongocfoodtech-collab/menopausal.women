---

# 🌸 Adaptive AI: Menopause Health & Quality of Life Advisor 🌸

## 🎯 Đề Tài Luận Án Thạc Sĩ

> **"MỐI LIÊN HỆ CỦA GIAI ĐOẠN MÃN KINH ĐẾN MỨC ĐỘ CĂNG THẲNG VÀ CHẤT LƯỢNG CUỘC SỐNG CỦA PHỤ NỮ TỪ 45 ĐẾN 65 TUỔI TẠI THÀNH PHỐ HỒ CHÍ MINH"**

---

## 🏆 KẾT QUẢ THỰC NGHIỆM ĐỘT PHÁ (SENSATIONAL RESULTS) 🏆

Hệ thống AI thích nghi (Adaptive Expert System) đã chứng minh sức mạnh vượt trội với độ chính xác gần như tuyệt đối trên tập dữ liệu kiểm thử.

| Chỉ số đánh giá | Mức độ Căng thẳng () | Chất lượng cuộc sống () | Trạng thái |
| --- | --- | --- | --- |
| **Hệ số xác định ()** | **0.9730** | **0.9801** | 🔥 **Perfect** |
| **Sai số trung bình ()** | **1.0666** | **0.0844** | 🎯 **Ultra Precise** |

---

## 🚀 TẠI SAO DỰ ÁN NÀY KHÁC BIỆT?

Dự án sử dụng cơ chế **Hệ thống chuyên gia thích nghi (Adaptive Expert System)** để giải quyết bài toán sức khỏe tâm lý:

* **🧬 Phân cụm thông minh (K-Means):** Tự động nhận diện 3 nhóm đối tượng dựa trên đặc điểm nhân khẩu học riêng biệt.
* **🗳️ Bầu chọn đặc trưng (Ensemble Selection):** Kết hợp **ExtraTrees**, **RFE** và **Mutual Information** để trích xuất **5 Câu hỏi vàng** cho từng nhóm.
* **⚖️ Cân bằng dữ liệu (SMOTE):** Xử lý triệt để vấn đề mất cân bằng mẫu, giúp mô hình công bằng với mọi đối tượng.
* **🔍 Trí tuệ nhân tạo minh bạch (SHAP):** Sử dụng giá trị SHAP để giải thích rõ ràng trọng số của từng triệu chứng ảnh hưởng đến sức khỏe.

---

## 🛠️ PIPELINE XỬ LÝ (STEP-BY-STEP)

1. **📊 EDA & Profiling:** Khám phá dữ liệu tự động với `ydata-profiling`.
2. **🧹 Cleaning:** Chuẩn hóa dữ liệu thô và tính điểm  theo quy tắc mapping nghiêm ngặt.
3. **🧩 Clustering:** Phân tách 3 cụm đối tượng chuyên biệt bằng K-Means.
4. **📈 SMOTE:** Nhân bản mẫu giúp mô hình học sâu hơn ở các nhóm yếu thế.
5. **🎖️ Ensemble Selection:** Bầu chọn bộ câu hỏi tối ưu nhất qua cơ chế Voting.
6. **⚙️ Fine-tuning:** Tinh chỉnh siêu tham số bằng `GridSearchCV`.
7. **🧠 Packaging:** Đóng gói toàn bộ mô hình vào tệp `.pkl` sẵn sàng triển khai.
8. **💬 Interactive Web App:** Giao diện Flask thông minh cho phép người dùng tự kiểm tra sức khỏe.

---

## 📂 CẤU TRÚC THƯ MỤC DỰ ÁN (PROJECT ARCHITECTURE)

Hệ thống được tổ chức khoa học theo mô hình Nghiên cứu & Triển khai (Research & Deployment).

```bash
📁 PNMK
├── 📁 menopause_qol/            # 🔬 PHẦN NGHIÊN CỨU (RESEARCH)
│   ├── 📁 data/                 # Dữ liệu gốc (Raw) và dữ liệu sạch (Processed)
│   ├── 📁 models/               # Các mô hình chuyên gia đã huấn luyện (.pkl)
│   ├── 📁 reports/              # Báo cáo thống kê, biểu đồ SHAP & Residuals
│   ├── 📁 src1/                 # Toàn bộ Pipeline xử lý (0.py - 15.py)
│   ├── 📄 requirements.txt      # Danh sách thư viện nghiên cứu
│   └── 📄 README.md             # Tài liệu này
├── 📁 menopausal_women_web/     # 🌐 PHẦN TRIỂN KHAI (DEPLOYMENT)
│   ├── 📁 templates/            # Giao diện Web (index.html)
│   ├── 📄 app.py                # Server chính chạy Flask
│   └── 📄 final_health_advisor.pkl # "Bộ não" AI đã đóng gói
└── 📄 requirements.txt          # Danh sách thư viện chung

```

---

## 🛠️ HƯỚNG DẪN CÀI ĐẶT & VẬN HÀNH

### 1️⃣ Cài đặt môi trường

Mở Terminal và chạy lệnh sau để cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt

```

### 2️⃣ Chạy ứng dụng Web (Demo)

Để trải nghiệm hệ thống tư vấn thích nghi trên trình duyệt:

```bash
cd menopausal_women_web
python app.py

```

Sau đó truy cập địa chỉ: `http://127.0.0.1:5000`

### 3️⃣ Chạy Pipeline Nghiên cứu

Để tái lập quá trình huấn luyện và kiểm thử:

```bash
cd menopause_qol/src1
python 7.py  # Kiểm thử hệ thống

```

---

## ✨ TÁC GIẢ & LIÊN HỆ

* **Researcher:** **Hồ Bảo Ngọc** - *Master Student at Ho Chi Minh City*
* **OrcID:** [0009-0007-0746-9521](https://orcid.org/my-orcid?orcid=0009-0007-0746-9521) 🧪
* **LinkedIn:** [Ngọc Bảo FoodTech](www.linkedin.com/in/ngoc-bao-foodtech) 🌏

---

> **"Technology is best when it brings people together and improves life."** 🌸

---
