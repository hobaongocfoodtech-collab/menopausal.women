import os
import shutil

# --- CẤU HÌNH ĐƯỜNG DẪN ---
# Thư mục chứa web sẽ nằm cùng cấp với dự án hiện tại của bạn
WEB_DIR = r"D:\PycharmProjects\PNMK\menopausal_women_web"
TEMPLATES_DIR = os.path.join(WEB_DIR, "templates")
# Đường dẫn file model gốc
ORIGINAL_MODEL = r"D:\PycharmProjects\PNMK\icatsd2026_menopause_qol\src1\final_health_advisor.pkl"


def setup_deployment():
    print("--- 🚀 ĐANG TRIỂN KHAI GIAO DIỆN WEB ADAPTIVE AI ---")

    # 1. Tạo cấu trúc thư mục
    if not os.path.exists(TEMPLATES_DIR):
        os.makedirs(TEMPLATES_DIR)
        print(f"✅ Đã tạo thư mục: {TEMPLATES_DIR}")

    # 2. Copy file model
    if os.path.exists(ORIGINAL_MODEL):
        shutil.copy(ORIGINAL_MODEL, os.path.join(WEB_DIR, "final_health_advisor.pkl"))
        print(f"✅ Đã sao chép model sang thư mục Web.")
    else:
        print(f"⚠️ Cảnh báo: Không tìm thấy model tại {ORIGINAL_MODEL}")

    # 3. Tạo file app.py (Backend)
    app_py_path = os.path.join(WEB_DIR, "app.py")
    app_code = """from flask import Flask, render_template, request, jsonify
import joblib
import os

app = Flask(__name__)

# Tải bộ não AI
MODEL_PATH = "final_health_advisor.pkl"
package = joblib.load(MODEL_PATH)
scaler = package['scaler']
kmeans = package['kmeans']
experts = package['experts']
strategy = package['strategy']
demo_feats = package['demo_feats']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_cluster', methods=['POST'])
def get_cluster():
    data = request.json
    user_demo = [float(data[feat]) for feat in demo_feats]
    scaled_demo = scaler.transform([user_demo])
    c_id = int(kmeans.predict(scaled_demo)[0])
    gold_qs = list(set(strategy[c_id]['PSS'] + strategy[c_id]['MEN']))
    return jsonify({"cluster": c_id, "questions": gold_qs})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    c_id = int(data['cluster'])
    full_user_data = data['full_data']
    results = {}
    for t_type in ['PSS', 'MEN']:
        features = demo_feats + strategy[c_id][t_type]
        input_data = [float(full_user_data[f]) for f in features]
        pred = experts[f"expert_{c_id}_{t_type}"].predict([input_data])[0]
        results[t_type] = round(float(pred), 2)
    return jsonify(results)

if __name__ == '__main__':
    print("🚀 Web AI đang chạy tại http://127.0.0.1:5000")
    app.run(debug=True)
"""
    with open(app_py_path, "w", encoding="utf-8") as f:
        f.write(app_code)

    # 4. Tạo file index.html (Frontend)
    index_html_path = os.path.join(TEMPLATES_DIR, "index.html")
    html_code = """<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <title>AI Health Advisor 🌸</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background: linear-gradient(135deg, #fff5f8 0%, #fce4ec 100%); min-height: 100vh; }
        .card { border-radius: 20px; border: none; box-shadow: 0 10px 30px rgba(255,105,180,0.15); }
        .btn-pink { background: #ff4081; color: white; border-radius: 30px; padding: 10px 30px; font-weight: bold; }
        .btn-pink:hover { background: #f50057; color: white; }
        .header-icon { font-size: 3rem; }
    </style>
</head>
<body>
<div class="container py-5">
    <div class="card p-5">
        <div class="text-center mb-4">
            <div class="header-icon">🌸</div>
            <h2 class="fw-bold">Hệ thống AI Thích nghi</h2>
            <p class="text-muted">Dự báo Căng thẳng & Chất lượng cuộc sống</p>
        </div>

        <div id="step1">
            <h5 class="mb-3">📍 Bước 1: Thông tin nền tảng</h5>
            <div class="row g-3">
                <div class="col-md-4"><label>Tuổi</label><input type="number" id="Age" class="form-control" value="50"></div>
                <div class="col-md-4"><label>BMI</label><input type="number" id="BMI" class="form-control" value="23"></div>
                <div class="col-md-4"><label>Số năm mãn kinh</label><input type="number" id="Meno_Duration_New" class="form-control" value="5"></div>
            </div>
            <button class="btn btn-pink w-100 mt-4" onclick="identify()">Xác định nhóm đối tượng 🚀</button>
        </div>

        <div id="step2" class="d-none mt-4">
            <h5 class="mb-3">✨ Bước 2: Câu hỏi chuyên biệt dành cho bạn</h5>
            <div id="questions-area" class="row g-3"></div>
            <button class="btn btn-pink w-100 mt-4" onclick="finalPredict()">Phân tích kết quả 📊</button>
        </div>

        <div id="result" class="d-none mt-5 p-4 border-start border-4 border-success bg-light">
            <h4 class="text-success fw-bold">🎯 Chẩn đoán từ AI:</h4>
            <h3 id="score-text" class="mt-2"></h3>
        </div>
    </div>
</div>

<script>
let clusterId = null;
let baseData = {};

async function identify() {
    baseData = {
        Age: document.getElementById('Age').value,
        BMI: document.getElementById('BMI').value,
        Meno_Duration_New: document.getElementById('Meno_Duration_New').value,
        Education_Code: 3, Income_Code: 2, Job_Code: 1, Marital_Code: 1, Meno_Status: 1, Chronic_Disease: 0
    };
    const res = await fetch('/get_cluster', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(baseData)
    });
    const data = await res.json();
    clusterId = data.cluster;

    const area = document.getElementById('questions-area');
    area.innerHTML = '';
    data.questions.forEach(q => {
        area.innerHTML += `<div class="col-md-6"><label class="small fw-bold text-secondary">${q}</label><input type="number" class="form-control gold-q" data-id="${q}"></div>`;
    });
    document.getElementById('step2').classList.remove('d-none');
    document.getElementById('step1').classList.add('opacity-50');
}

async function finalPredict() {
    let fullData = {...baseData};
    document.querySelectorAll('.gold-q').forEach(el => fullData[el.dataset.id] = el.value);
    const res = await fetch('/predict', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({cluster: clusterId, full_data: fullData})
    });
    const result = await res.json();
    document.getElementById('result').classList.remove('d-none');
    document.getElementById('score-text').innerHTML = `Stress (PSS): ${result.PSS} | Chất lượng sống (MEN): ${result.MEN}`;
}
</script>
</body>
</html>
"""
    with open(index_html_path, "w", encoding="utf-8") as f:
        f.write(html_code)

    print(f"\n✅ THÀNH CÔNG! Cấu trúc web đã sẵn sàng tại: {WEB_DIR}")
    print(f"👉 Để chạy web, hãy dùng Terminal gõ: python {os.path.join(WEB_DIR, 'app.py')}")


if __name__ == "__main__":
    setup_deployment()