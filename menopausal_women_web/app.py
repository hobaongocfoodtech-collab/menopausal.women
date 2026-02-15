from flask import Flask, render_template, request, jsonify
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
