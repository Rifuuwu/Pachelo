from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import pandas as pd
import pickle
import re
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Fungsi untuk parsing log
def parse_apache_log_line(line):
    log_pattern = re.compile(r'(?P<ip>\S+) (?P<dash1>\S+) (?P<user>\S+) \[(?P<timestamp>[^\]]+)\] "(?P<method>\S+) (?P<url>\S+) (?P<protocol>\S+)" (?P<status>\d{3}) (?P<size>\d+) "(?P<referrer>[^\"]+)" "(?P<user_agent>[^\"]+)"')
    match = log_pattern.match(line)
    return match.groupdict() if match else None

def convert_log_to_csv(log_file):
    with open(log_file, 'r') as file:
        lines = file.readlines()
    log_data = [parse_apache_log_line(line) for line in lines]
    log_data = [entry for entry in log_data if entry]
    df = pd.DataFrame(log_data)
    csv_file = log_file.replace('.log', '.csv')
    df.to_csv(csv_file, index=False)
    return csv_file

# Load model, encoder, and scaler
model = load_model('ids.keras')
with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def upload_file():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    filepath = './uploads/' + file.filename
    os.makedirs('./uploads', exist_ok=True)
    file.save(filepath)

    if file.filename.endswith('.log'):
        filepath = convert_log_to_csv(filepath)

    new_data = pd.read_csv(filepath)
    features = ['status', 'user_agent']
    if not all(feature in new_data.columns for feature in features):
        return "File tidak memiliki kolom yang diperlukan", 400

    X_new = new_data[features]

    # Terapkan Label Encoding
    for col in features:
        if col in label_encoders:
            X_new[col] = label_encoders[col].transform(X_new[col].astype(str))

    # Standarisasi data baru
    X_new_scaled = scaler.transform(X_new)

    # Prediksi menggunakan model yang dimuat
    predictions = (model.predict(X_new_scaled) > 0.5).astype("int32").flatten()
    new_data['is_anomaly'] = predictions

    # Simpan hasil ke file baru
    output_path = "hasiltest.csv"
    new_data.to_csv(output_path, index=False)

    # Hitung jumlah anomali per IP
    anomaly_counts = new_data[new_data['is_anomaly'] == 1].groupby('ip').size().sort_values(ascending=False).head(10)
    top_anomalies = anomaly_counts.reset_index(name='anomali_counter')

    return render_template('result.html', top_anomalies=top_anomalies, download_link=output_path)

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(directory='.', path=filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
