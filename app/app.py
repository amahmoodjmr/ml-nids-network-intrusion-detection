from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# ── Load model and preprocessors ──────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model    = joblib.load(os.path.join(BASE, 'models/random_forest.pkl'))
scaler   = joblib.load(os.path.join(BASE, 'models/scaler.pkl'))
encoders = joblib.load(os.path.join(BASE, 'models/encoders.pkl'))

COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate'
]

CAT_COLS = ['protocol_type', 'service', 'flag']


def preprocess_input(df):
    """Clean and scale uploaded CSV data."""
    # Drop label/difficulty if present
    for col in ['label', 'difficulty']:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    # Keep only expected columns
    df = df[COLUMNS]

    # Encode categoricals
    for col in CAT_COLS:
        le = encoders[col]
        df[col] = df[col].map(
            lambda x: int(le.transform([x])[0]) if x in le.classes_ else -1
        )

    return scaler.transform(df)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        df = pd.read_csv(file)
        X  = preprocess_input(df)

        predictions  = model.predict(X)
        probabilities = model.predict_proba(X)

        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            confidence = round(float(max(prob)) * 100, 2)
            results.append({
                'row':        i + 1,
                'prediction': 'Attack 🚨' if pred == 1 else 'Normal ✅',
                'confidence': f"{confidence}%",
                'status':     'danger' if pred == 1 else 'success'
            })

        total   = len(results)
        attacks = sum(1 for r in results if 'Attack' in r['prediction'])
        normal  = total - attacks

        return jsonify({
            'results': results,
            'summary': {
                'total':   total,
                'attacks': attacks,
                'normal':  normal
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("🚀 NIDS Web App running at http://127.0.0.1:5000")
    app.run(debug=True)
