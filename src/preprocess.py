import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

# ── Column names ──────────────────────────────────────────────────────────────
columns = [
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
    'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

def preprocess(filepath, is_train=True, encoders=None, scaler=None):
    print(f"\n📂 Loading: {filepath}")
    df = pd.read_csv(filepath, names=columns)

    # ── Drop difficulty column ────────────────────────────────────────────────
    df.drop('difficulty', axis=1, inplace=True)

    # ── Convert label to binary (normal=0, attack=1) ──────────────────────────
    df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)
    print("✅ Labels converted: normal=0, attack=1")

    # ── Encode categorical columns ────────────────────────────────────────────
    cat_cols = ['protocol_type', 'service', 'flag']

    if is_train:
        encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        print("✅ Categorical columns encoded")

        # Save encoders
        os.makedirs('models', exist_ok=True)
        joblib.dump(encoders, 'models/encoders.pkl')
        print("✅ Encoders saved to models/encoders.pkl")
    else:
        # Use existing encoders on test data
        for col in cat_cols:
            le = encoders[col]
            df[col] = df[col].map(lambda x: le.transform([x])[0]
                                  if x in le.classes_ else -1)
        print("✅ Categorical columns encoded using saved encoders")

    # ── Split features and labels ─────────────────────────────────────────────
    X = df.drop('label', axis=1)
    y = df['label']

    # ── Scale features ────────────────────────────────────────────────────────
    if is_train:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, 'models/scaler.pkl')
        print("✅ Scaler fitted and saved to models/scaler.pkl")
    else:
        X_scaled = scaler.transform(X)
        print("✅ Test data scaled using saved scaler")

    print(f"✅ Final shape — X: {X_scaled.shape}, y: {y.shape}")
    return X_scaled, y, encoders, scaler


# ── Run preprocessing ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Preprocess training data
    X_train, y_train, encoders, scaler = preprocess(
        'data/KDDTrain+.csv', is_train=True
    )

    # Preprocess test data using same encoders and scaler
    X_test, y_test, _, _ = preprocess(
        'data/KDDTest+.csv', is_train=False,
        encoders=encoders, scaler=scaler
    )

    # Save processed arrays
    np.save('data/X_train.npy', X_train)
    np.save('data/y_train.npy', y_train)
    np.save('data/X_test.npy', X_test)
    np.save('data/y_test.npy', y_test)

    print("\n🎉 Preprocessing complete! Arrays saved to data/")
