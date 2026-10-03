import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import time

# ── Load preprocessed data ────────────────────────────────────────────────────
print("📂 Loading preprocessed data...")
X_train = np.load('data/X_train.npy')
y_train = np.load('data/y_train.npy')
X_test  = np.load('data/X_test.npy')
y_test  = np.load('data/y_test.npy')
print(f"✅ Training samples: {X_train.shape[0]}")
print(f"✅ Testing samples:  {X_test.shape[0]}")

# ── Define models ─────────────────────────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree':       DecisionTreeClassifier(random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42)
}

os.makedirs('models', exist_ok=True)
results = {}

# ── Train and evaluate each model ─────────────────────────────────────────────
for name, model in models.items():
    print(f"\n🔧 Training: {name} ...")
    start = time.time()
    model.fit(X_train, y_train)
    duration = round(time.time() - start, 2)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

    print(f"✅ Done in {duration}s | Accuracy: {acc * 100:.2f}%")

    # Save each model
    filename = name.lower().replace(' ', '_')
    joblib.dump(model, f'models/{filename}.pkl')
    print(f"💾 Saved: models/{filename}.pkl")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n📊 ── MODEL COMPARISON ──────────────────────────────────")
for name, acc in results.items():
    bar = "█" * int(acc * 40)
    print(f"{name:<22} {acc*100:.2f}%  {bar}")

best = max(results, key=results.get)
print(f"\n🏆 Best Model: {best} ({results[best]*100:.2f}%)")
print("\n🎉 Training complete! All models saved to models/")
