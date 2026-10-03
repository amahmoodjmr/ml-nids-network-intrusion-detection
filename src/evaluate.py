import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)
import os

os.makedirs('static', exist_ok=True)

# ── Load test data ────────────────────────────────────────────────────────────
print("📂 Loading test data...")
X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy')

# ── Load models ───────────────────────────────────────────────────────────────
model_files = {
    'Logistic Regression': 'models/logistic_regression.pkl',
    'Decision Tree':       'models/decision_tree.pkl',
    'Random Forest':       'models/random_forest.pkl'
}

summary = []

for name, path in model_files.items():
    print(f"\n📊 Evaluating: {name}")
    model  = joblib.load(path)
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)

    summary.append({'Model': name, 'Accuracy': acc,
                    'Precision': prec, 'Recall': rec, 'F1': f1})

    print(f"  Accuracy : {acc*100:.2f}%")
    print(f"  Precision: {prec*100:.2f}%")
    print(f"  Recall   : {rec*100:.2f}%")
    print(f"  F1-Score : {f1*100:.2f}%")
    print("\n  Full Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['Normal', 'Attack']))

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'])
    plt.title(f'Confusion Matrix — {name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    fname = name.lower().replace(' ', '_')
    plt.tight_layout()
    plt.savefig(f'app/static/cm_{fname}.png')
    plt.close()
    print(f"  💾 Confusion matrix saved: app/static/cm_{fname}.png")

# ── Comparison bar chart ──────────────────────────────────────────────────────
metrics  = ['Accuracy', 'Precision', 'Recall', 'F1']
names    = [s['Model'] for s in summary]
x        = np.arange(len(metrics))
width    = 0.25
colors   = ['#4C72B0', '#DD8452', '#55A868']

fig, ax = plt.subplots(figsize=(10, 6))
for i, (s, color) in enumerate(zip(summary, colors)):
    vals = [s[m] for m in metrics]
    ax.bar(x + i * width, vals, width, label=s['Model'], color=color)

ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x + width)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.1)
ax.legend()
plt.tight_layout()
plt.savefig('app/static/model_comparison.png')
plt.close()
print("\n💾 Comparison chart saved: app/static/model_comparison.png")

# ── Feature importance (Random Forest only) ───────────────────────────────────
rf = joblib.load('models/random_forest.pkl')
importances = rf.feature_importances_

feature_names = [
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

indices = np.argsort(importances)[::-1][:15]  # top 15 features
plt.figure(figsize=(10, 6))
plt.bar(range(15), importances[indices], color='steelblue')
plt.xticks(range(15), [feature_names[i] for i in indices], rotation=45, ha='right')
plt.title('Top 15 Feature Importances — Random Forest')
plt.tight_layout()
plt.savefig('app/static/feature_importance.png')
plt.close()
print("💾 Feature importance chart saved: app/static/feature_importance.png")

print("\n🎉 Evaluation complete!")
