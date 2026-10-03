import pandas as pd

# ── Column names for NSL-KDD ──────────────────────────────────────────────────
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

# ── Load the full test dataset ────────────────────────────────────────────────
print("📂 Loading KDDTest+.csv ...")
df = pd.read_csv('data/KDDTest+.csv', names=columns)

print(f"✅ Full dataset loaded: {df.shape[0]} rows")

# ── Take a balanced sample (mix of normal and attack) ─────────────────────────
normal  = df[df['label'] == 'normal'].sample(n=10, random_state=42)
attack  = df[df['label'] != 'normal'].sample(n=10, random_state=42)
sample  = pd.concat([normal, attack]).sample(frac=1, random_state=42)  # shuffle

# ── Drop label and difficulty (not needed for prediction input) ───────────────
sample_input = sample.drop(columns=['label', 'difficulty'])

# ── Save sample ───────────────────────────────────────────────────────────────
sample_input.to_csv('data/test_sample.csv', index=False)
print(f"\n✅ Sample saved: data/test_sample.csv")
print(f"   Rows  : {len(sample_input)} (10 normal + 10 attack, shuffled)")
print(f"   Columns: {len(sample_input.columns)}")

# ── Also save a labelled version so you can verify predictions ────────────────
sample_labelled = sample[['label'] + list(sample_input.columns)]
sample_labelled.to_csv('data/test_sample_labelled.csv', index=False)
print(f"✅ Labelled version saved: data/test_sample_labelled.csv")
print("   (Use this to verify if predictions are correct)\n")

# ── Preview ───────────────────────────────────────────────────────────────────
print("📋 Preview of labels in sample:")
print(sample[['label']].value_counts().to_string())
print("\n🎉 Done! Upload data/test_sample.csv to your web interface.")
