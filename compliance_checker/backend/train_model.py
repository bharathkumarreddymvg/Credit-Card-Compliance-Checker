"""
Credit Card Compliance Checker - Dataset Generation + Model Training
Rules for COMPLIANT:
  - Interest Rate <= 36%
  - Late Payment Fee <= 1000 INR
  - Annual Fee <= 5000 INR
  - Billing Cycle between 25-31 days
  - Minimum Payment >= 5%
  - Disclosure Provided = Yes
Non-compliant if 2+ rules are violated (adds realism vs hard AND logic)
"""

import numpy as np
import pandas as pd
import pickle
import json
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

np.random.seed(42)
N = 300

def generate_dataset(n):
    data = []
    for _ in range(n):
        interest_rate     = round(np.random.uniform(8, 60), 2)
        late_payment_fee  = round(np.random.uniform(100, 2000), 0)
        annual_fee        = round(np.random.uniform(0, 10000), 0)
        billing_cycle     = int(np.random.uniform(15, 45))
        min_payment       = round(np.random.uniform(1, 15), 2)
        disclosure        = int(np.random.choice([0, 1], p=[0.25, 0.75]))  # 1=Yes, 0=No

        # Rule violations
        violations = 0
        if interest_rate > 36:       violations += 1
        if late_payment_fee > 1000:  violations += 1
        if annual_fee > 5000:        violations += 1
        if billing_cycle < 25 or billing_cycle > 31: violations += 1
        if min_payment < 5:          violations += 1
        if disclosure == 0:          violations += 1

        # Compliant if fewer than 2 violations (with tiny noise)
        if violations <= 1:
            label = 1  # Compliant
        elif violations >= 3:
            label = 0  # Non-Compliant
        else:
            # 2 violations → 30% chance still compliant (minor borderline)
            label = int(np.random.random() > 0.7)

        data.append([interest_rate, late_payment_fee, annual_fee,
                     billing_cycle, min_payment, disclosure, label])

    cols = ['interest_rate', 'late_payment_fee', 'annual_fee',
            'billing_cycle', 'min_payment', 'disclosure', 'label']
    return pd.DataFrame(data, columns=cols)


df = generate_dataset(N)
print("Dataset shape:", df.shape)
print("Label distribution:\n", df['label'].value_counts())
print(df.head(10))

# Save dataset
df.to_csv('/home/claude/ml_backend/credit_card_data.csv', index=False)

# Train model
X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {acc:.2%}")
print(classification_report(y_test, y_pred, target_names=['Non-Compliant', 'Compliant']))

# Feature importances
feature_names = X.columns.tolist()
importances = model.feature_importances_
fi = dict(zip(feature_names, [round(float(v), 4) for v in importances]))
print("\nFeature Importances:", fi)

# Save model + metadata
with open('/home/claude/ml_backend/model.pkl', 'wb') as f:
    pickle.dump(model, f)

metadata = {
    'feature_names': feature_names,
    'feature_importances': fi,
    'accuracy': round(float(acc), 4),
    'classes': ['Non-Compliant', 'Compliant'],
    'rules': {
        'interest_rate': {'threshold': 36, 'direction': 'lte', 'label': 'Interest Rate ≤ 36%'},
        'late_payment_fee': {'threshold': 1000, 'direction': 'lte', 'label': 'Late Fee ≤ ₹1,000'},
        'annual_fee': {'threshold': 5000, 'direction': 'lte', 'label': 'Annual Fee ≤ ₹5,000'},
        'billing_cycle': {'threshold_min': 25, 'threshold_max': 31, 'direction': 'range', 'label': 'Billing Cycle 25–31 days'},
        'min_payment': {'threshold': 5, 'direction': 'gte', 'label': 'Min Payment ≥ 5%'},
        'disclosure': {'threshold': 1, 'direction': 'eq', 'label': 'Disclosure Provided'},
    }
}

with open('/home/claude/ml_backend/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("\n✅ Model and metadata saved successfully!")
