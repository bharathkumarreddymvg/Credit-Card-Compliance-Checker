"""
Flask Backend - Credit Card Compliance Checker API
Endpoints:
  GET  /health       → health check
  POST /predict      → returns prediction, confidence, feature importances, rule violations
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import json
import numpy as np
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model + metadata on startup
with open(os.path.join(BASE_DIR, 'model.pkl'), 'rb') as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, 'metadata.json'), 'r') as f:
    metadata = json.load(f)

FEATURE_NAMES = metadata['feature_names']
CLASSES = metadata['classes']
RULES = metadata['rules']
FEATURE_IMPORTANCES = metadata['feature_importances']


def check_rule_violations(data: dict) -> list:
    """Return list of violated rules with human-readable messages."""
    violations = []

    ir = data.get('interest_rate', 0)
    lpf = data.get('late_payment_fee', 0)
    af = data.get('annual_fee', 0)
    bc = data.get('billing_cycle', 28)
    mp = data.get('min_payment', 5)
    disc = data.get('disclosure', 1)

    if ir > 36:
        violations.append({
            'field': 'interest_rate',
            'rule': 'Interest Rate must be ≤ 36%',
            'actual': f'{ir}%',
            'expected': '≤ 36%',
            'severity': 'high' if ir > 45 else 'medium'
        })
    if lpf > 1000:
        violations.append({
            'field': 'late_payment_fee',
            'rule': 'Late Payment Fee must be ≤ ₹1,000',
            'actual': f'₹{lpf:,.0f}',
            'expected': '≤ ₹1,000',
            'severity': 'high' if lpf > 1500 else 'medium'
        })
    if af > 5000:
        violations.append({
            'field': 'annual_fee',
            'rule': 'Annual Fee must be ≤ ₹5,000',
            'actual': f'₹{af:,.0f}',
            'expected': '≤ ₹5,000',
            'severity': 'medium'
        })
    if bc < 25 or bc > 31:
        violations.append({
            'field': 'billing_cycle',
            'rule': 'Billing Cycle must be between 25–31 days',
            'actual': f'{bc} days',
            'expected': '25–31 days',
            'severity': 'medium'
        })
    if mp < 5:
        violations.append({
            'field': 'min_payment',
            'rule': 'Minimum Payment must be ≥ 5%',
            'actual': f'{mp}%',
            'expected': '≥ 5%',
            'severity': 'medium'
        })
    if disc == 0:
        violations.append({
            'field': 'disclosure',
            'rule': 'Disclosure must be provided',
            'actual': 'Not Provided',
            'expected': 'Provided',
            'severity': 'high'
        })

    return violations


def get_feature_contributions(input_data: dict) -> list:
    """Map feature importances to human-readable contributions for this input."""
    contributions = []
    display_names = {
        'interest_rate': 'Interest Rate',
        'late_payment_fee': 'Late Payment Fee',
        'annual_fee': 'Annual Fee',
        'billing_cycle': 'Billing Cycle',
        'min_payment': 'Minimum Payment',
        'disclosure': 'Disclosure Status',
    }
    for feat, importance in sorted(FEATURE_IMPORTANCES.items(), key=lambda x: -x[1]):
        contributions.append({
            'feature': feat,
            'label': display_names.get(feat, feat),
            'importance': importance,
            'importance_pct': round(importance * 100, 1),
            'value': input_data.get(feat)
        })
    return contributions


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_accuracy': metadata['accuracy'],
        'features': FEATURE_NAMES
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        body = request.get_json(force=True)

        # Parse and validate inputs
        interest_rate    = float(body.get('interest_rate', 0))
        late_payment_fee = float(body.get('late_payment_fee', 0))
        annual_fee       = float(body.get('annual_fee', 0))
        billing_cycle    = int(body.get('billing_cycle', 28))
        min_payment      = float(body.get('min_payment', 5))
        disclosure       = 1 if str(body.get('disclosure', 'yes')).lower() in ('yes', '1', 'true') else 0

        input_data = {
            'interest_rate': interest_rate,
            'late_payment_fee': late_payment_fee,
            'annual_fee': annual_fee,
            'billing_cycle': billing_cycle,
            'min_payment': min_payment,
            'disclosure': disclosure,
        }

        # Build feature vector
        X = np.array([[
            interest_rate,
            late_payment_fee,
            annual_fee,
            billing_cycle,
            min_payment,
            disclosure,
        ]])

        # Predict
        prediction_idx = model.predict(X)[0]
        probabilities  = model.predict_proba(X)[0]
        confidence     = float(max(probabilities))
        label          = CLASSES[prediction_idx]

        # Rule violations
        violations = check_rule_violations(input_data)

        # Feature contributions
        contributions = get_feature_contributions(input_data)

        response = {
            'prediction': label,
            'is_compliant': bool(prediction_idx == 1),
            'confidence': round(confidence * 100, 1),
            'probabilities': {
                'Non-Compliant': round(float(probabilities[0]) * 100, 1),
                'Compliant': round(float(probabilities[1]) * 100, 1),
            },
            'rule_violations': violations,
            'violation_count': len(violations),
            'feature_contributions': contributions,
            'model_accuracy': metadata['accuracy'],
            'input_summary': input_data,
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    print("🚀 Starting Credit Card Compliance Checker API on port 5050")
    app.run(host='0.0.0.0', port=5050, debug=True)
