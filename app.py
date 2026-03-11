"""
=============================================================================
PHASE 4: Flask Web Application Backend
Hypertension Prediction System
=============================================================================
"""

from flask import Flask, render_template, request, flash
import joblib
import numpy as np
import os

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-in-production'

# Load trained model with error handling
try:
    model = joblib.load("logreg_model.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    print("WARNING: Model file not found. Using dummy predictions.")
    model = None
    scaler = None

# Mapping back numeric prediction to original stage
stage_map = {
    0: 'NORMAL',
    1: 'HYPERTENSION (Stage-1)',
    2: 'HYPERTENSION (Stage-2)',
    3: 'HYPERTENSIVE CRISIS'
}

# Medical-grade color mapping for results
color_map = {
    0: '#10B981',   # Medical green for normal
    1: '#F59E0B',   # Medical amber for stage 1
    2: '#F97316',   # Medical orange for stage 2
    3: '#EF4444'    # Medical red for crisis
}

# Detailed medical recommendations
recommendations = {
    0: {
        'title': 'Normal Blood Pressure',
        'description': 'Your cardiovascular risk assessment indicates normal blood pressure levels.',
        'actions': [
            'Maintain current healthy lifestyle',
            'Regular physical activity (150 minutes/week)',
            'Continue balanced, low-sodium diet',
            'Annual blood pressure monitoring',
            'Regular health check-ups'
        ],
        'priority': 'Low Risk'
    },
    1: {
        'title': 'Stage 1 Hypertension',
        'description': 'Mild elevation detected requiring lifestyle modifications and medical consultation.',
        'actions': [
            'Schedule appointment with healthcare provider',
            'Implement DASH diet plan',
            'Increase physical activity gradually',
            'Monitor blood pressure bi-weekly',
            'Reduce sodium intake (<2300mg/day)',
            'Consider stress management techniques'
        ],
        'priority': 'Moderate Risk'
    },
    2: {
        'title': 'Stage 2 Hypertension',
        'description': 'Significant hypertension requiring immediate medical intervention and treatment.',
        'actions': [
            'URGENT: Consult physician within 1-2 days',
            'Likely medication therapy required',
            'Comprehensive cardiovascular assessment',
            'Daily blood pressure monitoring',
            'Strict dietary sodium restriction',
            'Lifestyle modification counseling'
        ],
        'priority': 'High Risk'
    },
    3: {
        'title': 'Hypertensive Crisis',
        'description': 'CRITICAL: Dangerously elevated blood pressure requiring emergency medical care.',
        'actions': [
            'EMERGENCY: Seek immediate medical attention',
            'Call 911 if experiencing symptoms',
            'Do not delay treatment',
            'Monitor for stroke/heart attack signs',
            'Prepare current medication list',
            'Avoid physical exertion'
        ],
        'priority': 'EMERGENCY'
    }
}

# Encoding maps (must match data_preparation.py exactly)
encoding_maps = {
    'Gender': {'Male': 0, 'Female': 1},
    'Age': {'18-34': 1, '35-50': 2, '51-64': 3, '65+': 4},
    'History': {'No': 0, 'Yes': 1},
    'Patient': {'No': 0, 'Yes': 1},
    'TakeMedication': {'No': 0, 'Yes': 1},
    'Severity': {'Mild': 0, 'Moderate': 1, 'Sever': 2},
    'BreathShortness': {'No': 0, 'Yes': 1},
    'VisualChanges': {'No': 0, 'Yes': 1},
    'NoseBleeding': {'No': 0, 'Yes': 1},
    'Whendiagnoused': {'<1 Year': 1, '1 - 5 Years': 2, '>5 Years': 3},
    'Systolic': {'100 - 110': 0, '111 - 120': 1, '121 - 130': 2, '130+': 3},
    'Diastolic': {'70 - 80': 0, '81 - 90': 1, '91 - 100': 2, '100+': 3},
    'ControlledDiet': {'No': 0, 'Yes': 1}
}

# Define which features are ordinal (scaled by MinMaxScaler)
ordinal_feature_names = ['Age', 'Severity', 'Whendiagnoused', 'Systolic', 'Diastolic']
# Feature order must match training data
feature_order = ['Gender', 'Age', 'History', 'Patient', 'TakeMedication',
                 'Severity', 'BreathShortness', 'VisualChanges', 'NoseBleeding',
                 'Whendiagnoused', 'Systolic', 'Diastolic', 'ControlledDiet']


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.method == 'POST':
            # Collect user inputs from the form with validation
            required_fields = ['Gender', 'Age', 'History', 'Patient', 'TakeMedication',
                               'Severity', 'BreathShortness', 'VisualChanges', 'NoseBleeding',
                               'Whendiagnoused', 'Systolic', 'Diastolic', 'ControlledDiet']

            form_data = {}
            for field in required_fields:
                value = request.form.get(field)
                if not value:
                    flash(f'Please fill in the {field} field.', 'error')
                    return render_template('index.html')
                form_data[field] = value

            # Encode form values using encoding maps
            encoded = []
            for field in feature_order:
                raw_value = form_data[field]
                if field in encoding_maps and raw_value in encoding_maps[field]:
                    encoded.append(encoding_maps[field][raw_value])
                else:
                    flash(f'Invalid value for {field}: {raw_value}', 'error')
                    return render_template('index.html')

            # Convert to numpy array
            features = np.array(encoded).reshape(1, -1)

            # Apply scaler to ordinal features only
            if scaler is not None:
                ordinal_indices = [feature_order.index(f) for f in ordinal_feature_names]
                features_copy = features.copy().astype(float)
                ordinal_values = features_copy[0, ordinal_indices].reshape(1, -1)
                features_copy[0, ordinal_indices] = scaler.transform(ordinal_values).flatten()
                features = features_copy

            # Make prediction
            if model is not None:
                prediction = int(model.predict(features)[0])
                probabilities = model.predict_proba(features)[0]
                confidence = round(float(probabilities[prediction]) * 100, 1)
            else:
                prediction = 0
                confidence = 0.0

            # Get results
            result = {
                'stage': stage_map[prediction],
                'stage_num': prediction,
                'color': color_map[prediction],
                'confidence': confidence,
                'recommendation': recommendations[prediction],
                'form_data': form_data
            }

            return render_template('index.html', result=result, form_data=form_data)

    except Exception as e:
        flash(f'An error occurred during prediction: {str(e)}', 'error')
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, port=5000)
