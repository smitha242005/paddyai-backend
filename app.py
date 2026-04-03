# ══════════════════════════════════
# PaddyAI Flask Backend
# ══════════════════════════════════

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import pickle
import json
import base64
from PIL import Image
import io
import os
import gdown

app = Flask(__name__)
CORS(app)

# ── Auto Download Disease Model from Google Drive ──
if not os.path.exists('disease_model.h5'):
    print("Downloading disease model from Google Drive...")
    gdown.download(
        id='1XCBxp3hF69sTMS0pr2efn5PyfBkutZpe',
        output='disease_model.h5',
        quiet=False
    )
    print("✅ disease_model.h5 downloaded!")

# ── Load Disease Model ──
print("Loading disease model...")
disease_model = tf.keras.models.load_model('disease_model.h5')
with open('class_indices.json') as f:
    class_indices = json.load(f)
classes = {v: k for k, v in class_indices.items()}
print("✅ Disease model loaded!")
print("Classes:", classes)

# ── Load Yield Model ──
print("Loading yield model...")
with open('yield_model.pkl', 'rb') as f:
    yield_model = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
with open('yield_model_info.json') as f:
    yield_info = json.load(f)
print("✅ Yield model loaded!")

# ── Helper: decode image ──
def decode_image(base64_str, img_size=128):
    img_data = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_data)).convert('RGB')
    img = img.resize((img_size, img_size))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# ── Helper: yield category ──
def get_yield_category(yield_val):
    if yield_val >= 50000:
        return 'High'
    elif yield_val >= 30000:
        return 'Medium'
    else:
        return 'Low'

# ── Route: Health check ──
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'PaddyAI Backend Running ✅',
        'disease_model': 'Loaded',
        'yield_model': 'Loaded',
        'accuracy': {
            'disease': '81.25%',
            'yield': '96.14%'
        }
    })

# ── Route: Disease + Yield Prediction ──
@app.route('/predict/disease', methods=['POST'])
def predict_disease():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        # ── Disease Detection ──
        img = decode_image(data['image'])
        predictions = disease_model.predict(img)[0]

        disease_classes = [
            {
                'name': classes[i],
                'confidence': round(float(predictions[i]) * 100, 2),
                'color': ['#f44336','#ff9800','#9c27b0'][i]
            }
            for i in range(len(classes))
        ]
        disease_classes.sort(key=lambda x: x['confidence'], reverse=True)

        primary_disease = disease_classes[0]['name']
        primary_confidence = disease_classes[0]['confidence']

        if primary_confidence < 60:
            primary_disease = 'Healthy'

        # ── Yield Prediction ──
        try:
            area_encoded = label_encoder.transform(['India'])[0]
        except:
            area_encoded = 0

        yield_features = np.array([[
            area_encoded,
            2024,
            1200.0,
            121.0,
            28.0
        ]])

        predicted_yield = yield_model.predict(yield_features)[0]
        yield_category = get_yield_category(predicted_yield)
        yield_tonnes = round(predicted_yield / 10000, 2)

        # ── Build Response ──
        response = {
            'cropVariety': 'Paddy — Oryza sativa',
            'overallHealthScore': int(primary_confidence) if primary_disease != 'Healthy' else int(100 - primary_confidence + 60),
            'overallVerdict': 'Good' if primary_disease == 'Healthy' else 'Poor',
            'verdictSummary': f'Primary detection: {primary_disease} with {primary_confidence}% confidence.',
            'diseaseDetection': {
                'primaryDisease': primary_disease,
                'confidence': round(primary_confidence, 2),
                'classes': disease_classes
            },
            'yieldPrediction': {
                'predictedYield': f'{yield_tonnes} t/ha',
                'yieldConfidence': round(float(yield_info['r2_score']) * 100, 2),
                'yieldCategory': yield_category,
                'soilType': 'Clayey loam',
                'waterRequirement': '5–6 L/day',
                'season': 'Kharif (Jun–Nov)',
                'harvestMonth': 'October–November',
                'fertilizer': 'NPK 120:60:60 kg/ha',
                'growthStage': 'Tillering'
            },
            'recommendations': [
                {
                    'type': 'success' if primary_disease == 'Healthy' else 'danger',
                    'icon': '✅' if primary_disease == 'Healthy' else '⚠️',
                    'text': f'Disease detected: {primary_disease}. Immediate treatment recommended.' if primary_disease != 'Healthy' else 'Crop is healthy. Continue current care.'
                },
                {
                    'type': 'warning',
                    'icon': '💧',
                    'text': 'Maintain 3–5 cm standing water during tillering stage.'
                },
                {
                    'type': 'success',
                    'icon': '🧪',
                    'text': f'Expected yield: {yield_tonnes} t/ha — {yield_category} performance.'
                },
                {
                    'type': 'warning',
                    'icon': '🌾',
                    'text': 'Apply NPK 120:60:60 kg/ha for optimal growth.'
                }
            ]
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
