from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import json
import base64
from PIL import Image
import io
import os
import gdown
import tensorflow as tf

app = Flask(__name__)

# Allow frontend access from GitHub Pages and local testing
CORS(app, resources={r"/*": {"origins": "*"}})

# ============================================================
# Disease Model Weights Download
# ============================================================

WEIGHTS_FILE = "disease_weights.weights.h5"
GDRIVE_FILE_ID = "1Gy9rg6uAYgTmT6CCK2qkiunE3BxZuaOp"

if not os.path.exists(WEIGHTS_FILE):
    print("Downloading disease model weights...")

    try:
        gdown.download(
            id=GDRIVE_FILE_ID,
            output=WEIGHTS_FILE,
            quiet=False,
            fuzzy=True
        )

        if not os.path.exists(WEIGHTS_FILE):
            raise Exception("Weights file not downloaded")

        size_mb = os.path.getsize(WEIGHTS_FILE) / (1024 * 1024)
        print(f"Weights downloaded: {size_mb:.2f} MB")

        if size_mb < 1:
            raise Exception("Downloaded file too small / invalid")

    except Exception as e:
        print("Failed to download disease weights:", e)

# ============================================================
# Build Disease CNN
# ============================================================

disease_model = None

try:
    disease_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(128, 128, 3)),

        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(3, activation="softmax")
    ])

    disease_model.build((None, 128, 128, 3))

    if os.path.exists(WEIGHTS_FILE):
        disease_model.load_weights(WEIGHTS_FILE)
        print("Disease model loaded successfully")
    else:
        print("Disease model weights file missing")
        disease_model = None

except Exception as e:
    print("Failed to load disease model:", e)
    disease_model = None

# ============================================================
# Load Yield Model
# ============================================================

with open("yield_model.pkl", "rb") as f:
    yield_model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open("yield_model_info.json", "r") as f:
    yield_info = json.load(f)

with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

idx_to_class = {v: k for k, v in class_indices.items()}

# ============================================================
# Disease Details
# ============================================================

DISEASE_INFO = {
    "Bacterial leaf blight": {
        "medicine": "Streptomycin sulfate + Tetracycline (0.025%)",
        "pesticide": "Copper oxychloride 50 WP @ 3g/L",
        "recovery": "Drain the field, apply potash fertilizer, avoid excess nitrogen",
        "severity": "High",
        "color": "#ef4444"
    },
    "Brown spot": {
        "medicine": "Mancozeb 75 WP @ 2.5g/L",
        "pesticide": "Propiconazole 25 EC @ 1ml/L",
        "recovery": "Improve soil nutrition and apply potassium silicate",
        "severity": "Medium",
        "color": "#f59e0b"
    },
    "Leaf smut": {
        "medicine": "Carbendazim 50 WP @ 1g/L",
        "pesticide": "Tricyclazole 75 WP @ 0.6g/L",
        "recovery": "Remove infected leaves and improve drainage",
        "severity": "Low",
        "color": "#8b5cf6"
    }
}

# ============================================================
# Helper Functions
# ============================================================

def preprocess_image(image_data):
    if "," in image_data:
        image_data = image_data.split(",")[1]

    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((128, 128))

    arr = np.array(image, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    return arr


def get_yield_category(y):
    if y >= 50000:
        return "High"
    elif y >= 30000:
        return "Medium"
    return "Low"

# ============================================================
# Routes
# ============================================================

@app.route("/")
def root():
    return jsonify({
        "status": "running",
        "disease_model_loaded": disease_model is not None,
        "yield_model_loaded": True
    })


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "disease_model_loaded": disease_model is not None,
        "yield_model_loaded": True
    })


@app.route("/predict/disease", methods=["POST", "OPTIONS"])
def predict_disease():
    if request.method == "OPTIONS":
        return jsonify({"ok": True})

    try:
        data = request.get_json()

        if not data or "image" not in data:
            return jsonify({"error": "No image provided"}), 400

        if disease_model is None:
            return jsonify({"error": "Disease model not loaded"}), 500

        image = preprocess_image(data["image"])

        preds = disease_model.predict(image, verbose=0)[0]

        top_index = int(np.argmax(preds))
        disease = idx_to_class[top_index]
        confidence = round(float(preds[top_index]) * 100, 2)

        disease_info = DISEASE_INFO.get(disease, {})

        prediction_list = []

        for i, p in enumerate(preds):
            cls = idx_to_class[i]
            prediction_list.append({
                "name": cls,
                "confidence": round(float(p) * 100, 2),
                "color": DISEASE_INFO.get(cls, {}).get("color", "#6b7280")
            })

        return jsonify({
            "disease": disease,
            "confidence": confidence,
            "medicine": disease_info.get("medicine", ""),
            "pesticide": disease_info.get("pesticide", ""),
            "recovery": disease_info.get("recovery", ""),
            "severity": disease_info.get("severity", "Low"),
            "predictions": prediction_list
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict/yield", methods=["POST", "OPTIONS"])
def predict_yield():
    if request.method == "OPTIONS":
        return jsonify({"ok": True})

    try:
        data = request.get_json()

        country = data.get("country", "India")
        year = int(data.get("year", 2024))
        rainfall = float(data.get("rainfall", 1200))
        pesticides = float(data.get("pesticides", 121))
        avg_temp = float(data.get("avg_temp", 28))

        try:
            encoded_country = label_encoder.transform([country])[0]
        except:
            encoded_country = label_encoder.transform(["India"])[0]

        features = np.array([[
            encoded_country,
            year,
            rainfall,
            pesticides,
            avg_temp
        ]])

        pred = float(yield_model.predict(features)[0])

        return jsonify({
            "predictedYield": round(pred / 10000, 2),
            "yieldCategory": get_yield_category(pred),
            "confidence": round(float(yield_info["r2_score"]) * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
