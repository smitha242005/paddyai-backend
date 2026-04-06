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

# ─────────────────────────────────────────────
# CORS
# ─────────────────────────────────────────────
CORS(app, origins=[
    "https://smitha242005.github.io",
    "http://localhost:5500",
    "http://127.0.0.1:5500"
])

# ─────────────────────────────────────────────
# Disease Model Download + Load
# ─────────────────────────────────────────────
DISEASE_MODEL_PATH = "disease_model_fixed.keras"
GDRIVE_FILE_ID = "1eLQwu_VN1W0TWHtPYGij6fT7pjW5M1AI"

disease_model = None

if not os.path.exists(DISEASE_MODEL_PATH):
    print("⬇️ Downloading disease model from Google Drive...")

    try:
        gdown.download(
            id=GDRIVE_FILE_ID,
            output=DISEASE_MODEL_PATH,
            quiet=False,
            fuzzy=True
        )

        if os.path.exists(DISEASE_MODEL_PATH):
            size_mb = os.path.getsize(DISEASE_MODEL_PATH) / (1024 * 1024)
            print(f"📁 Downloaded file size: {size_mb:.2f} MB")

            if size_mb < 5:
                print("❌ Download failed: file too small")
                os.remove(DISEASE_MODEL_PATH)
            else:
                print("✅ Disease model downloaded successfully!")

    except Exception as e:
        print(f"❌ Error downloading disease model: {e}")

if os.path.exists(DISEASE_MODEL_PATH):
    try:
        print(f"TensorFlow version: {tf.__version__}")

        disease_model = tf.keras.models.load_model(
            DISEASE_MODEL_PATH,
            compile=False
        )

        print("✅ Disease model loaded successfully!")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Failed to load disease model: {e}")
else:
    print("❌ Disease model file not found")

# ─────────────────────────────────────────────
# Load Yield Prediction Files
# ─────────────────────────────────────────────
print("⬇️ Loading yield model files...")

with open("yield_model.pkl", "rb") as f:
    yield_model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open("yield_model_info.json", "r") as f:
    yield_info = json.load(f)

with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

idx_to_class = {v: k for k, v in class_indices.items()}

print("✅ Yield model loaded!")

# ─────────────────────────────────────────────
# Disease Information Database
# ─────────────────────────────────────────────
DISEASE_INFO = {
    "Bacterial leaf blight": {
        "medicine": "Streptomycin sulfate + Tetracycline (0.025%)",
        "pesticide": "Copper oxychloride 50 WP @ 3g/L",
        "recovery": "Drain field, apply potash fertilizer, avoid excess nitrogen",
        "severity": "High",
        "color": "#f44336"
    },
    "Brown spot": {
        "medicine": "Mancozeb 75 WP @ 2.5g/L or Iprodione",
        "pesticide": "Propiconazole 25 EC @ 1ml/L",
        "recovery": "Improve soil nutrition, apply potassium silicate",
        "severity": "Medium",
        "color": "#ff9800"
    },
    "Leaf smut": {
        "medicine": "Carbendazim 50 WP @ 1g/L",
        "pesticide": "Tricyclazole 75 WP @ 0.6g/L",
        "recovery": "Remove infected leaves, improve drainage",
        "severity": "Low",
        "color": "#9c27b0"
    }
}

# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────
def get_yield_category(predicted_yield):
    if predicted_yield >= 50000:
        return "High"
    elif predicted_yield >= 30000:
        return "Medium"
    return "Low"


def preprocess_image(base64_image):
    if "," in base64_image:
        base64_image = base64_image.split(",")[1]

    image_bytes = base64.b64decode(base64_image)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((128, 128))

    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    return image_array

# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "✅ PaddyAI Backend Running",
        "disease_model_loaded": disease_model is not None,
        "yield_model_loaded": True,
        "accuracy": {
            "disease_model": "81.25%",
            "yield_model": f"{round(yield_info['r2_score'] * 100, 2)}%"
        },
        "endpoints": [
            "/health",
            "/predict/disease",
            "/predict/yield"
        ]
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "disease_model_loaded": disease_model is not None,
        "yield_model_loaded": True
    })


@app.route("/predict/disease", methods=["POST"])
def predict_disease():
    try:
        data = request.get_json()

        if not data or "image" not in data:
            return jsonify({"error": "No image provided"}), 400

        image_array = preprocess_image(data["image"])

        primary_disease = "Unknown"
        confidence = 0.0
        all_predictions = []

        if disease_model is not None:
            predictions = disease_model.predict(image_array, verbose=0)[0]

            top_index = int(np.argmax(predictions))
            primary_disease = idx_to_class[top_index]
            confidence = round(float(predictions[top_index]) * 100, 2)

            for i, pred in enumerate(predictions):
                disease_name = idx_to_class[i]

                all_predictions.append({
                    "name": disease_name,
                    "confidence": round(float(pred) * 100, 2),
                    "color": DISEASE_INFO[disease_name]["color"]
                })

        else:
            return jsonify({
                "error": "Disease model not loaded"
            }), 500

        treatment = DISEASE_INFO.get(primary_disease, {})

        return jsonify({
            "disease": primary_disease,
            "confidence": confidence,
            "medicine": treatment.get("medicine", ""),
            "pesticide": treatment.get("pesticide", ""),
            "recovery": treatment.get("recovery", ""),
            "severity": treatment.get("severity", ""),
            "predictions": all_predictions
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": str(e)
        }), 500


@app.route("/predict/yield", methods=["POST"])
def predict_yield():
    try:
        data = request.get_json()

        country = data.get("country", "India")
        year = int(data.get("year", 2024))
        rainfall = float(data.get("rainfall", 1200))
        pesticides = float(data.get("pesticides", 121))
        avg_temp = float(data.get("avg_temp", 28))

        try:
            area_encoded = label_encoder.transform([country])[0]
        except Exception:
            area_encoded = label_encoder.transform(["India"])[0]

        features = np.array([
            [area_encoded, year, rainfall, pesticides, avg_temp]
        ])

        predicted_yield = float(yield_model.predict(features)[0])
        yield_tonnes = round(predicted_yield / 10000, 2)
        category = get_yield_category(predicted_yield)

        return jsonify({
            "country": country,
            "predictedYield": f"{yield_tonnes} t/ha",
            "yieldCategory": category,
            "confidence": round(yield_info["r2_score"] * 100, 2)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
