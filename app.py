from flask import Flask, render_template, request, jsonify
import tempfile
from pathlib import Path
import joblib
import numpy as np

from cbc_extractor import extract_cbc_from_pdf_to_row


# =====================================================
# UI CBC FIELD  →  MODEL FEATURE NAME (CRITICAL FIX)
# =====================================================
UI_TO_MODEL_FEATURE_MAP = {
    "Hemoglobin": "hemoglobin",
    "Hematocrit": "hematocrit",
    "MCV": "mcv",
    "MCH": "mch",
    "RDW-CV": "rdw",

    "Total Leucocyte Count": "wbc",

    "Neutrophils (%)": "neutrophils_pct",
    "Lymphocytes (%)": "lymphocytes_pct",
    "Monocytes (%)": "monocytes_pct",
    "Eosinophils (%)": "eosinophils_pct",
    "Basophils (%)": "basophils_pct",

    "Platelet Count": "platelets",
    "RBC Count": "rbc",
    "MPV": "mpv"
}



# =========================
# FLASK APP SETUP
# =========================
app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)

# =========================
# LOAD ML MODELS
# =========================
print("🔄 Loading model and label encoder...")

model = joblib.load("models/lightgbm_model.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

print("\n==============================")
print("✅ MODEL LOADED SUCCESSFULLY")
print("==============================")

print("\n📌 MODEL FEATURES (ORDER MATTERS):")
for i, f in enumerate(model.feature_names_in_):
    print(f"{i+1:02d}. {f}")

print("\n📌 MODEL CLASSES:")
for i, c in enumerate(label_encoder.classes_):
    print(f"{i}: {c}")

print("==============================\n")


# =========================
# ROUTES (PAGES)
# =========================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload")
def upload():
    return render_template("upload.html")


@app.route("/analysis")
def analysis():
    return render_template("analysis.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/diagnosis")
def diagnosis():
    return render_template("diagnosis.html")


@app.route("/insights")
def insights():
    return render_template("insights.html")


# =========================
# API ENDPOINTS
# =========================

# 🔹 FILE UPLOAD + CBC EXTRACTION
@app.route("/extract", methods=["POST"])
def extract():
    print("➡️ /extract called")

    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
        file.save(tmp.name)
        print("📄 Temp PDF saved:", tmp.name)

        row = extract_cbc_from_pdf_to_row(Path(tmp.name))
        print("✅ CBC extraction completed")

    extracted = {
        k: v for k, v in row.items()
        if k not in ["SourceFile", "FileHash", "PagesUsed"]
    }

    print("\n📊 EXTRACTED VALUES (UI):")
    for k, v in extracted.items():
        print(f"  {k:25s} -> {v}")
    print("--------------------------------\n")

    return jsonify(extracted)


# ==================================================
# UNIT NORMALIZATION → MATCH TRAINING UNITS
# ==================================================
def normalize_to_training_units(ui_key: str, value: float) -> float:
    """
    Convert extracted CBC values to the SAME units used during training
    """

    if value is None:
        return 0.0

    try:
        v = float(value)
    except:
        return 0.0

    key = ui_key.lower()

    # ---- WBC & DIFFERENTIAL ABS (×10³ / µL) ----
    if key in ["total leucocyte count"]:
        # reports may give 9000 → model expects 9.0
        if v > 1000:
            return v / 1000.0
        return v

    # ---- PLATELETS (×10³ / µL) ----
    if key in ["platelet count"]:
        # 250000 → 250
        if v > 1000:
            return v / 1000.0
        return v

    # ---- RBC (×10⁶ / µL) ----
    if key in ["rbc count"]:
        # 4500000 → 4.5
        if v > 100:
            return v / 1_000_000.0
        return v

    # ---- HEMOGLOBIN (g/dL) ----
    if key == "hemoglobin":
        # sometimes reported as g/L
        if v > 50:
            return v / 10.0
        return v

    # ---- DEFAULT (already correct units) ----
    return v



# 🔹 ANALYZE CBC → ML
# 🔹 ANALYZE CBC → ML
@app.route("/analyze", methods=["POST"])
def analyze():
    print("➡️ /analyze called")

    data = request.json
    if not data:
        return jsonify({"error": "No data received"}), 400

    # -------------------------------------------------
    # RAW INPUT LOG
    # -------------------------------------------------
    print("\n📥 RAW DATA FROM UI:")
    for k, v in data.items():
        print(f"  {k:25s} -> {v}")

    # -------------------------------------------------
    # DERIVE ABSOLUTE COUNTS FROM PERCENTAGES
    # -------------------------------------------------
    try:
        wbc_raw = float(data.get("Total Leucocyte Count", 0))
    except:
        wbc_raw = 0.0

    # normalize WBC to training units (×10³)
    wbc = wbc_raw / 1000.0 if wbc_raw > 1000 else wbc_raw

    def abs_from_pct(pct):
        try:
            return (float(pct) / 100.0) * wbc
        except:
            return 0.0

    derived_features = {
        "neutrophils_abs": abs_from_pct(data.get("Neutrophils (%)")),
        "lymphocytes_abs": abs_from_pct(data.get("Lymphocytes (%)")),
        "monocytes_abs": abs_from_pct(data.get("Monocytes (%)")),
        "eosinophils_abs": abs_from_pct(data.get("Eosinophils (%)")),
        "basophils_abs": abs_from_pct(data.get("Basophils (%)")),
    }

    # -------------------------------------------------
    # BUILD FEATURE VECTOR (STRICT = 19 FEATURES)
    # -------------------------------------------------
    feature_vector = []

    print("\n🧠 MODEL FEATURE VECTOR BUILD:")
    for model_feature in model.feature_names_in_:

        value = 0.0  # default → guarantees ONE append

        # 1️⃣ Derived absolute features
        if model_feature in derived_features:
            value = derived_features[model_feature]
            print(f"  {model_feature:20s} <- DERIVED = {value:.4f}")

        else:
            # 2️⃣ Map UI → model feature
            ui_key = None
            for ui_name, mf_name in UI_TO_MODEL_FEATURE_MAP.items():
                if mf_name == model_feature:
                    ui_key = ui_name
                    break

            if ui_key:
                raw_val = data.get(ui_key, "")
                try:
                    raw_val = float(raw_val)
                except:
                    raw_val = 0.0

                # ---------- UNIT NORMALIZATION ----------
                if model_feature == "wbc" and raw_val > 1000:
                    value = raw_val / 1000.0
                elif model_feature == "platelets" and raw_val > 1000:
                    value = raw_val / 1000.0
                elif model_feature == "rbc" and raw_val > 100:
                    value = raw_val / 1_000_000.0
                elif model_feature == "hemoglobin" and raw_val > 50:
                    value = raw_val / 10.0
                else:
                    value = raw_val

                print(
                    f"  {model_feature:20s} <- {ui_key:25s} "
                    f"RAW={raw_val} → NORM={value}"
                )
            else:
                print(f"  {model_feature:20s} -> MISSING (0.0)")

        feature_vector.append(float(value))  # EXACTLY ONE APPEND

    # -------------------------------------------------
    # FINAL NUMERIC VECTOR
    # -------------------------------------------------
    X = np.array(feature_vector).reshape(1, -1)

    print("\n🧪 FINAL NUMERIC VECTOR SHAPE:", X.shape)
    print(X)

    if X.shape[1] != len(model.feature_names_in_):
        raise RuntimeError(
            f"Feature mismatch: expected {len(model.feature_names_in_)} "
            f"but got {X.shape[1]}"
        )

    # -------------------------------------------------
    # SAFETY CHECK
    # -------------------------------------------------
    critical_features = ["wbc", "hemoglobin", "rbc", "platelets"]
    for i, fname in enumerate(model.feature_names_in_):
        if fname in critical_features and X[0][i] == 0.0:
            return jsonify({
                "screening_result": "No Conclusive Screening",
                "confidence": 0.0,
                "advice": f"Missing critical parameter: {fname}"
            })

    # -------------------------------------------------
    # MODEL INFERENCE
    # -------------------------------------------------
    probs = model.predict_proba(X)[0]
    pred_idx = int(np.argmax(probs))

    disease = label_encoder.inverse_transform([pred_idx])[0]
    confidence = round(float(probs[pred_idx]) * 100, 2)

    # -------------------------------------------------
    # CONFIDENCE GATING
    # -------------------------------------------------
    CONFIDENCE_THRESHOLD = 55.0

    if confidence < CONFIDENCE_THRESHOLD:
        print("⚠️ Low confidence – marking as inconclusive")
        return jsonify({
            "screening_result": "No Conclusive Screening",
            "confidence": confidence,
            "advice": "Consult a qualified physician for further evaluation",
            "top_patterns": [
                {
                    "condition": d,
                    "probability": round(p * 100, 2)
                }
                for d, p in sorted(
                    zip(label_encoder.classes_, probs),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
            ]
        })

    # -------------------------------------------------
    # SUCCESS RESPONSE
    # -------------------------------------------------
    print("\n🏥 FINAL SCREENING RESULT:")
    print("  Result     :", disease)
    print("  Confidence :", confidence, "%")
    print("==============================\n")

    top3 = sorted(
        zip(label_encoder.classes_, probs),
        key=lambda x: x[1],
        reverse=True
    )[:3]

    return jsonify({
        "screening_result": disease,
        "confidence": confidence,
        "advice": "Screening result based on CBC pattern analysis",
        "top_patterns": [
            {"condition": d, "probability": round(p * 100, 2)}
            for d, p in top3
        ]
    })


# =========================
# START SERVER
# =========================
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        threaded=True
    )
