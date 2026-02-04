from flask import Flask, render_template, request, jsonify
import tempfile
from pathlib import Path
import joblib
import numpy as np

from cbc_extractor import extract_cbc_from_pdf_to_row

# =====================================================
# UI CBC FIELD → MODEL FEATURE NAME
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

# =====================================================
# FLASK APP
# =====================================================
app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)

# =====================================================
# LOAD ML MODEL
# =====================================================
print("🔄 Loading ML model...")
BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "models" / "lightgbm_model.pkl")
label_encoder = joblib.load(BASE_DIR / "models" / "label_encoder.pkl")

print("✅ Model loaded")

# =====================================================
# DISEASE → RECOMMENDED TESTS
# =====================================================
DISEASE_TESTS = {
    "Allergy": ["IgE Test", "Skin Prick Test", "Allergen Blood Test"],
    "Anemia": ["CBC", "Iron Studies", "Vitamin B12 & Folate"],
    "Bacterial_Infection": ["Blood Culture", "CRP", "Procalcitonin"],
    "Chronic_Inflammation": ["ESR", "CRP", "Autoimmune Panel"],
    "Dengue": ["NS1 Antigen", "Dengue IgM / IgG", "Platelet Monitoring"],
    "Leukemia": ["Peripheral Smear", "Bone Marrow Biopsy", "Flow Cytometry"],
    "Normal": [],
    "Parasitic_Infection": ["Stool Examination", "Blood Smear", "Eosinophil Count"],
    "Polycythemia": ["Hemoglobin", "JAK2 Mutation", "EPO Level"],
    "Sepsis": ["Blood Culture", "Serum Lactate", "CRP"],
    "Thrombocytopenia": ["Platelet Count", "Peripheral Smear", "Bone Marrow Exam"],
    "Thrombocytosis": ["Platelet Count", "Iron Studies", "JAK2 Mutation"],
    "Viral_Infection": ["PCR Test", "Viral Serology", "CRP"]
}

# =====================================================
# TEST → WHY REQUIRED
# =====================================================
TEST_INFO = {
    "IgE Test": "Measures Immunoglobulin E levels to identify allergies.",
    "Skin Prick Test": "Helps detect specific allergens causing reactions.",
    "Allergen Blood Test": "Identifies the presence of allergen-specific antibodies.",
    "CBC": "Complete Blood Count checks red cells, white cells, and platelets to detect anemia or infection.",
    "Iron Studies": "Evaluates iron levels to determine iron-deficiency anemia.",
    "Vitamin B12 & Folate": "Checks vitamin levels essential for red blood cell production.",
    "Blood Culture": "Identifies bacteria in the bloodstream for infection diagnosis.",
    "CRP": "C-reactive protein test detects inflammation in the body.",
    "Procalcitonin": "Used to detect bacterial infections and sepsis.",
    "ESR": "Erythrocyte Sedimentation Rate detects chronic inflammation.",
    "Autoimmune Panel": "Tests for autoimmune disorders causing chronic inflammation.",
    "NS1 Antigen": "Detects dengue virus early in infection.",
    "Dengue IgM / IgG": "Detects recent or past dengue infection.",
    "Platelet Monitoring": "Monitors platelet levels in diseases like dengue.",
    "Peripheral Smear": "Examines blood cells under a microscope for leukemia or platelet disorders.",
    "Bone Marrow Biopsy": "Helps confirm leukemia or other marrow disorders.",
    "Flow Cytometry": "Analyzes blood cell types for leukemia detection.",
    "Stool Examination": "Detects intestinal parasites.",
    "Eosinophil Count": "Elevated in parasitic infections and allergies.",
    "JAK2 Mutation": "Detects genetic mutations causing polycythemia or thrombocytosis.",
    "EPO Level": "Assesses erythropoietin hormone affecting red blood cell production.",
    "Serum Lactate": "Elevated in sepsis and tissue hypoxia.",
    "PCR Test": "Detects viral DNA/RNA for viral infections.",
    "Viral Serology": "Checks antibodies for viral infections."
}

# =====================================================
# NORMAL CBC RANGES (for frontend bars)
# =====================================================
NORMAL_RANGES = {
    "Hemoglobin": (12, 16),  # g/dL
    "Hematocrit": (36, 48),  # %
    "MCV": (80, 100),        # fL
    "MCH": (27, 33),         # pg
    "RDW-CV": (11.5, 14.5),  # %
    "Total Leucocyte Count": (4, 10),  # 10^3/µL
    "Neutrophils (%)": (40, 70), 
    "Lymphocytes (%)": (20, 45),
    "Monocytes (%)": (2, 10),
    "Eosinophils (%)": (1, 6),
    "Basophils (%)": (0, 1),
    "Platelet Count": (150, 450),  # 10^3/µL
    "RBC Count": (4.2, 5.9),  # 10^6/µL
    "MPV": (7, 11)  # fL
}

# =====================================================
# ROUTES
# =====================================================
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

@app.route("/insights")
def insights():
    return render_template("insights.html")

@app.route("/diagnosis")
def diagnosis():
    disease = request.args.get("disease", "Normal")
    tests = DISEASE_TESTS.get(disease, [])
    test_info = {test: TEST_INFO.get(test, "No additional info available") for test in tests}
    return render_template("diagnosis.html", disease=disease, tests=tests, test_info=test_info)

# =====================================================
# PDF EXTRACTION
# =====================================================
@app.route("/extract", methods=["POST"])
def extract():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        row = extract_cbc_from_pdf_to_row(Path(tmp_path))
    finally:
        if tmp_path and Path(tmp_path).exists():
            Path(tmp_path).unlink()

    extracted = {k: v for k, v in row.items() if k not in ["SourceFile", "FileHash", "PagesUsed"]}
    return jsonify(extracted)

@app.route("/system", methods=["GET", "POST"])
def system():
    if request.method == "POST":
        file = request.files.get("dataset")
        if file:
            # process CSV
            info = {
                "rows": 123  # example
            }
            return render_template("system.html", info=info)

    # GET request (page load)
    return render_template("system.html")



# =====================================================
# ANALYZE CBC → ML
# =====================================================
@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    if not data:
        return jsonify({"error": "No data"}), 400

    # Convert WBC to proper scale
    try:
        wbc_raw = float(data.get("Total Leucocyte Count", 0))
    except:
        wbc_raw = 0.0
    wbc = wbc_raw / 1000 if wbc_raw > 1000 else wbc_raw

    def abs_from_pct(pct):
        try:
            return (float(pct)/100) * wbc
        except:
            return 0.0

    derived = {
        "neutrophils_abs": abs_from_pct(data.get("Neutrophils (%)")),
        "lymphocytes_abs": abs_from_pct(data.get("Lymphocytes (%)")),
        "monocytes_abs": abs_from_pct(data.get("Monocytes (%)")),
        "eosinophils_abs": abs_from_pct(data.get("Eosinophils (%)")),
        "basophils_abs": abs_from_pct(data.get("Basophils (%)")),
    }

    # Create feature vector
    feature_vector = []
    for mf in model.feature_name_:
        val = 0.0
        if mf in derived:
            val = derived[mf]
        else:
            for ui, m in UI_TO_MODEL_FEATURE_MAP.items():
                if m == mf:
                    try:
                        raw = float(data.get(ui, 0))
                    except:
                        raw = 0.0
                    # Normalize large numbers
                    if mf == "wbc" and raw > 1000:
                        val = raw / 1000
                    elif mf == "platelets" and raw > 1000:
                        val = raw / 1000
                    elif mf == "rbc" and raw > 100:
                        val = raw / 1_000_000
                    elif mf == "hemoglobin" and raw > 50:
                        val = raw / 10
                    else:
                        val = raw
                    break
        feature_vector.append(float(val))

    X = np.array(feature_vector).reshape(1, -1)
    probs = model.predict_proba(X)[0]
    idx = int(np.argmax(probs))
    disease = label_encoder.inverse_transform([idx])[0]
    confidence = round(float(probs[idx])*100, 2)

    # Determine critical parameters for frontend
    important_parameters = {}
    for ui_field, value in data.items():
        if ui_field in NORMAL_RANGES:
            low, high = NORMAL_RANGES[ui_field]
            try:
                v = float(value)
            except:
                v = 0.0
            status = "normal"
            if v < low:
                status = "low"
            elif v > high:
                status = "high"
            important_parameters[ui_field] = {"value": v, "status": status}

    if confidence < 55:
        return jsonify({
            "screening_result": "No Conclusive Screening",
            "confidence": confidence,
            "important_parameters": important_parameters
        })

    return jsonify({
        "screening_result": disease,
        "confidence": confidence,
        "important_parameters": important_parameters
    })

# =====================================================
# RUN SERVER
# =====================================================
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

