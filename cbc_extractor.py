"""
CBC Extractor – SympCare (GENERAL + ROBUST)

✔ Digital + scanned PDFs
✔ Extremely tolerant aliases
✔ Table / multiline aware
✔ Reference-range safe
✔ Unit normalization
✔ Clinically plausible filtering
✔ Flask-safe (single PDF)

Dependencies:
pillow pytesseract opencv-python numpy PyMuPDF pdf2image
System:
tesseract-ocr, poppler-utils
"""

import re
import math
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import fitz  # PyMuPDF
import numpy as np
import cv2
import pytesseract
from PIL import Image
from pdf2image import convert_from_path


# ==========================================================
# CBC FIELD DEFINITIONS (CANONICAL + ALIASES + PLAUSIBLE RANGES)
# ==========================================================
CBC_FIELDS: Dict[str, Dict] = {
    "Hemoglobin": {
        "aliases": ["hb", "hgb", "hemoglobin", "haemoglobin"],
        "range": (2, 25)
    },
    "Hematocrit": {
        "aliases": ["hct", "hematocrit", "haematocrit", "pcv"],
        "range": (10, 70)
    },
    "RBC Count": {
        "aliases": ["rbc", "rbc count", "red blood cell","r.b.c","r.b.c count"],
        "range": (1, 10)
    },
    "MCV": {
        "aliases": ["mcv"],
        "range": (50, 130)
    },
    "MCH": {
        "aliases": ["mch"],
        "range": (15, 40)
    },
    "MCHC": {
        "aliases": ["mchc"],
        "range": (25, 40)
    },
    "RDW-CV": {
        "aliases": ["rdw", "rdw-cv"],
        "range": (8, 25)
    },
    "Platelet Count": {
        "aliases": ["platelet", "platelet count", "plt"],
        "range": (20000, 1000000)
    },
    "Total Leucocyte Count": {
        "aliases": ["wbc", "total wbc", "leucocyte", "tlc"],
        "range": (1000, 100000)
    },
    "Neutrophils (%)": {
        "aliases": ["neutrophil", "neutrophils", "neut %"],
        "range": (0, 100)
    },
    "Lymphocytes (%)": {
        "aliases": ["lymphocyte", "lymphocytes", "lymph %"],
        "range": (0, 100)
    },
    "Monocytes (%)": {
        "aliases": ["monocyte", "monocytes", "mono %"],
        "range": (0, 100)
    },
    "Eosinophils (%)": {
        "aliases": ["eosinophil", "eosinophils", "eos %"],
        "range": (0, 100)
    },
    "Basophils (%)": {
        "aliases": ["basophil", "basophils", "baso %"],
        "range": (0, 10)
    }
}


CSV_COLUMNS = ["SourceFile", "FileHash", "PagesUsed"] + list(CBC_FIELDS.keys())


# ==========================================================
# REGEX
# ==========================================================
NUM_RE = re.compile(r"\d+(?:\.\d+)?")
RANGE_RE = re.compile(r"\b\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?\b")


# ==========================================================
# OCR PREPROCESSING
# ==========================================================
def preprocess_image(img: Image.Image) -> np.ndarray:
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 9
    )


def ocr_image(img: Image.Image) -> str:
    return pytesseract.image_to_string(
        preprocess_image(img),
        config="--psm 6"
    )


# ==========================================================
# UTILITIES
# ==========================================================
def compute_file_hash(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_value(text: str, value: float) -> float:
    t = text.lower()

    if "x10^3" in t or "×10^3" in t:
        return value * 1000

    if "x10^6" in t or "×10^6" in t:
        return value * 1000000

    if "lakh" in t or "lac" in t:
        return value * 100000

    return value


def pick_plausible(field: str, values: List[float]) -> Optional[float]:
    lo, hi = CBC_FIELDS[field]["range"]
    for v in values:
        if lo <= v <= hi:
            return v
    return values[0] if values else None


# ==========================================================
# CORE EXTRACTION LOGIC
# ==========================================================
def extract_from_lines(lines: List[str]) -> Dict[str, float]:
    results: Dict[str, float] = {}

    for i, line in enumerate(lines):
        low = line.lower()

        for field, meta in CBC_FIELDS.items():
            if field in results:
                continue

            if not any(a in low for a in meta["aliases"]):
                continue

            values: List[float] = []

            # Look on same line + next 2 lines (table-safe)
            for j in range(3):
                if i + j >= len(lines):
                    continue

                ln = lines[i + j]
                if RANGE_RE.search(ln):
                    ln = RANGE_RE.split(ln)[0]

                nums = [float(n) for n in NUM_RE.findall(ln)]
                for n in nums:
                    values.append(normalize_value(ln, n))

            chosen = pick_plausible(field, values)
            if chosen is not None:
                results[field] = chosen

    return results


# ==========================================================
# PDF ENTRY POINT
# ==========================================================
def extract_best_cbc_from_pdf(pdf_path: Path) -> Tuple[Dict[str, float], List[int]]:
    doc = fitz.open(str(pdf_path))
    page_texts: List[Tuple[int, str]] = []

    for i in range(len(doc)):
        txt = doc[i].get_text()
        if txt and len(txt.strip()) > 30:
            page_texts.append((i + 1, txt))

    # Digital PDF path
    if page_texts:
        combined = "\n".join(t for _, t in page_texts)
        result = extract_from_lines(combined.splitlines())
        if result:
            return result, [p for p, _ in page_texts]

    # OCR fallback
    images = convert_from_path(str(pdf_path), dpi=300)
    ocr_text = "\n".join(ocr_image(img) for img in images)
    result = extract_from_lines(ocr_text.splitlines())

    return result, list(range(1, len(images) + 1))


# ==========================================================
# FLASK-SAFE FUNCTION
# ==========================================================
def extract_cbc_from_pdf_to_row(pdf_path: Path) -> Dict[str, object]:
    values, pages = extract_best_cbc_from_pdf(pdf_path)

    row = {col: "" for col in CSV_COLUMNS}
    row["SourceFile"] = pdf_path.name
    row["FileHash"] = compute_file_hash(pdf_path)
    row["PagesUsed"] = ",".join(map(str, pages))

    for field in CBC_FIELDS:
        if field in values:
            row[field] = values[field]

    return row
