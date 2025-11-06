import json
from pathlib import Path
from typing import List, Dict, Tuple
import html

import streamlit as st
import pandas as pd
import spacy

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="PII Detection & Anonymization", layout="wide")

LABEL_COLORS = {
    "name": "#E74C3C",
    "email": "#3498DB",
    "phone": "#9B59B6",
    "address": "#16A085",
    "credit_card": "#F39C12",
    "company": "#2ECC71",
    "url": "#1ABC9C",
    "ssn": "#E67E22",
}

REPLACEMENTS = {
    "name": "[NAME REDACTED]",
    "email": "[EMAIL REDACTED]",
    "phone": "[PHONE REDACTED]",
    "address": "[ADDRESS REDACTED]",
    "credit_card": "[CREDIT CARD REDACTED]",
    "company": "[COMPANY REDACTED]",
    "url": "[URL REDACTED]",
    "ssn": "[SSN REDACTED]",
}

# -----------------------------
# Helpers
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")
    return spacy.load(str(model_path))


def predict(nlp, text: str) -> List[Dict]:
    doc = nlp(text)
    ents = []
    for ent in doc.ents:
        ents.append({
            "start": ent.start_char,
            "end": ent.end_char,
            "label": ent.label_.lower(),
            "text": ent.text,
        })
    return ents


def anonymize(text: str, ents: List[Dict]) -> str:
    # replace from end to start to keep spans stable
    out = text
    for ent in sorted(ents, key=lambda e: e["start"], reverse=True):
        label = ent["label"].lower()
        replacement = REPLACEMENTS.get(label, "[REDACTED]")
        out = out[: ent["start"]] + replacement + out[ent["end"] :]
    return out


def render_highlighted(text: str, ents: List[Dict]) -> str:
    # Build HTML with colored spans
    parts = []
    last = 0
    for ent in sorted(ents, key=lambda e: e["start"]):
        color = LABEL_COLORS.get(ent["label"].lower(), "#BDC3C7")
        parts.append(html.escape(text[last:ent["start"]]))
        span = f"<span style='background-color:{color}; padding:2px 4px; border-radius:3px;' title='{ent['label']}'>{html.escape(text[ent['start']:ent['end']])}</span>"
        parts.append(span)
        last = ent["end"]
    parts.append(html.escape(text[last:]))
    return "".join(parts)


# -----------------------------
# UI
# -----------------------------
st.title("PII Detection & Anonymization")

# Try default model path under repo
repo_root = Path(__file__).resolve().parents[1]
default_model = repo_root / "PII Model"
alt_model = repo_root / "Code" / "PII Model"

with st.sidebar:
    st.header("Model")
    use_alt = False
    model_dir_str = None
    if default_model.exists():
        model_dir_str = str(default_model)
    elif alt_model.exists():
        model_dir_str = str(alt_model)
    model_dir_input = st.text_input("Model directory", value=model_dir_str or "", placeholder="path/to/PII Model")
    st.caption("Train the model first using the Code script, which saves to 'PII Model'.")

    st.header("Batch Processing")
    st.caption("Upload a CSV with a 'text' column.")
    batch_file = st.file_uploader("CSV file", type=["csv"], accept_multiple_files=False)

# Load model
nlp = None
if model_dir_input:
    try:
        nlp = load_model(Path(model_dir_input))
    except Exception as e:
        st.error(f"Failed to load model: {e}")
else:
    st.warning("Please provide a model directory.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Single Text")
    sample_text = st.text_area(
        "Enter text",
        height=220,
        placeholder="Paste a document containing PII...",
    )
    run = st.button("Detect PII", type="primary", disabled=nlp is None)

with col2:
    st.subheader("Options")
    show_table = st.checkbox("Show entities table", value=True)
    do_anonymize = st.checkbox("Anonymize detected PII", value=True)

if run and sample_text and nlp:
    ents = predict(nlp, sample_text)

    if len(ents) == 0:
        st.info("No entities detected.")

    # Highlighted view
    st.markdown("**Detected Entities (highlighted):**")
    st.markdown(render_highlighted(sample_text, ents), unsafe_allow_html=True)

    # Table
    if show_table and ents:
        st.dataframe(pd.DataFrame(ents))

    # Anonymized
    if do_anonymize and ents:
        anon = anonymize(sample_text, ents)
        st.markdown("**Anonymized Text:**")
        st.code(anon)

st.divider()

# Batch processing
if batch_file is not None and nlp:
    try:
        df = pd.read_csv(batch_file)
        if "text" not in df.columns:
            st.error("CSV must contain a 'text' column.")
        else:
            results = []
            anonymized = []
            for t in df["text"].astype(str).tolist():
                ents = predict(nlp, t)
                results.append(json.dumps(ents, ensure_ascii=False))
                anonymized.append(anonymize(t, ents))
            out_df = df.copy()
            out_df["predictions"] = results
            out_df["anonymized_text"] = anonymized

            st.success(f"Processed {len(out_df)} rows.")
            st.dataframe(out_df.head(50))

            csv_bytes = out_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download results CSV",
                data=csv_bytes,
                file_name="pii_results.csv",
                mime="text/csv",
            )
    except Exception as e:
        st.error(f"Batch processing failed: {e}")

st.caption(
    "Tip: Labels supported by the model include name, email, phone, address, credit_card, company, url, ssn."
)
