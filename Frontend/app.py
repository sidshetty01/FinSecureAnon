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


def extract_pdf_text(file) -> str:
    """
    Extract text from a PDF UploadedFile or path-like using pdfplumber if available,
    falling back to PyPDF2. Returns a single concatenated string.
    """
    text = ""
    # Try pdfplumber first
    try:
        import pdfplumber  # type: ignore
        try:
            # Ensure file pointer at start
            if hasattr(file, "seek"):
                file.seek(0)
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    t = page.extract_text() or ""
                    text += t + "\n"
            if text.strip():
                return text
        except Exception:
            pass
    except Exception:
        pass

    # Fallback: PyPDF2
    try:
        from PyPDF2 import PdfReader  # type: ignore
        if hasattr(file, "seek"):
            file.seek(0)
        reader = PdfReader(file)
        for page in reader.pages:
            t = page.extract_text() or ""
            text += t + "\n"
        return text
    except Exception as e:
        raise RuntimeError(f"Failed to read PDF: {e}")


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
    st.caption("Upload one or more CSVs with a 'text' column OR process a local folder of CSVs.")
    batch_files = st.file_uploader("CSV file(s)", type=["csv"], accept_multiple_files=True)

    st.caption("Or upload PDFs to extract, detect, and anonymize text.")
    pdf_files = st.file_uploader("PDF file(s)", type=["pdf"], accept_multiple_files=True)

    st.divider()
    st.subheader("Process Local Folder")
    dataset_folder = st.text_input("Folder path (contains CSV files)", value="", placeholder=r"C:\\path\\to\\dataset")
    text_col_name = st.text_input("Text column name", value="text")
    run_folder = st.button("Process folder of CSVs")

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

# Batch processing (uploaded files)
if batch_files and nlp:
    try:
        dfs = []
        for f in batch_files:
            try:
                dfs.append(pd.read_csv(f))
            except UnicodeDecodeError:
                f.seek(0)
                dfs.append(pd.read_csv(f, encoding="latin-1"))
        df = pd.concat(dfs, ignore_index=True)
        if "text" not in df.columns:
            st.error("CSV must contain a 'text' column. You can also use the folder mode and specify a custom column name.")
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

            st.success(f"Processed {len(out_df)} rows from {len(batch_files)} file(s).")
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

# Folder-based batch processing
if run_folder and nlp:
    try:
        p = Path(dataset_folder).expanduser()
        if not p.exists() or not p.is_dir():
            st.error("Folder not found or not a directory.")
        else:
            import glob
            csv_paths = sorted(glob.glob(str(p / "*.csv")))
            if not csv_paths:
                st.error("No CSV files found in the folder.")
            else:
                dfs = []
                for path in csv_paths:
                    try:
                        dfs.append(pd.read_csv(path))
                    except UnicodeDecodeError:
                        dfs.append(pd.read_csv(path, encoding="latin-1"))
                df = pd.concat(dfs, ignore_index=True)
                if text_col_name not in df.columns:
                    st.error(f"Column '{text_col_name}' not found. Available columns: {list(df.columns)}")
                else:
                    results = []
                    anonymized = []
                    for t in df[text_col_name].astype(str).tolist():
                        ents = predict(nlp, t)
                        results.append(json.dumps(ents, ensure_ascii=False))
                        anonymized.append(anonymize(t, ents))
                    out_df = df.copy()
                    out_df["predictions"] = results
                    out_df["anonymized_text"] = anonymized

                    st.success(f"Processed {len(out_df)} rows from {len(csv_paths)} CSV file(s).")
                    st.dataframe(out_df.head(50))

                    csv_bytes = out_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download results CSV",
                        data=csv_bytes,
                        file_name="pii_results_folder.csv",
                        mime="text/csv",
                    )
    except Exception as e:
        st.error(f"Folder processing failed: {e}")

# PDF processing (uploaded PDFs)
if 'pdf_files' in locals() and pdf_files and nlp:
    try:
        for updf in pdf_files:
            try:
                pdf_text = extract_pdf_text(updf)
            except Exception as e:
                st.error(f"Failed to read {updf.name}: {e}")
                continue
            if not pdf_text or not pdf_text.strip():
                st.warning(f"No extractable text found in {updf.name}.")
                continue
            ents = predict(nlp, pdf_text)
            anon = anonymize(pdf_text, ents)

            with st.expander(f"PDF: {updf.name} – {len(ents)} entities detected"):
                st.markdown("**Preview (first 1500 chars, anonymized):**")
                st.code(anon[:1500] + ("..." if len(anon) > 1500 else ""))
                st.markdown("**Download anonymized full text:**")
                st.download_button(
                    label=f"Download {updf.name}.anonymized.txt",
                    data=anon.encode("utf-8"),
                    file_name=f"{updf.name}.anonymized.txt",
                    mime="text/plain",
                )
    except Exception as e:
        st.error(f"PDF processing failed: {e}")

st.caption(
    "Tip: Labels supported by the model include name, email, phone, address, credit_card, company, url, ssn."
)
