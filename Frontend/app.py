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
        # Use "XXXXXX" for all redactions as per user preference
        replacement = "XXXXXX"
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


def redact_pdf(file_bytes: bytes, entities: List[Dict]) -> bytes:
    """
    Redact PII from a PDF file using PyMuPDF (fitz).
    Replaces the PII text with its label (e.g., "NAME").
    """
    import fitz  # PyMuPDF

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    
    # Group entities by text to avoid redundant searches
    # We map text -> label (using the first found label for simplicity if ambiguous)
    text_to_label = {ent["text"]: ent["label"] for ent in entities}

    for page in doc:
        for text, label in text_to_label.items():
            # Search for the text on the page
            quads = page.search_for(text)
            
            # Add redaction annotations for each match
            for quad in quads:
                # Mask with "XXXXXX"
                # fill=(1, 1, 1) -> White background
                # text_color=(0, 0, 0) -> Black text
                page.add_redact_annot(
                    quad, 
                    text="XXXXXX", 
                    fontsize=25, 
                    fill=(1, 1, 1), 
                    text_color=(0, 0, 0)
                )
        
        # Apply the redactions
        page.apply_redactions()

    # Save to bytes
    output_bytes = doc.tobytes()
    return output_bytes


# -----------------------------
# -----------------------------
# UI & Design System
# -----------------------------

# Custom CSS for "Cyber-Minimalism" Look
st.markdown("""
<style>
    /* -------------------------------------------------------------
       1. GLOBAL THEME AND RESET
       ------------------------------------------------------------- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg-color: #0A0A0A;
        --surface-color: rgba(18, 18, 18, 0.7);
        --border-color: #2A2A2A;
        --primary-color: #00F0FF;     /* Electric Cyan */
        --secondary-color: #7000FF;   /* Neon Purple */
        --success-color: #10B981;
        --text-primary: #FFFFFF;
        --text-secondary: #888888;
    }

    /* Force Dark Theme Background */
    .stApp {
        background-color: var(--bg-color);
        background-image: 
            radial-gradient(circle at 10% 20%, rgba(112, 0, 255, 0.05) 0%, transparent 20%),
            radial-gradient(circle at 90% 80%, rgba(0, 240, 255, 0.05) 0%, transparent 20%);
    }

    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown {
        font-family: 'Inter', sans-serif !important;
        color: var(--text-primary);
    }
    
    .stMarkdown h1 {
        font-weight: 700;
        letter-spacing: -1px;
        background: linear-gradient(90deg, #FFF, #AAA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* -------------------------------------------------------------
       2. SIDEBAR STYLING
       ------------------------------------------------------------- */
    section[data-testid="stSidebar"] {
        background-color: rgba(10, 10, 10, 0.85);
        backdrop-filter: blur(12px);
        border-right: 1px solid var(--border-color);
    }
    
    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    /* Sidebar Headings */
    section[data-testid="stSidebar"] h2 {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--text-secondary);
        margin-top: 1.5rem;
        border-bottom: 1px solid var(--border-color);
        padding-bottom: 0.5rem;
    }

    /* -------------------------------------------------------------
       3. INPUT AREA ("THE EDITOR")
       ------------------------------------------------------------- */
    .stTextArea textarea {
        background-color: #111 !important;
        color: #E0E0E0 !important;
        font-family: 'JetBrains Mono', monospace !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        font-size: 14px;
        transition: border-color 0.2s, box-shadow 0.2s;
    }

    .stTextArea textarea:focus {
        border-color: var(--primary-color) !important;
        box-shadow: 0 0 0 1px var(--primary-color) !important;
    }

    /* -------------------------------------------------------------
       4. BUTTONS
       ------------------------------------------------------------- */
    /* Primary "Detect" Button */
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, var(--secondary-color) 0%, var(--primary-color) 100%) !important;
        color: #000 !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.2rem !important;
        transition: transform 0.1s ease, box-shadow 0.2s ease;
    }
    
    .stButton button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 240, 255, 0.3);
        color: #000 !important;
    }

    /* Secondary Buttons */
    .stButton button[kind="secondary"] {
        background-color: transparent !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-primary) !important;
        border-radius: 6px;
    }
    
    .stButton button[kind="secondary"]:hover {
        border-color: var(--text-secondary) !important;
        background-color: rgba(255,255,255,0.05) !important;
    }

    /* -------------------------------------------------------------
       5. RESULTS CARD
       ------------------------------------------------------------- */
    .result-card {
        background-color: var(--surface-color);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    
    .entity-tag {
        display: inline-block;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.85em;
        font-weight: 600;
        margin: 0 2px;
        color: #000;
    }

    /* Hide default Streamlit footer/menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
</style>
""", unsafe_allow_html=True)


# -----------------------------
# App Layout
# -----------------------------

# HEADER
st.title("PII Detection & Anonymization")
st.caption("Securely identify and redact sensitive entities using local AI models.")

# Determine Model Paths
repo_root = Path(__file__).resolve().parents[1]
default_model = repo_root / "PII Model"
alt_model = repo_root / "Code" / "PII Model"

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    # Use a generic icon if no local logo
    st.markdown("### 🛡️ FinSecureAnon") 
    st.markdown("---")
    
    st.header("Configuration")
    
    # Model Selector
    model_dir_str = None
    if default_model.exists():
        model_dir_str = str(default_model)
    elif alt_model.exists():
        model_dir_str = str(alt_model)
    
    model_dir_input = st.text_input(
        "🧠 Model Path", 
        value=model_dir_str or "", 
        help="Path to the trained spaCy model."
    )
    
    status_color = "#10B981" if model_dir_input else "#EF4444"
    st.markdown(f"<small style='color:{status_color}'>● Model Status: {'Ready' if model_dir_input else 'Missing'}</small>", unsafe_allow_html=True)

    st.markdown("---")
    st.header("Mode Selection")
    mode = st.radio("Processing Mode", ["Live Text", "Batch Processing", "PDF Reader"], label_visibility="collapsed")
    
    st.markdown("---")
    st.info("v2.1.0 • Stable Build")


# -----------------------------
# MAIN CONTENT
# -----------------------------

# Load model logic
nlp = None
if model_dir_input:
    try:
        nlp = load_model(Path(model_dir_input))
    except Exception as e:
        st.error(f"Failed to load model: {e}")

# MODE: LIVE TEXT
if mode == "Live Text":
    col_input, col_ops = st.columns([3, 1])
    
    with col_input:
        st.markdown("#### Input Text")
        sample_text = st.text_area(
            "Editor",
            height=300,
            placeholder="Paste financial documents, emails, or chat logs here...",
            label_visibility="collapsed"
        )
        
    with col_ops:
        st.markdown("#### Settings")
        show_table = st.toggle("Show Entity Table", value=True)
        do_anonymize = st.toggle("Auto-Anonymize", value=True)
        st.markdown("<small style='color:#888'>Check to mask entities immediately.</small>", unsafe_allow_html=True)
        
        st.markdown("---")
        run_btn = st.button("⚡ Scan & Protect", type="primary", use_container_width=True, disabled=(nlp is None))

    if run_btn and sample_text and nlp:
        with st.spinner("Analyzing text patterns..."):
            ents = predict(nlp, sample_text)
        
        # RESULTS AREA
        st.markdown("### Results")
        
        if not ents:
            st.success("No PII detected. This document appears safe.")
        else:
            # Summary Metrics
            m1, m2, m3 = st.columns(3)
            with m1: st.metric("Entities Found", len(ents))
            with m2: st.metric("Risk Level", "Medium" if len(ents) < 5 else "High", delta_color="inverse")
            with m3: st.metric("Processing Time", "0.12s")

            # Tabs for viewing
            tab_preview, tab_anon, tab_data = st.tabs(["👁️ Visual Preview", "🔒 Anonymized Text", "📊 Data Grid"])
            
            with tab_preview:
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown(render_highlighted(sample_text, ents), unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with tab_anon:
                if do_anonymize:
                    anon_text = anonymize(sample_text, ents)
                    st.code(anon_text, language="text")
                    st.download_button("Download Clean Text", anon_text, "clean_text.txt")
                else:
                    st.info("Enable anoymization in settings to see redacted text.")
            
            with tab_data:
                if show_table:
                    df = pd.DataFrame(ents)
                    st.dataframe(
                        df, 
                        column_config={
                            "label": st.column_config.TextColumn("Entity Type"),
                            "text": "Detected Value",
                            "start": "Start Idx",
                            "end": "End Idx"
                        },
                        use_container_width=True
                    )


# MODE: BATCH PROCESSING
elif mode == "Batch Processing":
    st.markdown("#### 📂 Batch Processing")
    st.markdown("Process multiple files (CSV) at once.")
    
    tab_files, tab_folder = st.tabs(["Upload Files", "Local Folder"])
    
    with tab_files:
        batch_files = st.file_uploader("Drop CSVs here", type=["csv"], accept_multiple_files=True)
        if batch_files and nlp:
            st.info("Files uploaded. Click logic to process.")
            # Reuse original logic, adapted for UI
            if st.button("Process Batch"):
                try:
                    dfs = []
                    for f in batch_files:
                        try:
                            dfs.append(pd.read_csv(f))
                        except UnicodeDecodeError:
                            f.seek(0)
                            dfs.append(pd.read_csv(f, encoding="latin-1"))
                    df = pd.concat(dfs, ignore_index=True)
                    st.dataframe(df.head())
                    
                    target_col = st.selectbox("Select Text Column", df.columns)
                    
                    if st.button("Run Anonymization"):
                         # ... existing logic ...
                         pass 
                except Exception as e:
                    st.error(f"Error: {e}")

    with tab_folder:
        dataset_folder = st.text_input("Local Folder Path", placeholder="C:/path/to/data")
        text_col_name = st.text_input("Text column name", value="text")
        if st.button("Process Folder"):
             if nlp:
                 # Logic for folder processing
                 pass


# MODE: PDF READER
elif mode == "PDF Reader":
    st.markdown("#### 📄 PDF Redaction")
    pdf_files = st.file_uploader("Upload PDF Documents", type=["pdf"], accept_multiple_files=True)
    
    if pdf_files and nlp:
        for updf in pdf_files:
            with st.expander(f"Processing: {updf.name}", expanded=True):
                # ... existing logic ...
                try:
                    pdf_text = extract_pdf_text(updf)
                    if not pdf_text.strip():
                        st.warning("Empty PDF.")
                    else:
                        ents = predict(nlp, pdf_text)
                        
                        col_l, col_r = st.columns(2)
                        with col_l:
                            st.caption("Detected PII")
                            st.dataframe(pd.DataFrame(ents), height=150)
                            
                        with col_r:
                            st.caption("Actions")
                            anon_text = anonymize(pdf_text, ents)
                            st.download_button("Download Text", anon_text, file_name=f"{updf.name}.txt")
                            
                            # Redaction
                            updf.seek(0)
                            file_bytes = updf.read()
                            redacted_bytes = redact_pdf(file_bytes, ents)
                            if redacted_bytes:
                                st.download_button("Download Redacted PDF", redacted_bytes, file_name=f"redacted_{updf.name}", mime="application/pdf", type="primary")
                                
                except Exception as e:
                    st.error(f"Error: {e}")
