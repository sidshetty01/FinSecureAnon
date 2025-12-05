
import sys
import os
from pathlib import Path

# Add Frontend to path to import app functions
sys.path.append(os.path.join(os.getcwd(), 'Frontend'))

try:
    # Mocking streamlit file uploader
    class MockUploadedFile:
        def __init__(self, path):
            self.path = Path(path)
            self.name = self.path.name
            self.type = "application/pdf"
            self._file = open(self.path, "rb")

        def read(self, *args):
            self._file.seek(0)
            return self._file.read(*args)

        def seek(self, pos, whence=0):
            self._file.seek(pos, whence)

        def tell(self):
            return self._file.tell()

        def close(self):
            self._file.close()
            
    # Import functions from app.py
    # Note: app.py runs streamlit on import, which might be tricky. 
    # Ideally we should extract logic to a separate file, but for now we try to import select functions 
    # or just copy the logic to reproduce.
    
    print("Attempting to reproduce PDF processing logic...")
    
    import fitz # PyMuPDF
    import spacy
    
    def extract_pdf_text(file) -> str:
        text = ""
        # Try pdfplumber first
        try:
            import pdfplumber
            try:
                if hasattr(file, "seek"):
                    file.seek(0)
                with pdfplumber.open(file) as pdf:
                    for page in pdf.pages:
                        t = page.extract_text() or ""
                        text += t + "\n"
                if text.strip():
                    return text
            except Exception as e:
                print(f"pdfplumber failed: {e}")
        except Exception as e:
            print(f"pdfplumber import failed: {e}")

        # Fallback: PyPDF2
        try:
            from PyPDF2 import PdfReader
            if hasattr(file, "seek"):
                file.seek(0)
            reader = PdfReader(file)
            for page in reader.pages:
                t = page.extract_text() or ""
                text += t + "\n"
            return text
        except Exception as e:
            print(f"PyPDF2 failed: {e}")
            raise RuntimeError(f"Failed to read PDF: {e}")

    def redact_pdf(file_bytes: bytes, entities) -> bytes:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text_to_label = {ent["text"]: ent["label"] for ent in entities}

        for page in doc:
            for text, label in text_to_label.items():
                quads = page.search_for(text)
                for quad in quads:
                    page.add_redact_annot(
                        quad, 
                        text="XXXXXX", 
                        fontsize=25, 
                        fill=(1, 1, 1), 
                        text_color=(0, 0, 0)
                    )
            page.apply_redactions()

        return doc.tobytes()

    # Load Model
    model_path = Path("PII Model")
    if not model_path.exists():
        print("Model not found")
        exit(1)
        
    nlp = spacy.load(str(model_path))
    print("Model loaded.")

    # Test PDF
    pdf_path = r"Dataset/Testing/Real World Data/Amazon Vendor Invoice.pdf"
    if not os.path.exists(pdf_path):
         print(f"PDF not found: {pdf_path}")
         exit(1)

    print(f"Processing {pdf_path}...")
    
    mock_file = MockUploadedFile(pdf_path)
    
    # 1. Extract
    text = extract_pdf_text(mock_file)
    print(f"Extracted text length: {len(text)}")
    if len(text) < 100:
        print("Warning: Very little text extracted.")
        
    # 2. Detect
    doc = nlp(text)
    ents = []
    for ent in doc.ents:
        ents.append({
            "start": ent.start_char,
            "end": ent.end_char,
            "label": ent.label_.lower(),
            "text": ent.text,
        })
    print(f"Detected {len(ents)} entities.")
    
    # 3. Redact
    try:
        mock_file.seek(0)
        file_bytes = mock_file.read()
        redacted_bytes = redact_pdf(file_bytes, ents)
        print(f"Redaction successful. Output bytes: {len(redacted_bytes)}")
        
        with open("repro_redacted.pdf", "wb") as f:
            f.write(redacted_bytes)
        print("Saved repro_redacted.pdf")
        
    except Exception as e:
        print(f"Redaction FAILED: {e}")
        import traceback
        traceback.print_exc()

except Exception as e:
    print(f"General Error: {e}")
    import traceback
    traceback.print_exc()
