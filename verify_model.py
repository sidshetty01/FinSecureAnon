import spacy
from pathlib import Path

try:
    model_path = Path("PII Model")
    if not model_path.exists():
        print(f"Model path {model_path} does not exist.")
        exit(1)
        
    print(f"Loading model from {model_path.absolute()}...")
    nlp = spacy.load(str(model_path))
    print("Model loaded successfully.")
    
    text = "John Smith works at Acme Corp. His email is john.smith@acme.com and phone is 555-123-4567. His SSN is 000-12-3456."
    print(f"Processing text: {text}")
    
    doc = nlp(text)
    ents = []
    for ent in doc.ents:
        ents.append((ent.label_, ent.text))
        
    print(f"Detected {len(ents)} entities:")
    for label, text in ents:
        print(f" - {label}: {text}")
        
except Exception as e:
    print(f"Error: {e}")
