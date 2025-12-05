
import spacy
from pathlib import Path

# Load old model
try:
    nlp_old = spacy.load("PII Model")
    print("Loaded Original Model.")
except:
    nlp_old = None
    print("Could not load Original Model.")

# Load new model
try:
    nlp_new = spacy.load("PII Model Improved")
    print("Loaded Improved Model (Transfer Learning).")
except:
    nlp_new = None
    print("Could not load Improved Model.")

text = "Payment verified by Jane Doe from Global Tech Solutions using card 4444-5555-6666-7777."

print(f"\nTest Sentence: {text}\n")

if nlp_old:
    print("--- Original Model ---")
    doc = nlp_old(text)
    for ent in doc.ents:
        print(f"{ent.label_}: {ent.text}")

if nlp_new:
    print("\n--- Improved Model ---")
    doc = nlp_new(text)
    for ent in doc.ents:
        print(f"{ent.label_}: {ent.text}")
