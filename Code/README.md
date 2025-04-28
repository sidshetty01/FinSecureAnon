# PII Detection & Anonymization Framework
This Folder Contains the PII Detection and Anonymization Code
---

## Table of contents
1. [Key Features](#key-Features)  
2. [Code Structure](#Code-Structure)  
3. [Quick Start](#Quick-Start)  
4. [Requirements](#Requirements)  
5. [License](#License)

---

## Key Features

| Feature | Description |
|:--------|:------------|
| **Hybrid Rule-Based and ML Pipeline** | Combines simple regex-based rules and a custom-trained spaCy NER model to maximize precision and context-based PII detection. |
| **Synthetic Dataset Generation** | Creates 45,030 entries of synthetic PII data (names, emails, SSNs, phone numbers, etc.) using the Faker library, simulating realistic financial document contexts. |
| **Contextual Embedding with Templates** | Embeds generated PII into templates mimicking actual financial documents like audit reports, tax filings, and compliance notices to ensure domain realism. |
| **Noise Introduction for Robustness** | Introduces random noise such as punctuation loss to make the model resilient against real-world formatting inconsistencies. |
| **Custom Lightweight spaCy NER Model** | Trains a blank spaCy NER model optimized for PII detection with minimal resource requirements, enabling rapid training on CPU/GPU setups. |
| **Automated Annotation of True PII Positions** | Precisely annotates start and end indices of all PII types within synthetic texts, ensuring accurate supervised learning. |
| **Extensive Evaluation Metrics** | Evaluates model using precision, recall, F1 score, accuracy, confusion matrices, precision-recall curves, and ROC curves on both synthetic and real-world financial documents. |
| **Real-World Validation** | Validates model performance on actual financial audit reports and vendor invoices to demonstrate practical applicability. |
| **Anonymization Pipeline** | Replaces detected PII entities with standardized placeholders (e.g., `[NAME REDACTED]`) while maintaining document readability and semantic structure. |
| **Scalable and Efficient Deployment** | Designed for rapid, scalable deployment without reliance on heavy transformer models, balancing performance with computational efficiency. |

---

---

## Code Structure

The Structure of the Code is as Follows - 

PII Detection And Anonymization Framework ─> Importing All Required Libraries ─> Generation Of Training Dataset ─> Annotation Of True PII Data Position In Training Dataset ─> Training Model ─> Testing Dataset Generation ─> Testing Code To Get Results ─> Anonymization Of Texts To Anonymize PII Data In Texts ─> Generation of Graphs and Matrix along with Evaluation Scores

---

---

## Quick Start
```bash
# 1. Clone the repo
git clone https://github.com/Kush-Mishra-403/PII-Detection-and-Anonymization.git
cd Code

# 2. Install dependencies
python -m pip install -r requirements.txt

# 3. Run the full pipeline
python PII_Detection_and_Anonymization.py
# → Creates Training_Set.csv, Testing_Set.csv, Results.xlsx, and Saves the model in /PII Model/
```
---

## Requirements

### Hardware (Trained and Tested on)

| Component | Spec |
|-----------|------|
| **CPU**   | Intel i5-10300H @ 2.5 GHz |
| **GPU**   | Nvidia GTX 1650 Ti (4 GB) – *optional, CPU-only works* |
| **RAM**   | 8 GB DDR4 (more = faster) |

### Software & Python packages
Information on Software/Libraries of Python along with their Version are Present in requirements.txt within this Folder

---

---

## License

The source code is released under a **Custom Academic License**:

* **Permitted use** – Research and educational projects only with Proper Permission from Owner.  
* **Prohibited without prior written consent** – Any Commercial Use, Redistribution, or Modification.  
* **Attribution required** – If you Publish Work or Create Derivatives that rely on this code, you **Must Cite**:

---
