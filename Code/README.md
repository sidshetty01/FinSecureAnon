# PII Detection & Anonymization Framework
This Folder Contains the PII Detection and Anonymization Code
---

## Table of contents
1. [Key Features](#key-Features)  
2. [Code Structure](#Code-Structure)  
3. [Quick Start](#quick-start)  
4. [Requirements](#requirements)  
5. [License](#license)

---

## Key Features

| # | Feature | Why It Matters |
|---|---------|---------------|
| 1 | **Hybrid Rule-Based And ML Pipeline** | Simple rules grab the easy PII; a spaCy model handles the tricky, context-based cases. |
| 2 | **Big Synthetic Training Data** | Faker makes 45 k fake names, cards, SSNs, etc., so no real data is exposed. |
| 3 | **Realistic Finance Templates** | PII is dropped into audit letters, invoices, and tax mails, so the model learns true document flow. |
| 4 | **Noise And Format Variety** | Random punctuation loss, multi-line addresses, and global phone codes toughen the model for messy text. |
| 5 | **Lightweight Custom spaCy NER** | Trains fast on a laptop; easy to add new PII labels with `ner.add_label()`. |
| 6 | **Auto Span Annotation** | Code marks the exact start-end of every fake PII, saving many hours of manual tagging. |
| 7 | **Careful Training Loop** | 20 passes, mini-batches, and dropout keep the model sharp without over-fit. |
| 8 | **Full Evaluation Metrics** | Prints precision, recall, F1, accuracy, plus confusion, ROC, and PR curves. |
| 9 | **Safe Token-Level Anonymization** | Replaces each hit with tags like `[NAME REDACTED]`, so files stay readable. |
|10| **Ready For Scale And Simple Deploy** | Pure-Python (spaCy, Faker, pandas). Runs end-to-end with no GPU. |

---

---

## Code Structure

The Structure of the Code is as Follows - 

PII Detection And Anonymization Framework
  └─> Importing All Required Libraries
      └─> Generation Of Training Dataset
          └─> Annotation Of True PII Data Position In Training Dataset
              └─> Training Model
                  └─> Testing Dataset Generation
                      └─> Testing Code To Get Results
                          └─> Anonymization Of Texts To Anonymize PII Data In Texts
                              └─> Generation of Graphs and Matrix along with Evaluation Scores

---

---
## Quick Start
