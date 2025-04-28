# PII Detection and Anonymization in Financial Documents
A scalable framework integrating hybrid rule-based NLP and Machine Learning approaches for detecting and anonymizing Personally Identifiable Information (PII) in financial documents. Achieves high precision, recall, and accuracy while preserving document readability.

## Overview
This project proposes a hybrid method combining pattern-based detection with a custom-trained spaCy NER model for robust PII detection and anonymization. It handles sensitive fields like names, phone numbers, emails, addresses, SSNs, credit card numbers, URLs, and company names across structured and unstructured financial texts.

## Features
### PII Types Detected:
1. Person Names
2. Email Addresses
3. Phone Numbers (various international formats)
4. Home and Office Addresses (single-line and multi-line formats)
5. Social Security Numbers (SSNs) (different valid structures)
6. Credit Card Numbers (Visa, MasterCard, JCB formats)
7. Company/Organization Names
8. Website URLs

### Custom Anonymization:
Detected PII is replaced with placeholders like [NAME REDACTED], [EMAIL REDACTED], [PHONE NUMBER REDACTED], etc.

### Data Handling:
1. Synthetic datasets generated with realistic financial document templates.
2. Real-world audit reports and vendor invoices tested for validation.

### Model:

1. Custom-trained lightweight spaCy Named Entity Recognition (NER) model.
2. Optimized for financial document structures.

### Evaluation Metrics:
Precision, Recall, F1-Score, and Accuracy computed for synthetic and real-world data

## Performance
### Synthetic Data - Fincnaial Document Templates such as Audit Reports, Tax Reports, Invoices, Commercial Transation Record, Customer Feedbacks:
1. Precision: 94.7%
2. Recall: 89.4%
3. F1 Score: 91.1%
4. Accuracy: 89.4%

### Real-world Financial Documents - Bank of America Audit Report and Vendor Invoices:
1. Approximate Accuracy: 93%

## License
This repository is released under a custom academic license for research and educational purposes only. Commercial use, redistribution, or modification requires explicit permission. Proper citation of this study is mandatory for any use.
