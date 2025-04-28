# PII Detection and Anonymization in Financial Documents
A scalable framework integrating hybrid rule-based NLP and Machine Learning approaches for detecting and anonymizing Personally Identifiable Information (PII) in financial documents. Achieves high precision, recall, and accuracy while preserving document readability.

## Overview
This project proposes a hybrid method combining pattern-based detection with a custom-trained spaCy NER model for robust PII detection and anonymization. It handles sensitive fields like names, phone numbers, emails, addresses, SSNs, credit card numbers, URLs, and company names across structured and unstructured financial texts.

## Features
1. Hybrid Rule-Based + ML-Based PII Detection
2. Anonymization with Placeholder Substitution
3. Synthetic Dataset Generation with Contextual Embedding
4. Real-world Financial Document Testing
5. Precision, Recall, F1 Score, and Accuracy Evaluation

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
