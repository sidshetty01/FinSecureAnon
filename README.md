# PII Detection and Anonymization in Financial Documents
A scalable framework integrating hybrid rule-based NLP and Machine Learning approaches for detecting and anonymizing Personally Identifiable Information (PII) in financial documents. Achieves high precision, recall, and accuracy while preserving document readability.

## Overview
This project proposes a hybrid method combining pattern-based detection with a custom-trained spaCy NER model for robust PII detection and anonymization. It handles sensitive fields like names, phone numbers, emails, addresses, SSNs, credit card numbers, URLs, and company names across structured and unstructured financial texts.

## Features
---

### PII Types Detected:
1. Person Names
2. Email Addresses
3. Phone Numbers (various international formats)
4. Home and Office Addresses (single-line and multi-line formats)
5. Social Security Numbers (SSNs) (different valid structures)
6. Credit Card Numbers (Visa, MasterCard, JCB formats)
7. Company/Organization Names
8. Website URLs

---

---

### Data Handling:
The model has been Trained and Evaluated on **both Synthetic and Real-World Financial Documents** to ensure practical robustness.

| Data Type | Details |
|:----------|:--------|
| **Synthetic Training and Testing Data** | Generated using finance-related templates with embedded fake PII using Faker library. Types include: <br> - Audit Reports <br> - Compliance Reports <br> - Transaction Reports <br> - Tax Reports <br> - Invoices <br> - Customer Conversations and Feedbacks <br> - Bank Reports <br> - Bank Statements |
| **Real-World Testing Data** | Real financial documents containing naturally occurring PII. Documents include: <br> - Bank of America Audit Report <br> - Amazon Vendor Invoice <br> - CPB Software Vendor Invoice |

🔗 ** Synthetic Dataset Link:** [Mendeley Data Repository](https://doi.org/10.17632/tzrjx692jy)

---

---

### Model:

- **Architecture:** A lightweight, custom-trained spaCy NER model optimized for the financial domain.
- **Training Approach:** 
  - Combined rule-based preprocessing and automatic span annotation for efficient data labeling.
  - Iterative model training with controlled dropout and mini-batch strategy for better generalization.
- **Anonymization Placeholders Used:**
  - `[NAME REDACTED]`
  - `[EMAIL REDACTED]`
  - `[PHONE NUMBER REDACTED]`
  - `[ADDRESS REDACTED]`
  - `[SSN REDACTED]`
  - `[CREDIT CARD REDACTED]`
  - `[COMPANY NAME REDACTED]`
  - `[URL REDACTED]`
  
Detected PII entities are replaced with these placeholders to anonymize sensitive information while preserving document readability.

---

---

## 📈 Evaluation Metrics

The model has been rigorously evaluated on both synthetic and real-world data:

| Dataset                         | Precision | Recall | F1-Score | Accuracy |
|:---------------------------------|:---------:|:------:|:--------:|:--------:|
| Synthetic Financial Documents   | 94.7%     | 89.4%  | 91.1%    | 89.4%    |
| Real-World Financial Documents   | ~93%      |  -     |   -      | ~93%     |

Metrics have been computed based on entity-level detection, ensuring high reliability even in complex financial document layouts.

---

## License
This repository is released under a custom academic license for research and educational purposes only. Commercial use, redistribution, or modification requires explicit permission. Proper citation of this study is mandatory for any use.
