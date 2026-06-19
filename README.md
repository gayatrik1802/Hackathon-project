# Hackathon Orchestrate — Visual Evidence Review

# Multi-Modal Insurance Claim Verification System

## Overview

This project is an AI-powered multi-modal insurance claim verification system developed for the Hackathon challenge. The system verifies damage claims by analyzing:

* Claim images (primary source of truth)
* User claim conversation
* User claim history
* Evidence requirements

The solution determines whether the submitted evidence supports, contradicts, or does not provide enough information to validate a claim.

---

## Supported Claim Types

The system supports damage verification for the following objects:

* Car
* Laptop
* Package

---

## Features

* Multi-modal image analysis using Google Gemini Vision models
* Evidence-based claim verification
* Detection of damaged parts and issue types
* Risk flag identification
* Severity assessment
* Incremental processing with resume capability
* Automatic model fallback and retry handling
* Generates structured predictions in CSV format

---

## Project Structure

```text
Hackathon-project/
│
├── code/
│   ├── main.py
│   ├── config.py
│   ├── image_analyzer.py
│   ├── decision_engine.py
│   ├── utils.py
│   └── evaluation/
│
├── dataset/
│   ├── claims.csv
│   ├── sample_claims.csv
│   ├── user_history.csv
│   ├── evidence_requirements.csv
│   └── images/
│
├── evaluation/
├── output.csv
├── requirements.txt
└── README.md
```

---

## Technologies Used

### Programming Language

* Python 3.11+

### Libraries

* Google Gemini API
* google-genai / google-generativeai
* Pandas
* Pillow (PIL)
* tqdm
* python-dotenv

---

## Installation

### Clone Repository

```bash
git clone <repository-url>
cd Hackathon-project
```

### Create Virtual Environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
GEMINI_MODEL=gemini-2.5-flash
```

---

## Running the Project

Generate predictions:

```bash
python code/main.py
```

Run evaluation:

```bash
python code/evaluation/main.py
```

---

## Output

The generated predictions are stored in:

```text
output.csv
```

Output columns:

* user_id
* image_paths
* user_claim
* claim_object
* evidence_standard_met
* evidence_standard_met_reason
* risk_flags
* issue_type
* object_part
* claim_status
* claim_status_justification
* supporting_image_ids
* valid_image
* severity

---

## Claim Status Categories

### Supported

The images clearly support the user's claim.

### Contradicted

The images contradict the user's claim.

### Not Enough Information

The available evidence is insufficient to verify the claim.

---

## Risk Flags

Examples include:

* blurry_image
* cropped_or_obstructed
* low_light_or_glare
* wrong_angle
* wrong_object
* damage_not_visible
* possible_manipulation
* manual_review_required

---

## Model Strategy

Primary Model:

* Gemini 2.5 Flash

Fallback Models:

* Gemini 2.5 Flash Lite
* Gemini 2.0 Flash Lite (optional)

The system automatically:

* Handles rate limits
* Retries failed requests
* Switches to fallback models when quotas are exhausted
* Resumes processing from already completed rows

---

## Evaluation Metrics

The system evaluates:

* Evidence standard compliance
* Damage identification accuracy
* Object part detection
* Claim verification status
* Risk assessment quality

---

## Submission Files

The following files are included for submission:

* code/
* evaluation/
* README.md
* requirements.txt
* output.csv
* hackerrank_orchestrate/log.txt

---

## Authors

**Gayatri Dhanraj Kakde**
B.Tech Information Technology (2026)
Aurangabad, Maharashtra, India

Developed as part of the Multi-Modal Evidence Review Hackathon challenge.
