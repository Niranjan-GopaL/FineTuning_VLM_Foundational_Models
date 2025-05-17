# Benchmarking_Generative_Models

This repository contains code and resources for benchmarking foundational Vision-Language Models (VLMs) on a custom single-word Visual Question Answering (VQA) task, with a focus on e-commerce product images. The project includes dataset curation, baseline evaluation of multiple models, fine-tuning experiments, and model optimization.

---

## Folder Structure

```
Benchmarking_Generative_Models/
├── dataset.ipynb                # Notebook for dataset curation and processing
├── baseline_model_evaluation/   # Scripts for loading and evaluating 7 VLMs (CLIP, ViLBERT, BLIP2, BLIP, Qwen, OFA, SmolVLM)
├── dataset_csv/                 # Contains 18 CSV files with curated QA pairs
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
└── ...                          # Additional scripts and resources
```

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Niranjan-GopaL/FineTuning_VLM_Foundational_Models.git
cd FineTuning_VLM_Foundational_Models
```

### 2. Install Dependencies

It is recommended to use a virtual environment (e.g., `venv` or `conda`).

```bash
pip install -r requirements.txt
```

---

## Dataset Curation

- Run `dataset.ipynb` to process the ABO dataset and generate single-word VQA pairs.
- Curated data is stored as CSV files in the `dataset_csv/` folder (18 files, each containing question-answer pairs with difficulty levels).

---

## Baseline Model Evaluation

- The `baseline_model_evaluation/` directory contains scripts to load and evaluate the following foundational VLMs on the curated dataset:
    - CLIP
    - ViLBERT
    - BLIP
    - BLIP2
    - Qwen
    - OFA
    - SmolVLM

- Models are evaluated on VQA tasks using the curated CSVs.

---

## Fine-Tuning and Optimization

- Fine-tuning experiments (e.g., LoRA) and model compression (e.g., quantization) are included for efficient adaptation and deployment.
- Refer to the respective scripts and documentation for details.
