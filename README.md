# Capstone Project — Incident Classifier  
## Facilitator Run Guide

This guide is intended for the project facilitator/marker to run and validate the repository for demonstration and assessment purposes.

> **Important note about the dataset:**  
> The original project dataset is not included in this GitHub repository due to NDA and confidentiality restrictions.
> For marking and demonstration, this branch includes a small **synthetic sample dataset** with the same expected column structure. This allows the repository to be installed, trained, tested, and validated without exposing client data.
> The sample records were manually generated using the examples of different damaging energy types described in the `InterSafe` damaging energy classification document. Each row contains a generic incident description, an assigned damaging energy category: `Human, Gravitational, Vehicular, Machine, Object, etc.`, and a simplified potential damage label: `Temporary / Minor Damage, Permanent Disabling Injury, or Fatal`. The dataset is not intended to represent real incidents or model performance, but to allow the repository, preprocessing pipeline, model training, and evaluation workflow to run end-to-end with data in the expected project format.

---

## 1. Project Overview

This project is a Health and Safety incident classification system. It classifies incident descriptions into:

- **Damaging Energy Type**
- **Type of Potential Damage**

The system supports:

- Data preprocessing
- Model training
- Batch inference
- Model evaluation/leaderboard inspection
- Streamlit Web UI demonstration
- CLI-based usage

The sample dataset is only for demonstration. It is not intended to represent the real client dataset or final production performance.

---

## 2. Repository Branch for Facilitator Use

Please use the facilitator release branch:

```bash
git checkout release/facilitator_use
```

This branch contains:

- Source code required to run the system
- Installation scripts
- CLI commands
- Web UI
- Synthetic sample train/validation/test datasets
- Documentation for setup and demonstration

---

## 3. Prerequisites

Before running the project, please ensure the following are installed:

- **conda**
- **Python 3.12 or higher**

You can check your Python version using:

```bash
python --version
```

---

## 4. Setup Instructions

### Option A: Automatic Setup — Recommended

The installer creates a conda environment named:

```bash
hs_classifier
```

It also installs the required Python packages and downloads the required spaCy model.

### Linux / macOS

If the shell scripts are not executable, run:

```bash
chmod +x install.sh run_ui.sh build_docs.sh
```

Then run:

```bash
./install.sh
```

To reinstall from scratch:

```bash
./install.sh --force
```

### Windows

> The Windows `.bat` scripts are converted from the tested shell scripts and may require manual troubleshooting.

```bat
install.bat
```

To reinstall from scratch:

```bat
install.bat --force
```

---

### Option B: Manual Setup

Create and activate the conda environment:

```bash
conda create -n hs_classifier python=3.12 -y
conda activate hs_classifier
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install the spaCy English model:

```bash
python -m spacy download en_core_web_sm
```

---

## 5. Sample Dataset

The real client dataset is excluded from the repository due to NDA restrictions.

For demonstration, this branch includes a small synthetic dataset using the same column format expected by the project.

Expected dataset columns:

```text
Reference,
Date and Time of Event,
Detailed Description of Event,
Energy Type,
Type of Potential Damage,
Source
```

Sample dataset files:

```text
sample_dataset/model1_train.csv
sample_dataset/model1_valid.csv
sample_dataset/model1_test.csv
```

These files are used to demonstrate that the repository can run end-to-end.

---

## 6. Running the Web UI

The Web UI is built using Streamlit.

### Linux / macOS

```bash
./run_ui.sh
```

### Windows

```bat
run_ui.bat
```

### Manual

Make sure the environment is activated:

```bash
conda activate hs_classifier
```

Then run:

```bash
streamlit run app/app.py
```

The terminal will display a local URL, usually similar to:

```text
http://localhost:8501
```

Open this link in a browser to use the interface.

---

## 7. Running the Project from CLI

The project can also be run using the command line interface:

```bash
python cli.py [COMMAND] [OPTIONS]
```

To see all available commands:

```bash
python cli.py -h
```

---

## 8. Training a Model

The `train` command trains a model using the sample dataset.

General command format:

```bash
python cli.py train \
  --train sample_dataset/model1_train.csv \
  --valid sample_dataset/model1_valid.csv \
  --test sample_dataset/model1_test.csv \
  --model-type energy \
  --architecture tf_idf
```

Recommended facilitator test command:

```bash
python cli.py train \
  --train sample_dataset/model1_train.csv \
  --valid sample_dataset/model1_valid.csv \
  --test sample_dataset/model1_test.csv \
  --model-type energy \
  --architecture tf_idf
```

This command is recommended for quick validation because TF-IDF is lightweight and does not require GPU acceleration.

On completion, the system should print:

- The saved model directory
- The best tracked metric
- Evaluation results

Saved models are written to:

```text
trained_models/
```

---

## 9. Running Batch Inference

After training a model, use the saved model directory for inference.

General format:

```bash
python cli.py infer \
  --dataset sample_dataset/model1_test.csv \
  --output results.csv \
  --energy-model path/to/saved/model_directory
```

Example:

```bash
python cli.py infer \
  --dataset sample_dataset/model1_test.csv \
  --output results.csv \
  --energy-model trained_models/<saved_model_directory>
```

Replace:

```text
<saved_model_directory>
```

with the actual folder created inside `trained_models/`.

The output file will be saved as:

```text
results.csv
```

The output includes prediction labels, confidence values, and confidence/action tiers.

---

## 10. Inspecting Model Metrics

To inspect the leaderboard:

```bash
python cli.py metrics
```

To show the top 10 energy classification runs sorted by test F1 macro:

```bash
python cli.py metrics \
  --model-type energy \
  --sort-by test_f1_macro \
  --top 10
```

To inspect a specific saved run:

```bash
python cli.py metrics --model-dir trained_models/<saved_model_directory>
```

---

## 11. Recommended Facilitator Demonstration Flow

For marking, the following sequence can be used to quickly verify that the project runs:

Install dependencies:

```bash
./install.sh
```

Activate the environment if needed:

```bash
conda activate hs_classifier
```

Train a lightweight model:

```bash
python cli.py train \
  --train sample_dataset/model1_train.csv \
  --valid sample_dataset/model1_valid.csv \
  --test sample_dataset/model1_test.csv \
  --model-type energy \
  --architecture tf_idf
```

Run inference using the saved model:

```bash
python cli.py infer \
  --dataset sample_dataset/model1_test.csv \
  --output results.csv \
  --energy-model trained_models/<saved_model_directory>
```

Inspect metrics:

```bash
python cli.py metrics
```

Optionally run the Web UI:

```bash
streamlit run app/app.py
```

---

## 12. Notes on Model Performance

The included sample dataset is synthetic and very small. Therefore:

- The results are only for repository validation.
- The metrics should not be interpreted as real project performance.
- The sample data is included so the facilitator can confirm that the codebase runs correctly.
- Real model performance was evaluated separately using the confidential client dataset.

---

## 13. Troubleshooting

### Conda environment not found

Activate the environment manually:

```bash
conda activate hs_classifier
```

If it does not exist, rerun:

```bash
./install.sh
```

or manually create it:

```bash
conda create -n hs_classifier python=3.12 -y
conda activate hs_classifier
pip install -r requirements.txt
```

---

### Missing spaCy model

If you see an error related to `en_core_web_sm`, run:

```bash
python -m spacy download en_core_web_sm
```

---

### Streamlit command not found

Make sure the environment is activated:

```bash
conda activate hs_classifier
```

Then reinstall dependencies:

```bash
pip install -r requirements.txt
```

---

### Model path error during inference

Inference requires a previously trained model directory.

First train a model:

```bash
python cli.py train \
  --train sample_dataset/model1_train.csv \
  --valid sample_dataset/model1_valid.csv \
  --test sample_dataset/model1_test.csv \
  --model-type energy \
  --architecture tf_idf
```

Then copy the generated folder name from:

```text
trained_models/
```

Use that path in the inference command:

```bash
python cli.py infer \
  --dataset sample_dataset/model1_test.csv \
  --output results.csv \
  --energy-model trained_models/<saved_model_directory>
```

---

## 14. Building Code Documentation

To build the code documentation:

### Linux / macOS

```bash
./build_docs.sh
```

### Windows

```bat
build_docs.bat
```

### Manual

```bash
cd docs
make html
```

The generated documentation will be available at:

```text
docs/build/html/index.html
```

---

## 15. Confidentiality Statement

The original incident dataset belongs to the client and is excluded from this repository due to NDA and confidentiality requirements.

The synthetic sample dataset in this branch is only provided to support marking, demonstration, and reproducibility of the code execution process.
