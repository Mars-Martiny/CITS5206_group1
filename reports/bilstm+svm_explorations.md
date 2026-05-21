# Model Exploration Report

## Overview

Several alternative model architectures were explored during development to investigate whether different combinations of traditional machine learning and deep learning approaches could improve classification performance on mining incident reports.

The explored approaches included:
- TF-IDF + SVM
- Vanilla Bi-LSTM
- Bi-LSTM + SVM feature extraction

The experiments focused on both energy classification and damage classification tasks.

---

## Why These Models Were Explored

### Why TF-IDF + SVM Was Selected

TF-IDF + SVM was explored as a traditional machine learning baseline.

TF-IDF is a widely used statistical text representation method that converts documents into weighted word-frequency vectors. Combined with a Support Vector Machine (SVM) classifier, it forms a common and computationally efficient NLP pipeline.

This approach was explored because:
- the dataset size is relatively small
- the dataset contains significant class imbalance
- traditional machine learning models can sometimes generalise better under limited-data conditions
- simpler pipelines are easier to maintain and interpret

TF-IDF + SVM was used as a comparison baseline against more complex deep learning architectures.

---

### Why Bi-LSTM Was Selected

Bi-LSTM was selected because it is a commonly used sequence-based architecture for natural language processing tasks.

Unlike TF-IDF, which treats text as unordered word frequencies, Bi-LSTM processes text sequentially and bidirectionally. This allows the model to capture contextual relationships and token ordering within incident descriptions.

For example:
- "Worker struck the machine"
- "Machine struck the worker"

These sentences appear similar under TF-IDF but represent different injury mechanisms.

Bi-LSTM was explored to investigate whether contextual sequence modelling could improve:
- damaging energy prediction
- critical risk classification
- minority-class identification

---

### Why Bi-LSTM + SVM Was Selected

A hybrid Bi-LSTM + SVM architecture was also explored.

The objective was to combine:
- contextual feature extraction from Bi-LSTM
- simpler and potentially more stable classification boundaries from SVM

In this pipeline:
1. Bi-LSTM was used as a feature extractor
2. contextual embedding vectors were generated
3. an SVM classifier performed downstream classification

The hybrid approach was explored because traditional classifiers can sometimes outperform larger neural models on smaller datasets.

---

## Experimental Findings

| Model Variant | run | Best Validation F1 macro | Best Test F1 macro | General Observation |
|---|---|---|---|---|
| TF-IDF + SVM | 1 | 0.60 | | 0.59 | Stable and consistent |
| Vanilla Bi-LSTM | 21 | 0.33 | 0.32 | Context-aware but unstable |
| Bi-LSTM + SVM | 1 | 0.35 | 0.35 | Increased complexity with limited improvement |

### Key observations

- TF-IDF + SVM remained one of the most stable baseline approaches explored during experimentation.
- Bi-LSTM architectures were able to capture contextual information unavailable to TF-IDF.
- Validation performance for Bi-LSTM-based models was unstable across runs.
- Strong overfitting was observed during training.
- Minority-class prediction performance remained weak.
- The Bi-LSTM + SVM hybrid architecture increased pipeline complexity without providing sufficiently strong performance improvements.

The dataset contains approximately 1,100 labelled samples with severe class imbalance. Several classes contain only one or several examples, making robust generalisation difficult for both deep learning and traditional classifiers.

---

## Tuning Attempts

Several approaches were explored to reduce overfitting and improve generalisation:
- reducing batch size
- increasing pooling
- simplifying hidden dimensions
- adjusting dropout configurations

These changes produced only limited improvements.

---

## Comparison Against TF-IDF + SVM

TF-IDF + SVM remained highly competitive despite its relatively simple architecture.

Although Bi-LSTM-based approaches were able to model contextual sequence information, the observed performance improvements were generally insufficient relative to:
- increased training complexity
- additional maintenance overhead
- larger pipeline complexity

The hybrid Bi-LSTM + SVM pipeline in particular did not significantly outperform the simpler TF-IDF + SVM baseline.

---

## Experimental Branch Reference

The experimental implementations explored during this investigation were retained in a separate development branch rather than merged into the final production pipeline.

Relevant implementations, including:
- Bi-LSTM exploration
- Bi-LSTM + SVM feature extraction
- TF-IDF + SVM experiments

can be found in the following experimental branch:

`feature/Explorations`

These implementations were preserved for future exploration and client reference.

---

## Summary

The explored architectures demonstrated that contextual sequence modelling can capture information unavailable to traditional bag-of-words approaches.

However, the experiments remained heavily constrained by:
- small dataset size
- severe class imbalance
- limited minority-class examples
- overfitting

Although the explored models provided useful engineering and research insights, the observed performance improvements were insufficient to justify integration into the final production development pipeline.

The experimental implementations were therefore retained as research-only branches for future exploration and client reference rather than merged into the main pipeline.
