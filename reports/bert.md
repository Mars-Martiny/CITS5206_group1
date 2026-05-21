# BERT Training Summary

## Purpose

This section reports the best BERT-based experiment for the model. The model was trained to classify incident text into the project target label using a BERT-style transformer encoder and a classification head. The best run was selected using validation macro F1, because macro F1 is more appropriate than accuracy for this task where some classes are less frequent.

## Best BERT Configuration

The best BERT run was `bert_hparam_trial_5`, trained using `bert-base-uncased`.

| Item | Value |
|---|---:|
| Model | `bert-base-uncased` |
| Model class | `BertClassifier` |
| Target | `potential_damage` |
| Text column | `description` |
| Fine-tuning | `True` |
| Pooling | `mean` |
| Batch size | `8` |
| Max sequence length | `160` |
| Learning rate | `9.05e-06` |
| Dropout | `0.155` |
| Weight decay | `0.01` |
| Class weights | `False` |
| Optimiser | `AdamW` |
| Scheduler | `StepLR` |
| Loss function | `CrossEntropyLoss` |
| Early stopping patience | `2` |
| Best epoch | `4` |
| Training time | `110.26 seconds` |

## Best Run Results

| Metric | Validation | Test |
|---|---:|---:|
| Loss | `0.7460` | `0.8674` |
| Accuracy | `0.7875` | `0.7208` |
| Macro F1 | `0.7062` | `0.6205` |
| Weighted F1 | `0.7959` | `0.7349` |
| Macro precision | `0.6810` | `0.5923` |
| Macro recall | `0.7514` | `0.6799` |

The validation macro F1 of `0.7062` made this the strongest BERT configuration among the reported trials. On the test set, the model achieved `72.08%` accuracy and `0.6205` macro F1. The drop from validation to test macro F1 suggests the model generalised reasonably, but still struggled with minority or harder-to-separate classes.

## Confidence-Based Project Metric

The model also met the high-confidence auto-classification requirement.

| Metric | Value |
|---|---:|
| Confidence threshold | `0.80` |
| Test auto-classification rate | `0.8208` |
| High-confidence requirement met | `True` |
| High-confidence predictions | `82.08%` |
| Medium-confidence predictions | `15.83%` |
| Low-confidence predictions | `2.08%` |

This means that more than 80% of test predictions exceeded the confidence threshold. From a business perspective, this is useful because the model could automatically classify high-confidence cases while leaving lower-confidence cases for human review.

## Implementation Summary

The BERT experiment used a shared training pipeline rather than a standalone training loop. The implementation performs label encoding, creates BERT dataloaders, builds a `BertEmbeddingConfig`, applies either CLS or mean pooling, and trains a `BertClassifier` using `AdamW` and cross-entropy loss. The code also saves the run output and records metadata such as model type, pooling strategy, fine-tuning setting, learning rate, dropout, threshold, and device.

The experiment runner supports both single BERT runs and hyperparameter search. The hyperparameter search explored fine-tuning, pooling strategy, dropout, batch size, epochs, and learning rate. This is important because transformer performance can change significantly with small learning-rate and dropout adjustments.

## SafetyBERT Experiment

SafetyBERT was also tested as an alternative pretrained encoder. In the implementation, SafetyBERT was loaded using the Hugging Face model identifier `adanish91/safetybert`, while the tokenizer was set to `bert-base-uncased`. The rest of the pipeline remained the same as the standard BERT experiment, which made the comparison fair: the encoder changed, but the dataloading, classifier structure, training loop, loss function, and evaluation process were kept consistent.

A separate best-found SafetyBERT configuration was recorded:

| Item | Value |
|---|---:|
| Model | `adanish91/safetybert` |
| Tokenizer | `bert-base-uncased` |
| Fine-tuning | `True` |
| Pooling | `cls` |
| Learning rate | `2e-05` |
| Dropout | `0.275` |
| Weight decay | `0.01` |
| Max sequence length | `160` |
| Batch size | `8` |
| Epochs | `6` |
| Scheduler | `ReduceLROnPlateau` |
| Scheduler monitor | `f1_macro` |

SafetyBERT was a useful experiment because it is a safety-focused BERT variant, so it was expected to be more aligned with incident and hazard language. However, based on the recorded best result, the strongest final run for this report was still the standard `bert-base-uncased` configuration from `bert_hparam_trial_5`.

## Key Interpretation

The final BERT result is strong enough to include as the best BERT baseline. Its main strengths are the high validation macro F1, good weighted F1, and strong high-confidence auto-classification rate. The main weakness is that test macro F1 is lower than weighted F1, which indicates that performance is likely uneven across classes. This is also supported by the per-class requirement check, where several target classes did not meet the required F1 threshold.

Overall, this run provides a strong transformer-based result for the report, but the model should still be reviewed using per-class metrics and confusion-matrix analysis before being treated as production-ready.
