# Looped Transformer

## Known Issues

- **AutoTokenizer from `adanish91/safetybert` is broken** and cannot be used with this architecture. The `AutoModel` still loads correctly, so the model weights remain available.

---

## Overview

The Looped Transformer is a weight-shared encoder stack where the same transformer block is applied repeatedly across multiple "loops." All runs used vanilla trainable embeddings (no SafetyBERT integration, due to the tokenizer issue above).

---

## Energy Classification

| Runs (valid) | Best Val F1 | Best Test F1 | Avg Test F1 |
|---|---|---|---|
| 14 | 0.4573 | 0.4109 | 0.2231 |

> The low average test F1 (0.22) is inflated downward by several collapsed runs that predict a single class. The best runs are competitive with Bi-GRU vanilla.

### Best Run — Per-class F1 (Test)

| Class | F1 | Target | Met |
|---|---|---|---|
| Gravitational | 0.6526 | 0.70 | ❌ |
| Human | 0.7126 | 0.70 | ✅ |
| Other | 0.4186 | 0.70 | ❌ |
| Vehicular | 0.7500 | 0.70 | ✅ |

The "Other" category is consistently the hardest for this model.

---

## Damage Classification

| Runs | Best Val F1 | Best Test F1 |
|---|---|---|
| 1 | 0.5180 | 0.4803 |

Only one well-behaved damage run is available, limiting conclusions.

---

## Observations

- The model is sensitive to initialisation: several runs collapse to predicting a single class (test F1 near 0.02–0.15), which skews aggregate statistics.
- Despite weight sharing, the Looped Transformer does not outperform vanilla Bi-GRU on average. The best individual runs reach comparable F1 (~0.41 test), but training is less stable.
- SafetyBERT contextual embeddings could not be tested due to the broken AutoTokenizer. This is the most likely path to improvement if the tokenizer issue is resolved.
