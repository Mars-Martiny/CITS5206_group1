# Bi-GRU

## Overview

Bi-GRU is a lightweight recurrent model used as a baseline for both the energy and damage classification tasks. Three embedding strategies were evaluated: vanilla trainable embeddings, static SafetyBERT embeddings, and contextual SafetyBERT embeddings.

### Why Bi-GRU can do things TF-IDF cannot

TF-IDF is a bag-of-words model — it counts which words appear in a document but throws away their order. This means the sentences _"Worker fell on the truck"_ and _"Truck fell on the worker"_ produce **identical feature vectors** and are therefore indistinguishable to a TF-IDF classifier, even though the injury mechanism is completely different.

Bi-GRU processes tokens sequentially and in both directions, so it can represent _who did what to whom_. The hidden state at each position carries information about what came before and after it, allowing the model to distinguish subject-verb-object ordering. This is particularly relevant for incident reports where the direction of force or the role of the victim (struck-by vs. struck-against, fell vs. fell-on-by) changes the class label.

---

## Energy Classification

| Embedding | Runs | Best Val F1 | Best Test F1 | Avg Test F1 |
|---|---|---|---|---|
| Vanilla | 21 | 0.4735 | 0.3697 | 0.3328 |
| SafetyBERT static | 9 | 0.5035 | 0.4294 | 0.3878 |
| SafetyBERT contextual | 6 | 0.5568 | 0.4931 | 0.4734 |

**Key observations:**
- Vanilla Bi-GRU validation F1 clusters around 0.44–0.47 across runs, consistent with the ~0.46 baseline expectation.
- Static SafetyBERT embedding shows mixed results: earlier experiments (May 7) showed slight degradation (0.43 vs 0.44 val F1) while later runs closed the gap — no consistent improvement over vanilla on average test F1.
- Contextual SafetyBERT is the clear winner for this task, pushing validation F1 up to **0.56** and test F1 to **0.49**, a substantial gain over vanilla.

### Best Contextual Run — Per-class F1 (Test)

| Class | F1 | Target | Met |
|---|---|---|---|
| class_5 | 0.7640 | 0.70 | ✅ |
| class_6 | 0.6742 | 0.70 | ❌ |
| class_11 | 0.5263 | 0.70 | ❌ |
| class_17 | 0.8043 | 0.70 | ✅ |

Class 11 remains the hardest to classify across all Bi-GRU variants.

---

## Damage Classification

| Embedding | Runs | Best Val F1 | Best Test F1 |
|---|---|---|---|
| Vanilla | 8 | 0.6070 | 0.5410 |
| SafetyBERT static | 2 | 0.5808 | 0.5953 |
| SafetyBERT contextual | 2 | 0.6611 | 0.5071 |

Damage classification consistently achieves higher F1 than energy classification across all embedding variants, suggesting the task is comparatively easier or the class distribution is more learnable for this architecture.

---

## Summary

Contextual SafetyBERT embeddings provide the most consistent uplift for Bi-GRU. Static embeddings do not reliably improve over vanilla and sometimes hurt performance, likely because freezing the embeddings prevents the model from adapting to task-specific vocabulary. The architecture as a whole is limited — even the best contextual run does not meet all per-class F1 targets.
