# A Reimagined RoBERTa-Based Model with Optimized Training Strategies for Deceptive Hotel Review Detection

## Overview

This study proposes an enhanced RoBERTa-based model for detecting deceptive hotel reviews through optimized training strategies rather than architectural modifications. The approach focuses on improving model generalization, reducing overfitting, and minimizing false negatives—deceptive reviews that evade detection.

The model was evaluated using the **Deceptive Opinion Spam Dataset** and compared against a baseline RoBERTa implementation based on the methodology of Hadi et al. (2025).

---

## Key Enhancements

The proposed model introduces four training optimizations:

* Cosine Learning Rate Scheduling with Linear Warmup
* Class-Balanced Weighted Cross-Entropy Loss
* Explicit Dropout Tuning (0.1)
* Early Stopping based on Validation F1-Score

Additional training stability was achieved through gradient accumulation and multi-seed evaluation.

---

## Dataset

**Deceptive Opinion Spam Dataset**

* 1,600 hotel reviews
* 800 truthful reviews
* 800 deceptive reviews
* Reviews collected from 20 hotels in Chicago
* Originally developed by Ott et al. (2011)

---

## Model Architecture

The study uses:

* `roberta-base`
* Binary sequence classification head
* Hugging Face Transformers
* PyTorch

No architectural modifications were applied to ensure performance gains result solely from training optimizations.

---

## Performance Comparison

| Metric          | Baseline | Proposed |
| --------------- | -------- | -------- |
| Accuracy        | 87.92%   | 89.58%   |
| Precision       | 83.21%   | 84.17%   |
| Recall          | 95.00%   | 97.50%   |
| F1-Score        | 88.72%   | 90.35%   |
| False Negatives | 6        | 3        |

### Key Finding

The proposed model reduced undetected deceptive reviews by **50%**, lowering false negatives from 6 to 3.

---

## Important Figures

### Baseline Conceptual Framework

![`Figure 1. Baseline Conceptual Framework`](readme_files/Baseline Conceptual Framework.png)

### Proposed Conceptual Framework

![`Figure 2. Proposed Conceptual Framework`](readme_files/Proposed Conceptual Framework.png)

### Baseline Training vs Validation Loss

![`Figure 3. Train vs Validation Loss Graph (Baseline)`](readme_files/Train vs Validation Loss Graph (Baseline).png)

### Proposed Training vs Validation Loss

![`Figure 4. Train vs Validation Loss Graph (Proposed)`](readme_files/Train vs Validation Loss Graph (Proposed).png)

### Baseline Confusion Matrix

![`Figure 5. Baseline Confusion Matrix`](readme_files/Baseline Confusion Matrix.png)

### Proposed Confusion Matrix

![`Figure 6. Proposed Confusion Matrix`](readme_files/Proposed Confusion Matrix`.png)

### Baseline vs Proposed Performance Comparison

![`Figure 7. Baseline vs Proposed Performance Comparison`](readme_files/Baseline vs Proposed Performance Comparison.png)

---

## Ablation Study

An ablation analysis was conducted to evaluate the contribution of each enhancement.

Key findings:

* Early Stopping had the greatest impact on reducing false negatives.
* Class-Balanced Loss significantly improved recall.
* Dropout Tuning improved generalization.
* Cosine Scheduling improved training stability and convergence.

---

## Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* Scikit-learn
* NumPy
* Pandas
* Matplotlib

---

## Authors

**Ryan Jay E. Compuesto**
Bachelor of Science in Computer Science
University of Mindanao

**Fletcher E. Malazarte**
Bachelor of Science in Computer Science
University of Mindanao

---

## Citation

If you use this work, please cite:

Compuesto, R. J. E., & Malazarte, F. E. (2026). *A Reimagined RoBERTa-Based Model with Optimized Training Strategies for Deceptive Hotel Review Detection*.
