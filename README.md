# QuasidoTS: Quasi Dot Product-Based Time Series Classification

[![Paper](https://img.shields.io/badge/paper-Journal%20of%20Classification-blue)](https://example.org)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Official implementation of the **QuasidoTS** classifier, introduced in:

> *Time Series Classification with Quasi Dot Product* — submitted to *Journal of Classification*, 2025.

---

## What is QuasidoTS?

QuasidoTS is a fast and accurate time series classifier built on a principled generalization of the vector dot product. Like [ROCKET](https://github.com/angus924/rocket), it applies a large set of random convolutional filters to extract features, then trains a ridge regression classifier. The key innovation is replacing the standard multiplication in the dot product with alternative **quasi dot product operators** — functions that deliberately break classical algebraic properties (commutativity, associativity, neutral element) in exchange for richer feature representations.

The method is evaluated on **137 UCR benchmark datasets** and outperforms ROCKET in **59 out of 81** classic benchmark datasets, while maintaining comparable training speed.

---

## Operators

Six quasi dot product operators are available, each capturing different nonlinear relationships between filter weights and time series values:

| Key | Name | Formula |
|-----|------|---------|
| `LA` | Log–Absolute | $\log(\lvert x \cdot y + 1 \rvert)$ |
| `LE` | Log–Excision | $\max\!\left(\log(\lvert x \cdot y + 1 \rvert),\ -1\right)$ |
| `SR` | Signum–Root | $x \cdot \mathrm{sign}(y) \cdot \sqrt{\lvert y \rvert}$ |
| `SM` | Signum–Multiplication | $x \cdot \sigma(y,\, 5)$ |
| `ES` | Enhanced Sigmoidal | $\mathrm{sign}(x \cdot y) \cdot \sigma(x \cdot y,\, 5)$ |
| `AT` | Adjusted Arc Tangent | $x \cdot \tfrac{2}{\pi} \arctan(y)$ |

**`AT` is recommended** — it achieves the best overall accuracy across benchmarks, owing to its smooth, sign-preserving, and outlier-resistant properties.

---

## Installation

```bash
git clone https://github.com/your-org/quasidots.git
cd quasidots
pip install -r requirements.txt
pip install -e .
```

---

## Quick Start

```python
from quasidots import QuasidoTSClassifier

# Recommended defaults
clf = QuasidoTSClassifier(operation="AT", num_kernels=500)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

`X_train` and `X_test` should be 2-D arrays of shape `(n_samples, series_length)` or 3-D arrays of shape `(n_samples, 1, series_length)` consistent with the `sktime` / `aeon` convention.

### Selecting an operator

```python
for op in ["LA", "LE", "SR", "SM", "ES", "AT"]:
    clf = QuasidoTSClassifier(operation=op, num_kernels=500)
    clf.fit(X_train, y_train)
    print(op, clf.score(X_test, y_test))
```

### Scaling the number of kernels

500 kernels is the recommended default — accuracy saturates quickly beyond this point. The table below shows the trade-off on the 137-dataset benchmark:

| Kernels | Relative training time | Notes |
|--------:|----------------------:|-------|
| 50 | ~0.1× | Useful for rapid prototyping |
| 100 | ~0.2× | Significant accuracy jump from 50 |
| **500** | **1×** | **Recommended default** |
| 1 000 | ~2× | Marginal gains over 500 |
| 10 000 | ~20× | Diminishing returns |

---

## Reproducing Paper Experiments

Benchmark results from the paper can be reproduced against the full UCR archive (137 datasets) via:

```bash
python experiments/run_example.py
```

Datasets are loaded automatically through [aeon](https://github.com/aeon-toolkit/aeon). A ROCKET baseline using the same experimental setup (number of kernels, 10 repetitions per dataset) is included for direct comparison.

Results are saved to `results/` as CSV files and match the accuracy values reported in Tables 1–2 of the paper.

### Reproducibility details

- Fixed random seeds are supported via the `random_state` parameter.
- No preprocessing is applied to raw time series (consistent with the paper).
- Variable-length series are zero-padded to the length of the longest series in the dataset.
- Ridge regression with L2 regularisation is used in one-vs-rest mode; the regularisation parameter is selected via cross-validation.

---

## How It Works

Feature extraction follows a two-step process for each random filter:

1. A filter of random length $k \in \{7, 9, 11\}$ is drawn from $\mathcal{N}(0, 1)$ and applied to each position in the time series with stride 1, computing a **quasi dot product** at every position.
2. The **maximum** of the resulting sequence (after adding a bias term drawn from $\mathcal{U}(-1, 1)$) is retained as a single scalar feature.

This yields one feature per filter, for a total of `num_kernels` features per time series. The feature matrix is then passed to a ridge regression classifier. The computational complexity is $\mathcal{O}(\text{kernels} \times \text{samples} \times \text{length})$ — equivalent to ROCKET.

---

## Comparison with ROCKET

On the 81-dataset "bake-off" benchmark (500 kernels each):

| | Datasets won |
|---|---|
| **QuasidoTS (AT)** | **59** |
| Tie | 6 |
| ROCKET | 17 |

QuasidoTS with 500 kernels also outperforms ROCKET with 10 000 kernels, while training roughly 20× faster.

---

## Citation

If you use QuasidoTS in your research, please cite:

```bibtex
@article{quasidots2025,
  title   = {Time Series Classification with Quasi Dot Product},
  journal = {Journal of Classification},
  year    = {2025},
  note    = {Preprint submitted May 2025}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.