# ML Basics

A collection of self-contained tutorial scripts that build machine learning and
deep learning concepts from the ground up — starting with gradient descent
implemented in plain Python lists, moving through NumPy and scikit-learn, and
ending with PyTorch neural networks for tabular and text data.

Each script is numbered and runnable on its own; later scripts build on ideas
introduced in earlier ones.

## Scripts

| Script | Topic |
| --- | --- |
| [01_ml_one_feature_regression.py](01_ml_one_feature_regression.py) | Linear regression with one feature, gradient descent implemented from scratch using plain Python, compared against scikit-learn's `SGDRegressor` and `LinearRegression` |
| [02_ml_multi_feature_regression.py](02_ml_multi_feature_regression.py) | Same idea generalized to multiple features using NumPy vectorized operations |
| [03_ml_housing_regression.py](03_ml_housing_regression.py) | Regression on the California housing dataset with scikit-learn (`SGDRegressor` vs. `LinearRegression`) |
| [04_ml_breast_cancer_classification.py](04_ml_breast_cancer_classification.py) | Binary classification on the breast cancer dataset with scikit-learn (`SGDClassifier` vs. `LogisticRegression`) |
| [05_dl_housing_regression.py](05_dl_housing_regression.py) | Housing price regression with a feed-forward neural network in PyTorch |
| [06_dl_breast_cancer_classification.py](06_dl_breast_cancer_classification.py) | Breast cancer classification with a feed-forward neural network in PyTorch |
| [07_dl_imdb_classification.py](07_dl_imdb_classification.py) | IMDB sentiment classification using a custom tokenizer/vocabulary and an embedding-based neural network in PyTorch |

Datasets used by scripts 03–07 live under [data/](data/) as CSV files.

## Requirements

- Python 3.14 (see [.python-version](.python-version))
- [uv](https://docs.astral.sh/uv/) for dependency management

Dependencies are declared in [pyproject.toml](pyproject.toml): `numpy`, `pandas`,
`scikit-learn`, and `torch` (CUDA 13.2 build by default).

## Setup

```bash
uv sync
```

This creates a `.venv` and installs all dependencies pinned in `uv.lock`.

## Running

Run any script directly with `uv run`:

```bash
uv run 01_ml_one_feature_regression.py
uv run 03_ml_housing_regression.py
```

Or activate the virtual environment and run with `python`:

```bash
# Windows
.venv\Scripts\activate
python 01_ml_one_feature_regression.py

# macOS/Linux
source .venv/bin/activate
python 01_ml_one_feature_regression.py
```

Scripts that load data from [data/](data/) (03 onward) expect to be run from the
repository root, since they reference paths like `data/housing.csv` relatively.
