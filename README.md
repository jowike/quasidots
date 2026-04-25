# SITU: Quasi Dot Product-Based Time Series Classification

## Overview
This repository provides an implementation of the SITU classifier proposed in the paper:
"Time series classification with quasi dot product"

SITU extends the ROCKET framework by replacing the dot product with quasi dot product operators.

## Installation

pip install -r requirements.txt
pip install -e .

## Usage

from situ import SITUClassifier

clf = SITUClassifier(operation="AT", num_kernels=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

## Experiments

python experiments/run_example.py

## Reproducibility

- Fixed random seeds supported
- Datasets from UCR/UEA archive via aeon
- Compatible with sktime ROCKET baseline
