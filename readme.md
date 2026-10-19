# Diabetes Progression – Screening Model (Lab)

Predicting **disease progression one year after baseline** with the Scikit-Learn Diabetes dataset,
to help physicians identify patients at risk.

## Models compared
1. Univariate polynomial regression on BMI (degree 0 – 5)
2. Multivariate polynomial regression (degree 2 and 3)
3. Decision trees (`max_depth` 3 and 5)
4. k-Nearest Neighbours (k = 5 and 15)
5. Logistic regression classifiers (C = 1.0 and 0.01) on a high / low risk target

Regression models are evaluated with **R², MAE and MAPE**. The logistic models are evaluated with
**accuracy, precision, recall and F1**. The data is split into a **75 % train / 10 % validation / 15 % test** set.

## Project structure
```
Root
|-- data/                 (empty - the data is loaded directly from scikit-learn)
|-- docs/
|   |-- notes.md          notes on the dataset, models and results
|-- src/
|   |-- __init__.py       makes "from src import ..." possible
|   |-- data_loader.py    DiabetesDataLoader  - loads the dataset
|   |-- eda.py            DataExplorer        - statistics and EDA plots
|   |-- data_splitter.py  DataSplitter        - train / validation / test split
|   |-- model_builder.py  ModelBuilder        - creates models, counts parameters, writes equations
|   |-- model_evaluator.py ModelEvaluator     - metrics, comparison tables, result plots
|-- main.ipynb            the full report (code + explanations + conclusions)
|-- readme.md
|-- requirements.txt
```

## How to run
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```
Open `main.ipynb` in VS Code, select the `.venv` kernel, and click **Run All**.
The notebook must be run from the project root so that `from src import ...` works.

## Key results (test set)
| Model | R² | MAE | MAPE |
|---|---|---|---|
| BMI only, degree 1 polynomial | 0.20 | 54.9 | 48.7 % |
| kNN (k = 15), 8 features | 0.41 | 45.0 | 38.8 % |

| Classifier | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic regression (C = 0.01) | 0.79 | 0.77 | 0.77 | 0.77 |
