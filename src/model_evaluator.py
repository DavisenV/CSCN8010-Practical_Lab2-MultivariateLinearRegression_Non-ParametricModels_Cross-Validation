"""
model_evaluator.py
------------------
Computes metrics and collects them in comparison tables.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    precision_score,
    r2_score,
    recall_score,
)


class ModelEvaluator:
    """Keeps a list of results (one row per model) that can be shown as a table."""

    def __init__(self):
        self.rows = []

    # ---------- Metric helpers ----------
    @staticmethod
    def regression_metrics(y_true, y_pred) -> dict:
        """R-squared, MAE and MAPE for a regression model.

        - R2  : share of the variation in y the model explains (1 = perfect, 0 = no better
                than always predicting the mean, negative = worse than the mean).
        - MAE : average absolute error, in the SAME units as the target.
        - MAPE: average absolute error as a PERCENTAGE of the true value.
                sklearn returns a fraction (0.35), so we multiply by 100 -> 35 %.
        """
        return {
            "R2": r2_score(y_true, y_pred),
            "MAE": mean_absolute_error(y_true, y_pred),
            "MAPE (%)": mean_absolute_percentage_error(y_true, y_pred) * 100,
        }

    @staticmethod
    def classification_metrics(y_true, y_pred) -> dict:
        """Accuracy, precision, recall and F1 for a classifier (positive class = 1 = high risk).

        - Accuracy : share of all patients classified correctly.
        - Precision: of the patients we FLAG as high risk, how many really are?
        - Recall   : of the patients who REALLY are high risk, how many did we catch?
                     (also called sensitivity - the key number for a screening tool)
        - F1       : harmonic mean of precision and recall (balances the two).
        """
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1": f1_score(y_true, y_pred, zero_division=0),
        }

    # ---------- Collect results ----------
    def add_regression_result(self, name, model, X_train, y_train, X_val, y_val, n_params=None):
        """Compute train and validation metrics for an already FITTED regression model."""
        train = self.regression_metrics(y_train, model.predict(X_train))
        val = self.regression_metrics(y_val, model.predict(X_val))
        row = {"Model": name, "Parameters": n_params}
        row.update({f"Train {k}": v for k, v in train.items()})
        row.update({f"Val {k}": v for k, v in val.items()})
        self.rows.append(row)

    def add_classification_result(self, name, model, X_train, y_train, X_val, y_val, n_params=None):
        """Compute train and validation metrics for an already FITTED classifier."""
        train = self.classification_metrics(y_train, model.predict(X_train))
        val = self.classification_metrics(y_val, model.predict(X_val))
        row = {"Model": name, "Parameters": n_params}
        row.update({f"Train {k}": v for k, v in train.items()})
        row.update({f"Val {k}": v for k, v in val.items()})
        self.rows.append(row)

    def results_table(self, decimals: int = 3) -> pd.DataFrame:
        """All collected rows as a DataFrame (one row per model)."""
        return pd.DataFrame(self.rows).set_index("Model").round(decimals)

    def best_model(self, metric: str = "Val R2", higher_is_better: bool = True) -> str:
        """Name of the model with the best value of `metric` on the validation set."""
        table = self.results_table()
        return table[metric].idxmax() if higher_is_better else table[metric].idxmin()

    @staticmethod
    def error_by_group(y_true, y_pred, bins, labels) -> pd.DataFrame:
        """Split patients into groups by their TRUE progression and show the error per group.

        This tells us WHERE the model fails (e.g. it may be fine for average patients
        but bad for the most severe ones).
        """
        df = pd.DataFrame({"actual": np.asarray(y_true), "predicted": np.asarray(y_pred)})
        df["group"] = pd.cut(df["actual"], bins=bins, labels=labels)
        df["error"] = df["predicted"] - df["actual"]          # > 0 means over-prediction
        df["abs_pct_error"] = (df["error"].abs() / df["actual"]) * 100
        summary = df.groupby("group", observed=False).agg(
            patients=("actual", "size"),
            mean_actual=("actual", "mean"),
            mean_predicted=("predicted", "mean"),
            mean_error=("error", "mean"),
            MAE=("error", lambda e: e.abs().mean()),
            MAPE_pct=("abs_pct_error", "mean"),
        )
        return summary.round(1)

    # ---------- Plots ----------
    @staticmethod
    def plot_residuals(y_true, y_pred, title: str = "Residuals"):
        """Left: predicted vs actual (perfect = on the diagonal). Right: residuals vs actual."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        residuals = y_true - y_pred
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].scatter(y_true, y_pred, alpha=0.6)
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        axes[0].plot(lims, lims, "r--", label="Perfect prediction")
        axes[0].set_xlabel("Actual progression")
        axes[0].set_ylabel("Predicted progression")
        axes[0].set_title("Predicted vs actual")
        axes[0].legend()

        axes[1].scatter(y_true, residuals, alpha=0.6)
        axes[1].axhline(0, color="red", linestyle="--")
        axes[1].set_xlabel("Actual progression")
        axes[1].set_ylabel("Residual (actual - predicted)")
        axes[1].set_title("Residuals vs actual")

        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_univariate_fit(model, feature, datasets: dict, title: str):
        """Plot data points of each set and the fitted curve on top (one panel per set).

        datasets = {"Train": (X_train, y_train), "Validation": (...), "Test": (...)}
        """
        # a smooth range of x values covering ALL sets, used to draw the curve
        all_x = pd.concat([X[feature] for X, _ in datasets.values()])
        x_line = pd.DataFrame({feature: np.linspace(all_x.min(), all_x.max(), 200)})
        y_line = model.predict(x_line)

        colors = {"Train": "tab:blue", "Validation": "tab:orange", "Test": "tab:green"}
        fig, axes = plt.subplots(1, len(datasets), figsize=(18, 5), sharey=True)
        for ax, (name, (X, y)) in zip(axes, datasets.items()):
            ax.scatter(X[feature], y, alpha=0.6, s=15, color=colors.get(name, "gray"),
                       label=f"{name} data")
            ax.plot(x_line[feature], y_line, color="red", linewidth=2, label="Model fit")
            ax.set_title(f"{name} set")
            ax.set_xlabel(feature)
            ax.legend()
        axes[0].set_ylabel("Disease progression")
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, title: str = "Confusion matrix"):
        """Show TN / FP / FN / TP as a small coloured grid."""
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(cm, cmap="Blues")
        labels = ["Low risk (0)", "High risk (1)"]
        ax.set_xticks([0, 1], labels)
        ax.set_yticks([0, 1], labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_title(title)
        plt.tight_layout()
        plt.show()
        return cm
