"""
eda.py
------
Exploratory Data Analysis (EDA) helpers: statistics and plots.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class DataExplorer:
    """Groups all the EDA steps for one DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The data to explore (features + target).
    target : str
        Name of the target column.
    """

    def __init__(self, df: pd.DataFrame, target: str = "target"):
        self.df = df
        self.target = target
        self.features = [col for col in df.columns if col != target]

    # ---------- Statistics ----------
    def summary_statistics(self) -> pd.DataFrame:
        """count, mean, std, min, quartiles and max for every column (rounded)."""
        return self.df.describe().T.round(2)

    def data_quality_report(self) -> pd.DataFrame:
        """One row per column: data type, missing values and number of unique values."""
        return pd.DataFrame({
            "dtype": self.df.dtypes.astype(str),
            "missing_values": self.df.isna().sum(),
            "unique_values": self.df.nunique(),
        })

    def count_duplicates(self) -> int:
        """Number of rows that are exact copies of an earlier row."""
        return int(self.df.duplicated().sum())

    def target_correlations(self) -> pd.Series:
        """Correlation of each feature with the target, strongest first."""
        corr = self.df.corr()[self.target].drop(self.target)
        # sort by absolute value so strong negative correlations also appear at the top
        return corr.reindex(corr.abs().sort_values(ascending=False).index).round(2)

    def count_outliers_iqr(self) -> pd.Series:
        """Count values outside 1.5 * IQR (the usual 'box plot' outlier rule)."""
        q1 = self.df.quantile(0.25)
        q3 = self.df.quantile(0.75)
        iqr = q3 - q1
        outside = (self.df < q1 - 1.5 * iqr) | (self.df > q3 + 1.5 * iqr)
        return outside.sum()

    # ---------- Plots ----------
    def plot_histograms(self, bins: int = 20):
        """One histogram per column, to see the shape of each distribution."""
        self.df.hist(bins=bins, figsize=(14, 10), edgecolor="black", color="steelblue")
        plt.suptitle("Histograms of all columns", fontsize=16)
        plt.tight_layout()
        plt.show()

    def plot_scatter_vs_target(self):
        """Scatter plot of every feature against the target."""
        n_cols = 5
        n_rows = -(-len(self.features) // n_cols)  # ceiling division
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
        for ax, feature in zip(axes.flat, self.features):
            ax.scatter(self.df[feature], self.df[self.target], alpha=0.5, s=12)
            r = self.df[feature].corr(self.df[self.target])
            ax.set_title(f"{feature} vs {self.target} (r = {r:.2f})")
            ax.set_xlabel(feature)
            ax.set_ylabel(self.target)
        # hide any empty sub-plots
        for ax in list(axes.flat)[len(self.features):]:
            ax.set_visible(False)
        plt.tight_layout()
        plt.show()

    def plot_correlation_matrix(self):
        """Heat map of the Pearson correlation between every pair of columns."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                    vmin=-1, vmax=1, square=True)
        plt.title("Correlation matrix")
        plt.show()

    def plot_boxplots(self):
        """Box plots to spot outliers (each column on its own axis, as units differ)."""
        n_cols = 6
        n_rows = -(-len(self.df.columns) // n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
        for ax, col in zip(axes.flat, self.df.columns):
            ax.boxplot(self.df[col])
            ax.set_title(col)
            ax.set_xticks([])
        for ax in list(axes.flat)[len(self.df.columns):]:
            ax.set_visible(False)
        plt.suptitle("Box plots (points beyond the whiskers are possible outliers)")
        plt.tight_layout()
        plt.show()
