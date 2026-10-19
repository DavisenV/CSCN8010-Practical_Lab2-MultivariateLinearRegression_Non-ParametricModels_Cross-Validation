"""
model_builder.py
----------------
Creates the different models as sklearn Pipelines, and explains their parameters.

A Pipeline chains steps together, e.g. "scale the data -> fit the model".
When we call pipeline.fit(X, y) every step is run in order. This guarantees that the
same preparation is applied to train, validation and test data (and prevents mistakes).
"""

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor


class ModelBuilder:
    """Factory class: each method returns a new, *unfitted* model."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    # ---------- Models ----------
    @staticmethod
    def univariate_polynomial(degree: int) -> Pipeline:
        """Polynomial regression on ONE feature: y = b0 + b1*x + b2*x^2 + ... + bd*x^d.

        PolynomialFeatures(include_bias=True) creates the columns [1, x, x^2, ..., x^d].
        The column of 1s plays the role of the intercept b0, so we switch off
        LinearRegression's own intercept (fit_intercept=False) to avoid fitting it twice.
        This also makes degree 0 possible: the only column is "1", so the model
        simply predicts a constant (the mean of y).
        """
        return Pipeline([
            ("poly", PolynomialFeatures(degree=degree, include_bias=True)),
            ("linreg", LinearRegression(fit_intercept=False)),
        ])

    @staticmethod
    def multivariate_polynomial(degree: int) -> Pipeline:
        """Polynomial regression on MANY features (including interaction terms like bmi*bp).

        We standardise first (mean 0, std 1) because features have very different units
        (cholesterol ~190 vs s5 ~4.6). Raising big numbers to powers makes huge values
        and numerical problems; scaling keeps them in a similar range.
        """
        return Pipeline([
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("linreg", LinearRegression()),
        ])

    def decision_tree(self, max_depth: int) -> Pipeline:
        """Decision tree regressor. max_depth limits how many questions the tree may ask.

        Trees split on thresholds ("is bmi > 27?"), so scaling is NOT needed.
        """
        return Pipeline([
            ("tree", DecisionTreeRegressor(max_depth=max_depth, random_state=self.random_state)),
        ])

    @staticmethod
    def knn(n_neighbors: int) -> Pipeline:
        """k-Nearest Neighbours regressor: predicts the average target of the k closest patients.

        kNN measures DISTANCE, so scaling is essential. Without it, a feature with large
        numbers (s1, cholesterol ~190) would dominate one with small numbers (s5 ~4.6).
        """
        return Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsRegressor(n_neighbors=n_neighbors)),
        ])

    def logistic_regression(self, C: float) -> Pipeline:
        """Logistic regression CLASSIFIER (predicts a class: high risk 1 / low risk 0).

        C is the inverse of regularisation strength: a small C = a stronger penalty on
        large coefficients = a simpler model.
        """
        return Pipeline([
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(C=C, max_iter=1000, random_state=self.random_state)),
        ])

    # ---------- Explaining the models ----------
    @staticmethod
    def count_trainable_parameters(model: Pipeline):
        """Return (number_of_parameters, short explanation) for a FITTED pipeline."""
        last_step = model.steps[-1][1]

        if isinstance(last_step, LinearRegression):
            n_coef = last_step.coef_.size
            if last_step.fit_intercept:
                return n_coef + 1, f"{n_coef} coefficients + 1 intercept"
            return n_coef, f"{n_coef} coefficients (the '1' column is the intercept)"

        if isinstance(last_step, LogisticRegression):
            n_coef = last_step.coef_.size
            return n_coef + 1, f"{n_coef} feature weights + 1 intercept"

        if isinstance(last_step, DecisionTreeRegressor):
            n_leaves = last_step.get_n_leaves()
            n_splits = last_step.tree_.node_count - n_leaves
            # each split learns a feature AND a threshold; each leaf learns a prediction value
            return (2 * n_splits + n_leaves,
                    f"{n_splits} splits x (feature + threshold) + {n_leaves} leaf values "
                    f"(non-parametric: learned structure, not fixed weights)")

        if isinstance(last_step, KNeighborsRegressor):
            return 0, "0 learned parameters - kNN just stores the training data (k is a hyperparameter)"

        return None, "unknown model type"

    @staticmethod
    def polynomial_equation(model: Pipeline, decimals: int = 2) -> str:
        """Write a fitted univariate polynomial pipeline as a readable equation."""
        names = model.named_steps["poly"].get_feature_names_out()   # e.g. ['1', 'bmi', 'bmi^2']
        coefs = model.named_steps["linreg"].coef_
        terms = []
        for name, coef in zip(names, coefs):
            value = f"{abs(coef):.{decimals}f}"
            term = value if name == "1" else f"{value}*{name.replace(' ', '*')}"
            sign = "-" if coef < 0 else "+"
            terms.append((sign, term))
        # first term: only show a sign if it is negative
        first_sign, first_term = terms[0]
        equation = ("-" if first_sign == "-" else "") + first_term
        for sign, term in terms[1:]:
            equation += f" {sign} {term}"
        return "progression = " + equation
