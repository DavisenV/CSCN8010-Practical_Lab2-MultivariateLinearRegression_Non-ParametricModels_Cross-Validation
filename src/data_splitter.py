"""
data_splitter.py
----------------
Splits the data into train / validation / test sets.
"""

from sklearn.model_selection import train_test_split


class DataSplitter:
    """Splits X and y into three parts: train, validation and test.

    sklearn's train_test_split can only cut data into TWO parts, so we call it twice:
        1) cut off the test set (15 %)
        2) cut the remaining 85 % into train (75 % of total) and validation (10 % of total)

    Parameters
    ----------
    val_size : float
        Fraction of ALL data used for validation (0.10).
    test_size : float
        Fraction of ALL data used for testing (0.15).
    random_state : int
        Fixes the random shuffle so we get the same split every time we run the notebook.
    """

    def __init__(self, val_size: float = 0.10, test_size: float = 0.15, random_state: int = 42):
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state

    def split(self, X, y):
        """Return X_train, X_val, X_test, y_train, y_val, y_test."""
        # Step 1: separate the test set (15 % of all rows).
        X_rest, X_test, y_rest, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        # Step 2: the validation set must be 10 % of ALL rows,
        # but we now split only the remaining 85 %, so: 0.10 / 0.85 ≈ 0.1176
        val_fraction_of_rest = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_rest, y_rest, test_size=val_fraction_of_rest, random_state=self.random_state
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    @staticmethod
    def describe_split(X_train, X_val, X_test):
        """Print how many rows ended up in each set (and the percentage)."""
        total = len(X_train) + len(X_val) + len(X_test)
        for name, part in [("Train", X_train), ("Validation", X_val), ("Test", X_test)]:
            print(f"{name:<11}: {len(part):>3} rows ({len(part) / total:.1%})")
