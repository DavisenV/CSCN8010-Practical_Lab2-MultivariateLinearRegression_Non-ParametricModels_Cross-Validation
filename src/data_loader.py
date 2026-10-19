"""
data_loader.py
--------------
Loads the Scikit-Learn diabetes dataset into a pandas DataFrame.
"""

import pandas as pd
from sklearn import datasets


class DiabetesDataLoader:
    """Loads the diabetes dataset and gives easy access to features and target.

    Parameters
    ----------
    scaled : bool
        False -> original units (BMI in kg/m^2, blood pressure in mm Hg, ...).
        True  -> sklearn's pre-scaled version (every column mean-centred).
        We use False so the numbers are easy to read for a physician
        (e.g. "BMI = 30" instead of "BMI = 0.0617").
    """

    TARGET_NAME = "target"

    # Readable names for the columns (taken from the dataset description).
    FEATURE_DESCRIPTIONS = {
        "age": "Age in years",
        "sex": "Sex (1 or 2)",
        "bmi": "Body mass index (kg/m^2)",
        "bp": "Average blood pressure (mm Hg)",
        "s1": "tc - total serum cholesterol",
        "s2": "ldl - low-density lipoproteins",
        "s3": "hdl - high-density lipoproteins",
        "s4": "tch - total cholesterol / HDL",
        "s5": "ltg - log of serum triglycerides level",
        "s6": "glu - blood sugar level",
        "target": "Disease progression one year after baseline",
    }

    def __init__(self, scaled: bool = False):
        self.scaled = scaled
        self.bunch = None   # the raw object sklearn returns
        self.df = None      # features + target in one DataFrame

    def load(self) -> pd.DataFrame:
        """Load the data and return it as one DataFrame (features + target)."""
        # as_frame=True  -> we get pandas objects instead of plain numpy arrays
        # scaled=False   -> keep the original measurement units
        self.bunch = datasets.load_diabetes(as_frame=True, scaled=self.scaled)
        self.df = self.bunch.frame.copy()
        return self.df

    def get_description(self) -> str:
        """Return the official dataset description text from sklearn."""
        self._check_loaded()
        return self.bunch.DESCR

    def get_features_and_target(self):
        """Split the DataFrame into X (features) and y (target)."""
        self._check_loaded()
        X = self.df.drop(columns=[self.TARGET_NAME])
        y = self.df[self.TARGET_NAME]
        return X, y

    def feature_table(self) -> pd.DataFrame:
        """A small table that explains what each column means."""
        return pd.DataFrame(
            list(self.FEATURE_DESCRIPTIONS.items()),
            columns=["column", "meaning"],
        )

    def _check_loaded(self):
        # A tiny "guard" so we get a clear error message instead of a confusing one.
        if self.df is None:
            raise RuntimeError("Call load() before using the data.")
