"""Training-only feature preparation for the UNSW-NB15 binary task."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class UNSWPreprocessor:
    """Drops only labels/identifier fields and fits transformations on train rows."""
    label_column = "label"
    category_column = "attack_cat"
    identifier_columns = ("id",)

    def __init__(self):
        self.dropped_columns = {
            "label": "binary prediction target",
            "attack_cat": "attack-category annotation; removing prevents direct label leakage",
            "id": "dataset row identifier; not a flow characteristic",
        }

    def _features(self, frame):
        required = {self.label_column, self.category_column, *self.identifier_columns}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"UNSW-NB15 columns missing: {sorted(missing)}")
        features = frame.drop(columns=[self.label_column, self.category_column, *self.identifier_columns]).copy()
        return features.replace([np.inf, -np.inf], np.nan)

    def fit(self, training_frame):
        features = self._features(training_frame)
        self.feature_columns = tuple(features.columns)
        self.categorical_columns = tuple(features.select_dtypes(include=["object", "category", "bool"]).columns)
        self.numeric_columns = tuple(column for column in self.feature_columns if column not in self.categorical_columns)
        numeric = Pipeline([( "imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
        categorical = Pipeline([( "imputer", SimpleImputer(strategy="most_frequent")),
                                ("encoder", OneHotEncoder(handle_unknown="ignore"))])
        self.transformer = ColumnTransformer([( "numeric", numeric, list(self.numeric_columns)),
                                              ("categorical", categorical, list(self.categorical_columns))])
        self.transformer.fit(features)
        return self

    def transform(self, frame):
        if not hasattr(self, "transformer"):
            raise RuntimeError("fit must be called before transform")
        transformed = self.transformer.transform(self._features(frame))
        values = transformed.data if sparse.issparse(transformed) else np.asarray(transformed)
        if not np.isfinite(values).all():
            raise ValueError("Preprocessing produced non-finite values")
        return transformed

    def labels(self, frame):
        return frame[self.label_column].to_numpy(dtype=int)
