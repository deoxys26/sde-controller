"""Chronological scaling utilities for synthetic graph telemetry."""

from __future__ import annotations

import numpy as np


class FeatureScaler:
    """Small serializable standard scaler fitted exclusively on training rows."""

    def fit(self, values):
        values = np.asarray(values, dtype=np.float32)
        self.mean_ = values.mean(axis=0)
        self.scale_ = values.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, values):
        return (np.asarray(values, dtype=np.float32) - self.mean_) / self.scale_

    def inverse_transform(self, values):
        return np.asarray(values, dtype=np.float32) * self.scale_ + self.mean_
