"""
base_model.py
=============
Clase base compartida por los 7 modelos ATU.
Define la interfaz: fit / predict / save / load + fallback en cascada.
"""

import pickle
import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


class BaseATUModel(ABC):
    """Interfaz común para los 7 modelos de predicción ATU."""

    MODEL_KEY: str = "BASE"

    def __init__(self):
        self.is_fitted = False
        self.models: dict = {}          # sub-modelos por sub-servicio si aplica
        self.route_means: dict = {}     # fallback fino por ruta
        self.feature_cols: list = []    # definido por cada subclase

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> None: ...

    @abstractmethod
    def predict(self, features: dict, **kwargs) -> dict: ...

    def _build_X(self, features: dict) -> pd.DataFrame:
        """Construye DataFrame de 1 fila con las feature_cols del modelo."""
        row = {col: features.get(col, 0) for col in self.feature_cols}
        return pd.DataFrame([row])

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Modelo {self.MODEL_KEY} guardado en {path}")

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"Modelo cargado desde {path}")
        return obj

    def __repr__(self):
        fitted = "fitted" if self.is_fitted else "not fitted"
        return f"<{self.__class__.__name__} [{self.MODEL_KEY}] {fitted}>"
