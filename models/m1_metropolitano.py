"""
m1_metropolitano.py — Modelo ETA/Regularidad para RA, RB, RC, RD + COSAC
=========================================================================
Data: Arribo S22 (delay por paradero), cosac_detalle (demanda por estación/hora)
Delay típico: RA=-130s, RB=-96s, RC=-105s (ligeramente adelantados)
Features especiales: CONTEO_HORA (validaciones como proxy de demanda)
"""

import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from models.base_model import BaseATUModel
from utils.service_registry import HISTORICAL_ETA, SPEED_BY_MODEL, get_period

logger = logging.getLogger(__name__)

CLASES = ["MUY_ADELANTADO", "ADELANTADO", "REGULAR", "RETRASADO", "ANOMALIA"]


class M1MetropolitanoModel(BaseATUModel):
    """
    ETA + Regularidad para el Metropolitano Troncal (RA/RB/RC/RD).
    Corredor exclusivo con alta frecuencia y datos de validaciones ricos.
    """
    MODEL_KEY = "M1_METROPOLITANO"

    FEATURES_ETA = [
        "HORA_SIN", "HORA_COS", "DOW_SIN", "DOW_COS",
        "IS_WEEKEND", "IS_PEAK_AM", "IS_PEAK_PM",
        "VELOCIDAD_MEDIA", "HEADWAY_MIN", "DIST_PARADERO_KM",
        "N_BUSES_EN_RUTA", "MODELO_NUM",
        "CONTEO_HORA",   # validaciones cosac por hora → proxy demanda
    ]
    FEATURES_REG = [
        "HORA_SIN", "HORA_COS", "DOW_SIN", "DOW_COS",
        "IS_WEEKEND", "IS_PEAK_AM", "IS_PEAK_PM",
        "HEADWAY_MIN", "DELAY_MIN", "DELAY_MIN_LAG1",
        "VELOCIDAD_MEDIA", "MODELO_NUM",
        "CONTEO_HORA",
    ]
    FEATURES_OD = [
        "HORA_SIN", "HORA_COS", "DOW_SIN", "DOW_COS",
        "IS_WEEKEND", "IS_PEAK_AM", "IS_PEAK_PM",
        "DIST_HAVERSINE_KM", "DIST_MANHATTAN_KM",
        "N_PARADAS", "VELOCIDAD_MEDIA", "MODELO_NUM",
    ]

    def __init__(self):
        super().__init__()
        self.feature_cols = self.FEATURES_ETA
        self.eta_model    = None
        self.reg_model    = None
        self.od_model     = None
        self.le           = None
        self.eta_mae      = 99.0
        self.od_mae       = 99.0
        self.route_means  = {}
        self.od_means     = {}
        self.demand_by_hour: dict = {}   # {hora: conteo_medio}

    # ── Carga demanda cosac ───────────────────────────────────────────────────
    def load_demand(self, cosac_df: pd.DataFrame):
        """Precalcula demanda media por hora desde cosac_detalle."""
        agg = cosac_df.groupby("hora")["conteo"].mean().to_dict()
        self.demand_by_hour = {int(k): float(v) for k, v in agg.items()}
        logger.info(f"M1: demanda cargada para {len(self.demand_by_hour)} horas")

    def _get_demand(self, hora_dec: float) -> float:
        hora_int = int(hora_dec) % 24
        return self.demand_by_hour.get(hora_int, 100.0)

    # ── Fit ───────────────────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame):
        # Inyectar CONTEO_HORA si no está
        if "CONTEO_HORA" not in df.columns:
            df = df.copy()
            df["CONTEO_HORA"] = df["HORA_DEC"].apply(self._get_demand)

        sub = df.dropna(subset=["ETA_MIN"])
        if len(sub) < 50:
            logger.warning("M1: datos insuficientes para ETA. Solo fallback.")
        else:
            Xe = sub[self.FEATURES_ETA].fillna(sub[self.FEATURES_ETA].median())
            ye = sub["ETA_MIN"]
            Xtr, Xte, ytr, yte = train_test_split(Xe, ye, test_size=0.2, random_state=42)
            self.eta_model = GradientBoostingRegressor(
                n_estimators=300, learning_rate=0.05, max_depth=5,
                subsample=0.8, random_state=42)
            self.eta_model.fit(Xtr, ytr)
            self.eta_mae = mean_absolute_error(yte, self.eta_model.predict(Xte))
            logger.info(f"M1 ETA MAE: {self.eta_mae:.2f} min | n={len(sub)}")

        # OD
        sub_od = df.dropna(subset=["TIEMPO_OD_MIN"])
        if len(sub_od) >= 50:
            Xo = sub_od[self.FEATURES_OD].fillna(0)
            yo = sub_od["TIEMPO_OD_MIN"]
            Xtr2, Xte2, ytr2, yte2 = train_test_split(Xo, yo, test_size=0.2, random_state=42)
            from sklearn.ensemble import RandomForestRegressor
            self.od_model = RandomForestRegressor(
                n_estimators=200, max_depth=10, min_samples_leaf=5,
                random_state=42, n_jobs=-1)
            self.od_model.fit(Xtr2, ytr2)
            self.od_mae = mean_absolute_error(yte2, self.od_model.predict(Xte2))
            logger.info(f"M1 OD MAE: {self.od_mae:.2f} min")

        # Regularidad
        sub_reg = df.dropna(subset=["ESTADO_REGULARIDAD", "DELAY_MIN"])
        if len(sub_reg) >= 50:
            Xr = sub_reg[self.FEATURES_REG].fillna(0)
            self.le = LabelEncoder()
            self.le.fit(CLASES)
            yr = self.le.transform(sub_reg["ESTADO_REGULARIDAD"])
            Xtr3, Xte3, ytr3, yte3 = train_test_split(Xr, yr, test_size=0.2,
                                                        stratify=yr, random_state=42)
            self.reg_model = GradientBoostingClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
            self.reg_model.fit(Xtr3, ytr3)
            logger.info(f"M1 Regularidad:\n{classification_report(yte3, self.reg_model.predict(Xte3), target_names=self.le.classes_, zero_division=0)}")

        # Fallbacks
        if "RUTA" in df.columns:
            self.route_means = df.groupby("RUTA")["ETA_MIN"].mean().to_dict()
        if "PARADA_ORIGEN" in df.columns and "PARADA_DESTINO" in df.columns:
            valid = df[(df["PARADA_ORIGEN"] != "") & (df["PARADA_DESTINO"] != "")]
            self.od_means = valid.groupby(["PARADA_ORIGEN", "PARADA_DESTINO"])["TIEMPO_OD_MIN"].mean().to_dict()

        self.is_fitted = True

    # ── Predict ETA ──────────────────────────────────────────────────────────
    def predict(self, features: dict, ruta_id: str = "",
                orig_stop: str = "", dest_stop: str = "", **kwargs) -> dict:
        hora_dec = features.get("HORA_DEC", 8.0)
        period   = get_period(hora_dec)

        feat = {**features, "CONTEO_HORA": self._get_demand(hora_dec)}

        if self.is_fitted and self.eta_model is not None:
            try:
                X   = pd.DataFrame([feat])[self.FEATURES_ETA].fillna(0)
                eta = float(self.eta_model.predict(X)[0])
                eta = max(0.5, round(eta, 1))
                conf = "ALTA" if self.eta_mae < 3 else ("MEDIA" if self.eta_mae < 6 else "BAJA")
                return {"eta_min": eta, "confianza": conf, "fuente": "modelo", "modelo": self.MODEL_KEY}
            except Exception as e:
                logger.warning(f"M1 ETA falló: {e}")

        if ruta_id in self.route_means:
            return {"eta_min": round(self.route_means[ruta_id], 1),
                    "confianza": "BAJA", "fuente": "fallback_ruta", "modelo": self.MODEL_KEY}

        eta = HISTORICAL_ETA[self.MODEL_KEY][period]
        return {"eta_min": eta, "confianza": "BAJA", "fuente": "fallback_historico", "modelo": self.MODEL_KEY}

    def predict_regularity(self, features: dict, delay_min: float = None) -> dict:
        hora_dec = features.get("HORA_DEC", 8.0)
        feat = {**features, "CONTEO_HORA": self._get_demand(hora_dec)}

        if self.is_fitted and self.reg_model is not None and self.le is not None:
            try:
                X = pd.DataFrame([feat])[self.FEATURES_REG].fillna(0)
                probs_arr = self.reg_model.predict_proba(X)[0]
                probs     = {c: round(float(p), 3) for c, p in zip(self.le.classes_, probs_arr)}
                estado    = self.le.classes_[np.argmax(probs_arr)]
                conf      = "ALTA" if max(probs_arr) > 0.70 else ("MEDIA" if max(probs_arr) > 0.50 else "BAJA")
                return {"estado": estado, "probabilidades": probs, "confianza": conf, "fuente": "modelo"}
            except Exception as e:
                logger.warning(f"M1 Reg falló: {e}")

        if delay_min is not None:
            return _regla_delay(delay_min)

        return {"estado": "REGULAR", "probabilidades": {c: 0.2 for c in CLASES},
                "confianza": "BAJA", "fuente": "fallback_prior"}

    def predict_od(self, features: dict, orig_stop: str = "", dest_stop: str = "") -> dict:
        dist_km = features.get("DIST_HAVERSINE_KM", 0)
        hora_dec = features.get("HORA_DEC", 8.0)
        period = get_period(hora_dec)

        if self.is_fitted and self.od_model is not None:
            try:
                X = pd.DataFrame([features])[self.FEATURES_OD].fillna(0)
                t = float(self.od_model.predict(X)[0])
                return {"tiempo_od_min": round(max(1, t), 1),
                        "confianza": "ALTA" if self.od_mae < 5 else "MEDIA",
                        "fuente": "modelo"}
            except Exception as e:
                logger.warning(f"M1 OD falló: {e}")

        key = (orig_stop, dest_stop)
        if orig_stop and dest_stop and key in self.od_means:
            t = self.od_means[key]
            return {"tiempo_od_min": round(t, 1), "confianza": "BAJA", "fuente": "fallback_od"}

        speed = SPEED_BY_MODEL[self.MODEL_KEY][period]
        t = round((dist_km / speed) * 60, 1) if dist_km > 0 else 20.0
        return {"tiempo_od_min": t, "confianza": "BAJA", "fuente": "fallback_velocidad"}


def _regla_delay(delay_min: float) -> dict:
    if delay_min < -5:   estado = "MUY_ADELANTADO"
    elif delay_min < -2: estado = "ADELANTADO"
    elif delay_min <= 2: estado = "REGULAR"
    elif delay_min <= 5: estado = "RETRASADO"
    else:                estado = "ANOMALIA"
    n    = len(CLASES)
    rest = round((1.0 - 0.8) / (n - 1), 4)
    probs = {c: (0.8 if c == estado else rest) for c in CLASES}
    return {"estado": estado, "probabilidades": probs, "confianza": "MEDIA", "fuente": "fallback_regla"}
