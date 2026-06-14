"""
m2_to_m7.py — Modelos M2 al M7
================================
M2_EXPRESO       : EXP1-EXP_13   (alta varianza de delay entre líneas)
M3_SUPEREXPRESO  : SX, SXN       (más puntuales del sistema, baja varianza)
M4_ALIMENTADORA  : AN_xx, AS_xx  (rutas radiales, alta varianza local)
M5_LECHUCERO     : SN, nocturno  (sin PO horario → modelo de velocidad)
M6_CORREDOR      : 201-412       (corredores complementarios, por eje EO/NS)
M7_METRO         : L1, L2        (ETA casi determinista, foco en demanda)
"""

import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import (GradientBoostingRegressor,
                               GradientBoostingClassifier,
                               RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from models.base_model import BaseATUModel
from models.m1_metropolitano import _regla_delay, CLASES
from utils.service_registry import HISTORICAL_ETA, SPEED_BY_MODEL, get_period

logger = logging.getLogger(__name__)

# ─── Features base reutilizables ─────────────────────────────────────────────

FEAT_TIME = ["HORA_SIN", "HORA_COS", "DOW_SIN", "DOW_COS",
             "IS_WEEKEND", "IS_PEAK_AM", "IS_PEAK_PM"]

FEAT_ETA_BASE  = FEAT_TIME + ["VELOCIDAD_MEDIA", "HEADWAY_MIN",
                               "DIST_PARADERO_KM", "N_BUSES_EN_RUTA", "MODELO_NUM"]
FEAT_REG_BASE  = FEAT_TIME + ["HEADWAY_MIN", "DELAY_MIN", "DELAY_MIN_LAG1",
                               "VELOCIDAD_MEDIA", "MODELO_NUM"]
FEAT_OD_BASE   = FEAT_TIME + ["DIST_HAVERSINE_KM", "DIST_MANHATTAN_KM",
                               "N_PARADAS", "VELOCIDAD_MEDIA", "MODELO_NUM"]


def _fit_eta(df, feat_cols, target="ETA_MIN", label=""):
    sub = df.dropna(subset=[target])
    if len(sub) < 50:
        logger.warning(f"{label}: insuficientes datos ETA ({len(sub)}).")
        return None, 99.0
    X = sub[feat_cols].copy()
    for col in feat_cols:
        if col not in X.columns: X[col] = 0
    X = X.fillna(0)
    y = sub[target]
    Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.2, random_state=42)
    m = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                   max_depth=5, subsample=0.8, random_state=42)
    m.fit(Xt, yt)
    mae = mean_absolute_error(ye, m.predict(Xe))
    logger.info(f"{label} ETA MAE={mae:.2f} min | n={len(sub)}")
    return m, mae


def _fit_od(df, feat_cols, target="TIEMPO_OD_MIN", label=""):
    sub = df.dropna(subset=[target])
    if len(sub) < 50:
        return None, 99.0
    X = sub[feat_cols].fillna(0)
    y = sub[target]
    Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.2, random_state=42)
    # Cuantiles independientes
    Xt2, _, yt2, _ = train_test_split(Xt, yt, test_size=0.3, random_state=99)
    m  = RandomForestRegressor(n_estimators=200, max_depth=10,
                                min_samples_leaf=5, random_state=42, n_jobs=-1)
    ql = GradientBoostingRegressor(loss="quantile", alpha=0.10, n_estimators=100, random_state=42)
    qh = GradientBoostingRegressor(loss="quantile", alpha=0.90, n_estimators=100, random_state=42)
    m.fit(Xt, yt); ql.fit(Xt2, yt2); qh.fit(Xt2, yt2)
    mae = mean_absolute_error(ye, m.predict(Xe))
    logger.info(f"{label} OD MAE={mae:.2f} min")
    return (m, ql, qh), mae


def _fit_reg(df, feat_cols, label=""):
    sub = df.dropna(subset=["ESTADO_REGULARIDAD", "DELAY_MIN"])
    if len(sub) < 50:
        return None, None
    le = LabelEncoder(); le.fit(CLASES)
    X  = sub[feat_cols].fillna(0)
    y  = le.transform(sub["ESTADO_REGULARIDAD"])
    Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    m = GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                    learning_rate=0.05, random_state=42)
    m.fit(Xt, yt)
    logger.info(f"{label} Reg:\n{classification_report(ye, m.predict(Xe), target_names=le.classes_, zero_division=0)}")
    return m, le


def _pred_eta(model, mae, feat_cols, features, model_key, ruta_id="", route_means={}):
    period = get_period(features.get("HORA_DEC", 8.0))
    if model is not None:
        try:
            X   = pd.DataFrame([features])[feat_cols].fillna(0)
            eta = max(0.5, round(float(model.predict(X)[0]), 1))
            conf = "ALTA" if mae < 3 else ("MEDIA" if mae < 6 else "BAJA")
            return {"eta_min": eta, "confianza": conf, "fuente": "modelo", "modelo": model_key}
        except Exception as e:
            logger.warning(f"{model_key} ETA err: {e}")
    if ruta_id in route_means:
        return {"eta_min": round(route_means[ruta_id], 1), "confianza": "BAJA",
                "fuente": "fallback_ruta", "modelo": model_key}
    eta = HISTORICAL_ETA[model_key][period]
    return {"eta_min": eta, "confianza": "BAJA", "fuente": "fallback_historico", "modelo": model_key}


def _pred_od(models, mae, feat_cols, features, model_key, orig_stop="", dest_stop="", od_means={}):
    dist_km = features.get("DIST_HAVERSINE_KM", 0)
    period  = get_period(features.get("HORA_DEC", 8.0))
    if models is not None:
        try:
            m, ql, qh = models
            X  = pd.DataFrame([features])[feat_cols].fillna(0)
            t  = max(1, float(m.predict(X)[0]))
            lo = max(1, float(ql.predict(X)[0]))
            hi = max(1, float(qh.predict(X)[0]))
            conf = "ALTA" if mae < 5 else ("MEDIA" if mae < 10 else "BAJA")
            return {"tiempo_od_min": round(t, 1), "rango_min": [round(lo, 1), round(hi, 1)],
                    "confianza": conf, "fuente": "modelo"}
        except Exception as e:
            logger.warning(f"{model_key} OD err: {e}")
    key = (orig_stop, dest_stop)
    if orig_stop and dest_stop and key in od_means:
        t = od_means[key]
        return {"tiempo_od_min": round(t, 1), "rango_min": [round(t*.8, 1), round(t*1.2, 1)],
                "confianza": "BAJA", "fuente": "fallback_od"}
    speed = SPEED_BY_MODEL[model_key][period]
    t = round((dist_km / speed) * 60, 1) if dist_km > 0 and speed > 0 else 20.0
    return {"tiempo_od_min": t, "rango_min": [round(t*.7, 1), round(t*1.4, 1)],
            "confianza": "BAJA", "fuente": "fallback_velocidad"}


def _pred_reg(model, le, feat_cols, features, delay_min=None):
    if model is not None and le is not None:
        try:
            X = pd.DataFrame([features])[feat_cols].fillna(0)
            pa = model.predict_proba(X)[0]
            probs = {c: round(float(p), 3) for c, p in zip(le.classes_, pa)}
            estado = le.classes_[np.argmax(pa)]
            conf = "ALTA" if max(pa) > 0.70 else ("MEDIA" if max(pa) > 0.50 else "BAJA")
            return {"estado": estado, "probabilidades": probs, "confianza": conf, "fuente": "modelo"}
        except Exception as e:
            logger.warning(f"Reg err: {e}")
    if delay_min is not None:
        return _regla_delay(delay_min)
    return {"estado": "REGULAR", "probabilidades": {c: 0.2 for c in CLASES},
            "confianza": "BAJA", "fuente": "fallback_prior"}


# ═══════════════════════════════════════════════════════════════════════════════
# M2 — Expresos (EXP1-EXP_13)
# ═══════════════════════════════════════════════════════════════════════════════
class M2ExpresoModel(BaseATUModel):
    """
    Alta varianza de delay entre líneas EXP.
    EXP5 median=-240s (muy adelantado), EXP_12=+456s (crónico retraso).
    Sub-modelo por línea EXP cuando hay suficientes datos.
    """
    MODEL_KEY = "M2_EXPRESO"
    FEAT_ETA = FEAT_ETA_BASE + ["SENTIDO_NUM"]
    FEAT_REG = FEAT_REG_BASE + ["SENTIDO_NUM"]
    FEAT_OD  = FEAT_OD_BASE

    def __init__(self):
        super().__init__()
        self.feature_cols = self.FEAT_ETA
        self._eta = None;  self._eta_mae = 99.0
        self._od  = None;  self._od_mae  = 99.0
        self._reg = None;  self._le      = None
        # Sub-modelos por línea EXP
        self._eta_by_line: dict = {}

    def fit(self, df: pd.DataFrame):
        df = df.copy()
        if "SENTIDO_NUM" not in df.columns:
            df["SENTIDO_NUM"] = 0

        # Sub-modelo por línea si tiene >= 100 filas
        if "RUTA" in df.columns:
            for ruta, grp in df.groupby("RUTA"):
                if len(grp) >= 100:
                    m, mae = _fit_eta(grp, self.FEAT_ETA, label=f"M2/{ruta}")
                    if m:
                        self._eta_by_line[ruta] = (m, mae)

        self._eta, self._eta_mae = _fit_eta(df, self.FEAT_ETA, label="M2")
        self._od, self._od_mae   = _fit_od(df, self.FEAT_OD, label="M2")
        self._reg, self._le      = _fit_reg(df, self.FEAT_REG, label="M2")
        self.route_means = df.groupby("RUTA")["ETA_MIN"].mean().to_dict() if "RUTA" in df.columns else {}
        if "PARADA_ORIGEN" in df.columns:
            v = df[(df["PARADA_ORIGEN"] != "") & (df["PARADA_DESTINO"] != "")]
            self.od_means = v.groupby(["PARADA_ORIGEN","PARADA_DESTINO"])["TIEMPO_OD_MIN"].mean().to_dict()
        else:
            self.od_means = {}
        self.is_fitted = True

    def predict(self, features: dict, ruta_id: str = "",
                orig_stop: str = "", dest_stop: str = "", **kwargs) -> dict:
        # Intentar sub-modelo por línea primero
        if ruta_id in self._eta_by_line:
            m, mae = self._eta_by_line[ruta_id]
            try:
                X = pd.DataFrame([features])[self.FEAT_ETA].fillna(0)
                eta = max(0.5, round(float(m.predict(X)[0]), 1))
                conf = "ALTA" if mae < 3 else "MEDIA"
                return {"eta_min": eta, "confianza": conf, "fuente": f"modelo_{ruta_id}", "modelo": self.MODEL_KEY}
            except Exception:
                pass
        return _pred_eta(self._eta, self._eta_mae, self.FEAT_ETA, features,
                         self.MODEL_KEY, ruta_id, self.route_means)

    def predict_regularity(self, features: dict, delay_min: float = None) -> dict:
        return _pred_reg(self._reg, self._le, self.FEAT_REG, features, delay_min)

    def predict_od(self, features: dict, orig_stop="", dest_stop="") -> dict:
        return _pred_od(self._od, self._od_mae, self.FEAT_OD, features,
                        self.MODEL_KEY, orig_stop, dest_stop, self.od_means)


# ═══════════════════════════════════════════════════════════════════════════════
# M3 — Super Expreso (SX, SXN)
# ═══════════════════════════════════════════════════════════════════════════════
class M3SuperExpresoModel(BaseATUModel):
    """
    Los más puntuales del sistema (median -32s a -38s).
    Modelo simple — poca varianza → GBM ligero.
    """
    MODEL_KEY = "M3_SUPEREXPRESO"
    FEAT_ETA = FEAT_ETA_BASE + ["SENTIDO_NUM"]
    FEAT_REG = FEAT_REG_BASE
    FEAT_OD  = FEAT_OD_BASE

    def __init__(self):
        super().__init__()
        self.feature_cols = self.FEAT_ETA
        self._eta = None; self._eta_mae = 99.0
        self._od  = None; self._od_mae  = 99.0
        self._reg = None; self._le      = None
        self.route_means = {}; self.od_means = {}

    def fit(self, df: pd.DataFrame):
        df = df.copy()
        if "SENTIDO_NUM" not in df.columns:
            df["SENTIDO_NUM"] = 0
        self._eta, self._eta_mae = _fit_eta(df, self.FEAT_ETA, label="M3")
        self._od,  self._od_mae  = _fit_od(df, self.FEAT_OD,  label="M3")
        self._reg, self._le      = _fit_reg(df, self.FEAT_REG, label="M3")
        if "RUTA" in df.columns:
            self.route_means = df.groupby("RUTA")["ETA_MIN"].mean().to_dict()
        if "PARADA_ORIGEN" in df.columns:
            v = df[(df["PARADA_ORIGEN"] != "") & (df["PARADA_DESTINO"] != "")]
            self.od_means = v.groupby(["PARADA_ORIGEN","PARADA_DESTINO"])["TIEMPO_OD_MIN"].mean().to_dict()
        self.is_fitted = True

    def predict(self, features: dict, ruta_id="", orig_stop="", dest_stop="", **kwargs) -> dict:
        return _pred_eta(self._eta, self._eta_mae, self.FEAT_ETA, features,
                         self.MODEL_KEY, ruta_id, self.route_means)

    def predict_regularity(self, features: dict, delay_min: float = None) -> dict:
        return _pred_reg(self._reg, self._le, self.FEAT_REG, features, delay_min)

    def predict_od(self, features: dict, orig_stop="", dest_stop="") -> dict:
        return _pred_od(self._od, self._od_mae, self.FEAT_OD, features,
                        self.MODEL_KEY, orig_stop, dest_stop, self.od_means)


# ═══════════════════════════════════════════════════════════════════════════════
# M4 — Alimentadoras (AN_xx, AS_xx)
# ═══════════════════════════════════════════════════════════════════════════════
class M4AlimentadoraModel(BaseATUModel):
    """
    Rutas radiales cortas. Alta varianza local (depende del tráfico del barrio).
    Sub-modelo por zona (NORTE / SUR) y prefijo de ruta (AN / AS).
    """
    MODEL_KEY = "M4_ALIMENTADORA"
    FEAT_ETA = FEAT_ETA_BASE + ["ZONA_NUM"]  # 0=NORTE 1=SUR
    FEAT_REG = FEAT_REG_BASE + ["ZONA_NUM"]
    FEAT_OD  = FEAT_OD_BASE  + ["ZONA_NUM"]

    def __init__(self):
        super().__init__()
        self.feature_cols = self.FEAT_ETA
        self._eta_norte = None; self._mae_norte = 99.0
        self._eta_sur   = None; self._mae_sur   = 99.0
        self._od  = None; self._od_mae = 99.0
        self._reg = None; self._le     = None
        self.route_means = {}; self.od_means = {}

    def _inject_zona(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "ZONA_NUM" not in df.columns:
            if "RUTA" in df.columns:
                df["ZONA_NUM"] = df["RUTA"].apply(
                    lambda r: 1 if str(r).upper().startswith("AS") else 0)
            else:
                df["ZONA_NUM"] = 0
        return df

    def fit(self, df: pd.DataFrame):
        df = self._inject_zona(df)
        norte = df[df["ZONA_NUM"] == 0]
        sur   = df[df["ZONA_NUM"] == 1]
        self._eta_norte, self._mae_norte = _fit_eta(norte, self.FEAT_ETA, label="M4/Norte")
        self._eta_sur,   self._mae_sur   = _fit_eta(sur,   self.FEAT_ETA, label="M4/Sur")
        self._od,  self._od_mae  = _fit_od(df, self.FEAT_OD,  label="M4")
        self._reg, self._le      = _fit_reg(df, self.FEAT_REG, label="M4")
        if "RUTA" in df.columns:
            self.route_means = df.groupby("RUTA")["ETA_MIN"].mean().to_dict()
        if "PARADA_ORIGEN" in df.columns:
            v = df[(df["PARADA_ORIGEN"] != "") & (df["PARADA_DESTINO"] != "")]
            self.od_means = v.groupby(["PARADA_ORIGEN","PARADA_DESTINO"])["TIEMPO_OD_MIN"].mean().to_dict()
        self.is_fitted = True

    def predict(self, features: dict, ruta_id="", orig_stop="", dest_stop="", **kwargs) -> dict:
        zona = features.get("ZONA_NUM", 0)
        model = self._eta_sur   if zona == 1 else self._eta_norte
        mae   = self._mae_sur   if zona == 1 else self._mae_norte
        return _pred_eta(model, mae, self.FEAT_ETA, features,
                         self.MODEL_KEY, ruta_id, self.route_means)

    def predict_regularity(self, features: dict, delay_min: float = None) -> dict:
        return _pred_reg(self._reg, self._le, self.FEAT_REG, features, delay_min)

    def predict_od(self, features: dict, orig_stop="", dest_stop="") -> dict:
        return _pred_od(self._od, self._od_mae, self.FEAT_OD, features,
                        self.MODEL_KEY, orig_stop, dest_stop, self.od_means)


# ═══════════════════════════════════════════════════════════════════════════════
# M5 — Lechucero / Servicio Nocturno
# ═══════════════════════════════════════════════════════════════════════════════
class M5LechuceroModel(BaseATUModel):
    """
    Sin PO horario → sin delay calculable.
    Modelo de velocidad GPS + hora nocturna + día semana.
    No tiene modelo de regularidad (sin programado de referencia).
    Alerta especial: "sin bus hace N minutos" basada en historial GPS nocturno.
    """
    MODEL_KEY = "M5_LECHUCERO"

    # Features más simples — velocidad domina sin congestión
    FEAT_ETA = [
        "HORA_SIN", "HORA_COS", "DOW_SIN", "DOW_COS",
        "IS_WEEKEND", "IS_NOCTURNO",
        "VELOCIDAD_MEDIA", "DIST_PARADERO_KM",
        "MODELO_NUM",
    ]
    FEAT_OD = [
        "HORA_SIN", "HORA_COS", "IS_WEEKEND", "IS_NOCTURNO",
        "VELOCIDAD_MEDIA", "DIST_HAVERSINE_KM", "MODELO_NUM",
    ]

    def __init__(self):
        super().__init__()
        self.feature_cols = self.FEAT_ETA
        self._eta = None; self._eta_mae = 99.0
        self._od  = None; self._od_mae  = 99.0
        # Historial GPS nocturno para alerta "sin bus"
        self.gap_p95_by_dow: dict = {}   # {dow: gap_minutos_p95}
        self.route_means = {}

    def fit(self, df: pd.DataFrame):
        if "IS_NOCTURNO" not in df.columns:
            df = df.copy()
            df["IS_NOCTURNO"] = ((df["HORA_DEC"] >= 22.0) | (df["HORA_DEC"] <= 5.0)).astype(int)

        # Modelo lineal simple para ETA nocturno (poca data, poca complejidad)
        sub = df.dropna(subset=["ETA_MIN"])
        if len(sub) >= 30:
            X = sub[self.FEAT_ETA].fillna(sub[self.FEAT_ETA].median())
            y = sub["ETA_MIN"]
            Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.2, random_state=42)
            self._eta = LinearRegression()
            self._eta.fit(Xt, yt)
            self._eta_mae = mean_absolute_error(ye, self._eta.predict(Xe))
            logger.info(f"M5 ETA MAE={self._eta_mae:.2f} min | n={len(sub)}")

        # OD también lineal
        sub_od = df.dropna(subset=["TIEMPO_OD_MIN"])
        if len(sub_od) >= 30:
            Xo = sub_od[self.FEAT_OD].fillna(0)
            yo = sub_od["TIEMPO_OD_MIN"]
            Xt2, Xe2, yt2, ye2 = train_test_split(Xo, yo, test_size=0.2, random_state=42)
            self._od = LinearRegression()
            self._od.fit(Xt2, yt2)
            self._od_mae = mean_absolute_error(ye2, self._od.predict(Xe2))

        # Gap histórico por día de semana (para alerta "sin bus")
        if "HEADWAY_MIN" in df.columns and "DOW" in df.columns:
            for dow, grp in df.groupby("DOW"):
                gaps = grp["HEADWAY_MIN"].dropna()
                if len(gaps) >= 5:
                    self.gap_p95_by_dow[int(dow)] = float(np.percentile(gaps, 95))

        if "RUTA" in df.columns:
            self.route_means = df.groupby("RUTA")["ETA_MIN"].mean().to_dict()
        self.is_fitted = True

    def predict(self, features: dict, ruta_id="", **kwargs) -> dict:
        period = get_period(features.get("HORA_DEC", 0.0))
        if self._eta is not None:
            try:
                X   = pd.DataFrame([features])[self.FEAT_ETA].fillna(0)
                eta = max(1.0, round(float(self._eta.predict(X)[0]), 1))
                conf = "MEDIA" if self._eta_mae < 8 else "BAJA"
                return {"eta_min": eta, "confianza": conf, "fuente": "modelo_lineal",
                        "modelo": self.MODEL_KEY}
            except Exception as e:
                logger.warning(f"M5 ETA err: {e}")
        if ruta_id in self.route_means:
            return {"eta_min": round(self.route_means[ruta_id], 1), "confianza": "BAJA",
                    "fuente": "fallback_ruta", "modelo": self.MODEL_KEY}
        eta = HISTORICAL_ETA[self.MODEL_KEY].get(period, 20.0)
        return {"eta_min": eta, "confianza": "BAJA", "fuente": "fallback_historico",
                "modelo": self.MODEL_KEY}

    def predict_regularity(self, features: dict, delay_min: float = None) -> dict:
        """M5 no tiene regularidad vs PO. Retorna estado informativo."""
        return {
            "estado": "SIN_PO",
            "probabilidades": {},
            "confianza": "N/A",
            "fuente": "no_aplica",
            "nota": "Lechucero opera sin programación horaria fija."
        }

    def predict_od(self, features: dict, **kwargs) -> dict:
        dist_km = features.get("DIST_HAVERSINE_KM", 0)
        if self._od is not None:
            try:
                X = pd.DataFrame([features])[self.FEAT_OD].fillna(0)
                t = max(1, float(self._od.predict(X)[0]))
                return {"tiempo_od_min": round(t, 1), "confianza": "MEDIA", "fuente": "modelo_lineal"}
            except Exception:
                pass
        speed = SPEED_BY_MODEL[self.MODEL_KEY]["NOCTURNO"]
        t = round((dist_km / speed) * 60, 1) if dist_km > 0 else 20.0
        return {"tiempo_od_min": t, "confianza": "BAJA", "fuente": "fallback_velocidad"}

    def alert_gap(self, dow: int, headway_actual: float) -> dict:
        """Alerta 'sin bus hace mucho tiempo' comparando con histórico nocturno."""
        p95 = self.gap_p95_by_dow.get(dow, 30.0)
        if headway_actual > p95:
            return {
                "alerta": True,
                "tipo": "SIN_BUS_NOCTURNO",
                "severidad": "ALTA" if headway_actual > p95 * 1.5 else "MEDIA",
                "mensaje": f"Sin bus hace {headway_actual:.0f} min (p95 histórico: {p95:.0f} min)",
                "headway_actual": headway_actual,
                "gap_p95": p95,
            }
        return {"alerta": False, "headway_actual": headway_actual, "gap_p95": p95}


# ═══════════════════════════════════════════════════════════════════════════════
# M6 — Corredores Complementarios (201-412)
# ═══════════════════════════════════════════════════════════════════════════════
class M6CorredorModel(BaseATUModel):
    """
    Corredores EO (este-oeste: 201,204,206,209) y NS (norte-sur: 301-412).
    La asimetría de tráfico por sentido es muy relevante en Lima.
    Sub-modelos por eje EO / NS_NORTE / NS_SUR.
    """
    MODEL_KEY = "M6_CORREDOR"
    FEAT_ETA = FEAT_ETA_BASE + ["SENTIDO_NUM", "EJE_NUM", "CONTEO_HORA"]
    FEAT_REG = FEAT_REG_BASE + ["SENTIDO_NUM", "EJE_NUM"]
    FEAT_OD  = FEAT_OD_BASE  + ["EJE_NUM"]

    def __init__(self):
        super().__init__()
        self.feature_cols = self.FEAT_ETA
        self._eta_by_eje: dict = {}  # {"EO": (m,mae), "NS_NORTE":..., "NS_SUR":...}
        self._eta     = None; self._eta_mae = 99.0
        self._od      = None; self._od_mae  = 99.0
        self._reg     = None; self._le      = None
        self.demand_by_hour: dict = {}
        self.route_means = {}; self.od_means = {}

    EJE_ENCODING = {"EO": 0, "NS_NORTE": 1, "NS_SUR": 2}

    def load_demand(self, corredores_df: pd.DataFrame):
        """Carga conteo de validaciones por hora desde corredoresC."""
        agg = corredores_df.groupby("hora")["conteo"].mean().to_dict()
        self.demand_by_hour = {int(k): float(v) for k, v in agg.items()}

    def _get_demand(self, hora_dec: float) -> float:
        return self.demand_by_hour.get(int(hora_dec) % 24, 50.0)

    def _inject(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "EJE_NUM" not in df.columns:
            df["EJE_NUM"] = 0
        if "SENTIDO_NUM" not in df.columns:
            df["SENTIDO_NUM"] = 0
        if "CONTEO_HORA" not in df.columns:
            df["CONTEO_HORA"] = df["HORA_DEC"].apply(self._get_demand)
        # Ensure all feature cols exist to avoid NaN on predict
        for col in self.FEAT_ETA + self.FEAT_REG + self.FEAT_OD:
            if col not in df.columns:
                df[col] = 0
        return df

    def fit(self, df: pd.DataFrame):
        df = self._inject(df)
        # Sub-modelos por eje
        for eje, grp in df.groupby("EJE_NUM"):
            eje_name = {v: k for k, v in self.EJE_ENCODING.items()}.get(eje, "EO")
            m, mae = _fit_eta(grp, self.FEAT_ETA, label=f"M6/{eje_name}")
            if m:
                self._eta_by_eje[eje] = (m, mae)

        self._eta, self._eta_mae = _fit_eta(df, self.FEAT_ETA, label="M6")
        self._od,  self._od_mae  = _fit_od(df, self.FEAT_OD,   label="M6")
        self._reg, self._le      = _fit_reg(df, self.FEAT_REG,  label="M6")
        if "RUTA" in df.columns:
            self.route_means = df.groupby("RUTA")["ETA_MIN"].mean().to_dict()
        if "PARADA_ORIGEN" in df.columns:
            v = df[(df["PARADA_ORIGEN"] != "") & (df["PARADA_DESTINO"] != "")]
            self.od_means = v.groupby(["PARADA_ORIGEN","PARADA_DESTINO"])["TIEMPO_OD_MIN"].mean().to_dict()
        self.is_fitted = True

    def predict(self, features: dict, ruta_id="", orig_stop="", dest_stop="", **kwargs) -> dict:
        features = {**features, "CONTEO_HORA": self._get_demand(features.get("HORA_DEC", 8.0))}
        eje_num  = features.get("EJE_NUM", 0)
        if eje_num in self._eta_by_eje:
            m, mae = self._eta_by_eje[eje_num]
            return _pred_eta(m, mae, self.FEAT_ETA, features, self.MODEL_KEY, ruta_id, self.route_means)
        return _pred_eta(self._eta, self._eta_mae, self.FEAT_ETA, features,
                         self.MODEL_KEY, ruta_id, self.route_means)

    def predict_regularity(self, features: dict, delay_min: float = None) -> dict:
        return _pred_reg(self._reg, self._le, self.FEAT_REG, features, delay_min)

    def predict_od(self, features: dict, orig_stop="", dest_stop="") -> dict:
        return _pred_od(self._od, self._od_mae, self.FEAT_OD, features,
                        self.MODEL_KEY, orig_stop, dest_stop, self.od_means)


# ═══════════════════════════════════════════════════════════════════════════════
# M7 — Metro L1 / L2
# ═══════════════════════════════════════════════════════════════════════════════
class M7MetroModel(BaseATUModel):
    """
    Infraestructura fija → ETA casi determinista (distancia / velocidad constante).
    Foco en demanda por estación/hora para alertas de crowding.
    No aplica delay vs PO de Arribo (metro tiene sistema propio).
    """
    MODEL_KEY = "M7_METRO"
    VELOCIDAD_FIJA = 35.0   # km/h constante en metro
    FEAT_ETA = [
        "HORA_SIN", "HORA_COS", "DOW_SIN", "DOW_COS",
        "IS_WEEKEND", "IS_PEAK_AM", "IS_PEAK_PM",
        "DIST_PARADERO_KM", "N_ESTACIONES",
        "CONTEO_HORA", "MODELO_NUM",
    ]
    FEAT_OD = [
        "HORA_SIN", "HORA_COS", "IS_WEEKEND", "IS_PEAK_AM", "IS_PEAK_PM",
        "DIST_HAVERSINE_KM", "N_ESTACIONES", "CONTEO_HORA", "MODELO_NUM",
    ]

    def __init__(self):
        super().__init__()
        self.feature_cols = self.FEAT_ETA
        self._eta = None; self._eta_mae = 99.0
        self._od  = None; self._od_mae  = 99.0
        self.demand_by_hour: dict = {}
        self.demand_by_station: dict = {}   # {desc_estacion: {hora: conteo_mean}}
        self.n_estaciones_l1 = 26
        self.n_estaciones_l2 = 27   # L2 en construcción, estimado

    def load_demand(self, metro_df: pd.DataFrame):
        """Carga demanda desde linea1_detalle.csv."""
        agg = metro_df.groupby("hora")["conteo"].mean().to_dict()
        self.demand_by_hour = {int(k): float(v) for k, v in agg.items()}
        if "desc_estacion" in metro_df.columns:
            for est, grp in metro_df.groupby("desc_estacion"):
                self.demand_by_station[est] = grp.groupby("hora")["conteo"].mean().to_dict()
        logger.info(f"M7: demanda cargada {len(self.demand_by_hour)} horas, {len(self.demand_by_station)} estaciones")

    def _get_demand(self, hora_dec: float, estacion: str = "") -> float:
        h = int(hora_dec) % 24
        if estacion and estacion in self.demand_by_station:
            return self.demand_by_station[estacion].get(h, 100.0)
        return self.demand_by_hour.get(h, 100.0)

    def _inject(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "N_ESTACIONES" not in df.columns:
            df["N_ESTACIONES"] = df.get("N_PARADAS", pd.Series(10, index=df.index))
        if "CONTEO_HORA" not in df.columns:
            df["CONTEO_HORA"] = df["HORA_DEC"].apply(self._get_demand)
        return df

    def fit(self, df: pd.DataFrame):
        df = self._inject(df)
        self._eta, self._eta_mae = _fit_eta(df, self.FEAT_ETA, label="M7")
        self._od,  self._od_mae  = _fit_od(df, self.FEAT_OD,   label="M7")
        self.is_fitted = True

    def predict(self, features: dict, ruta_id="", estacion: str = "", **kwargs) -> dict:
        features = {
            **features,
            "CONTEO_HORA": self._get_demand(features.get("HORA_DEC", 8.0), estacion),
            "N_ESTACIONES": features.get("N_ESTACIONES", 10),
        }
        # ETA determinista metro: distancia / velocidad_fija
        dist_km = features.get("DIST_PARADERO_KM", 1.0)
        eta_det = round((dist_km / self.VELOCIDAD_FIJA) * 60, 1)

        if self._eta is not None:
            try:
                X   = pd.DataFrame([features])[self.FEAT_ETA].fillna(0)
                eta = max(0.3, round(float(self._eta.predict(X)[0]), 1))
                conf = "ALTA" if self._eta_mae < 2 else "MEDIA"
                return {"eta_min": eta, "confianza": conf, "fuente": "modelo",
                        "eta_deterministico": eta_det, "modelo": self.MODEL_KEY}
            except Exception as e:
                logger.warning(f"M7 ETA err: {e}")

        return {"eta_min": eta_det, "confianza": "ALTA",
                "fuente": "deterministico", "modelo": self.MODEL_KEY}

    def predict_regularity(self, features: dict, delay_min: float = None) -> dict:
        """Metro tiene sistema de control propio; regularidad es muy alta."""
        return {
            "estado": "REGULAR",
            "probabilidades": {"REGULAR": 0.90, "RETRASADO": 0.06,
                               "ANOMALIA": 0.02, "ADELANTADO": 0.01, "MUY_ADELANTADO": 0.01},
            "confianza": "ALTA",
            "fuente": "prior_metro",
            "nota": "Metro L1 tiene sistema de control propio (puntualidad > 95%)."
        }

    def predict_od(self, features: dict, n_estaciones: int = None, estacion="", **kwargs) -> dict:
        features = {
            **features,
            "CONTEO_HORA": self._get_demand(features.get("HORA_DEC", 8.0), estacion),
            "N_ESTACIONES": n_estaciones or features.get("N_ESTACIONES", 10),
        }
        dist_km = features.get("DIST_HAVERSINE_KM", 5.0)
        t_det   = round((dist_km / self.VELOCIDAD_FIJA) * 60, 1)
        if self._od is not None:
            try:
                m, ql, qh = self._od
                X  = pd.DataFrame([features])[self.FEAT_OD].fillna(0)
                t  = max(1, float(m.predict(X)[0]))
                lo = max(1, float(ql.predict(X)[0]))
                hi = max(1, float(qh.predict(X)[0]))
                return {"tiempo_od_min": round(t, 1), "rango_min": [round(lo,1), round(hi,1)],
                        "confianza": "ALTA", "fuente": "modelo",
                        "tiempo_deterministico": t_det}
            except Exception as e:
                logger.warning(f"M7 OD err: {e}")
        return {"tiempo_od_min": t_det, "rango_min": [round(t_det*.9,1), round(t_det*1.1,1)],
                "confianza": "ALTA", "fuente": "deterministico"}

    def crowding_alert(self, hora_dec: float, estacion: str,
                       umbral_percentil: float = 90.0) -> dict:
        """Alerta de alta demanda en estación (crowding)."""
        conteo = self._get_demand(hora_dec, estacion)
        all_v  = [v for h in self.demand_by_station.values() for v in h.values()]
        if not all_v:
            return {"alerta": False}
        p90 = np.percentile(all_v, umbral_percentil)
        if conteo > p90:
            return {
                "alerta": True, "tipo": "CROWDING_METRO",
                "severidad": "ALTA" if conteo > p90 * 1.5 else "MEDIA",
                "mensaje": f"Alta demanda en {estacion}: {conteo:.0f} validaciones/h (p90={p90:.0f})",
                "conteo": conteo, "umbral_p90": round(p90, 0),
            }
        return {"alerta": False, "conteo": conteo, "umbral_p90": round(p90, 0)}
