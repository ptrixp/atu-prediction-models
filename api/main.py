"""
api/main.py — ATU Prediction API v2.0
=======================================
7 modelos de predicción para la red ATU de Lima.
FastAPI REST — compatible con Lovable y cualquier frontend.

Endpoints:
  POST /predict/eta          → ETA del bus en un paradero
  POST /predict/od           → Tiempo O-D
  POST /predict/regularity   → Estado de regularidad
  POST /predict/alerts       → Alertas combinadas (ETA + Regularidad)
  POST /predict/all          → 3 predicciones + alertas en una sola llamada
  POST /predict/metro/crowding → Alerta de alta demanda en estación metro
  POST /predict/lechucero/gap  → Alerta de "sin bus nocturno"
  GET  /health               → Estado de los 7 modelos
  GET  /services             → Mapa de servicios → modelo
"""

import os, sys, logging
import numpy as np
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.m1_metropolitano import M1MetropolitanoModel
from models.m2_to_m7 import (M2ExpresoModel, M3SuperExpresoModel,
                               M4AlimentadoraModel, M5LechuceroModel,
                               M6CorredorModel, M7MetroModel)
from utils.service_registry import (classify_service, MODEL_ENCODING,
                                     ETA_ALERT_THRESHOLD, STOP_SPACING_KM,
                                     get_corredor_axis, get_period)

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent / "trained"

MODEL_CLASSES = {
    "M1_METROPOLITANO": M1MetropolitanoModel,
    "M2_EXPRESO":       M2ExpresoModel,
    "M3_SUPEREXPRESO":  M3SuperExpresoModel,
    "M4_ALIMENTADORA":  M4AlimentadoraModel,
    "M5_LECHUCERO":     M5LechuceroModel,
    "M6_CORREDOR":      M6CorredorModel,
    "M7_METRO":         M7MetroModel,
}

# Diccionario global de modelos cargados
_models: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _models
    for model_key, ModelClass in MODEL_CLASSES.items():
        pkl = MODEL_DIR / f"{model_key.lower()}.pkl"
        try:
            _models[model_key] = ModelClass.load(str(pkl))
            logger.info(f"✓ {model_key} cargado")
        except Exception as e:
            logger.warning(f"✗ {model_key} no encontrado ({e}). Instancia vacía.")
            _models[model_key] = ModelClass()
    yield


ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app = FastAPI(
    title="ATU Prediction API",
    description="7 modelos de predicción de transporte público de Lima",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS,
                   allow_methods=["*"], allow_headers=["*"])


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _time_features(timestamp: Optional[str] = None) -> dict:
    dt = datetime.fromisoformat(timestamp) if timestamp else datetime.now()
    h  = dt.hour + dt.minute / 60
    d  = dt.weekday()
    return {
        "HORA_DEC":    h,
        "DOW":         d,
        "IS_WEEKEND":  int(d >= 5),
        "IS_PEAK_AM":  int(6.5 <= h <= 9.5),
        "IS_PEAK_PM":  int(17.5 <= h <= 20.0),
        "IS_NOCTURNO": int(h >= 22.0 or h <= 5.0),
        "HORA_SIN":    float(np.sin(2 * np.pi * h / 24)),
        "HORA_COS":    float(np.cos(2 * np.pi * h / 24)),
        "DOW_SIN":     float(np.sin(2 * np.pi * d / 7)),
        "DOW_COS":     float(np.cos(2 * np.pi * d / 7)),
    }


def _haversine(lat1, lon1, lat2, lon2):
    from math import radians, sin, cos, sqrt, atan2
    R = 6371.0
    p1, p2 = radians(lat1), radians(lat2)
    dp = radians(lat2 - lat1); dl = radians(lon2 - lon1)
    a  = sin(dp/2)**2 + cos(p1)*cos(p2)*sin(dl/2)**2
    hav = 2 * R * atan2(sqrt(a), sqrt(1 - a))
    mid = (lat1 + lat2) / 2
    man = abs(lat2-lat1)*111 + abs(lon2-lon1)*111*cos(radians(mid))
    return hav, man


def _get_model(model_key: str):
    return _models.get(model_key)


def _build_alerts(eta_res: dict, reg_res: dict, model_key: str,
                  route_id: str, paradero_id: str,
                  delay_min: float, headway_min: float) -> list:
    alerts = []
    estado = reg_res.get("estado", "REGULAR")
    eta    = eta_res.get("eta_min", 99)
    umbral = ETA_ALERT_THRESHOLD.get(model_key, 15)

    if estado in ("RETRASADO", "ANOMALIA") and eta > umbral:
        alerts.append({
            "tipo":      "BUS_RETRASADO",
            "severidad": "ALTA" if estado == "ANOMALIA" else "MEDIA",
            "mensaje":   f"Bus {route_id} en {paradero_id}: {delay_min:.0f} min retraso. ETA {eta} min.",
            "eta_min":   eta,
        })
    if estado == "MUY_ADELANTADO":
        p = reg_res.get("probabilidades", {}).get("MUY_ADELANTADO", 0)
        if p > 0.6:
            alerts.append({
                "tipo":      "BUS_ADELANTADO",
                "severidad": "MEDIA",
                "mensaje":   f"Bus {route_id} pasó antes de lo programado (ghost bus).",
                "probabilidad": round(p, 3),
            })
    if headway_min > umbral * 1.5 and estado in ("ANOMALIA", "RETRASADO"):
        alerts.append({
            "tipo":      "HEADWAY_EXCESIVO",
            "severidad": "ALTA",
            "mensaje":   f"Intervalo inusual: {headway_min:.0f} min entre buses.",
            "headway_min": headway_min,
        })
    return alerts


# ─── Schemas ─────────────────────────────────────────────────────────────────

class ETARequest(BaseModel):
    route_id:         str   = Field(..., example="RA")
    paradero_id:      str   = Field(..., example="NA06")
    timestamp:        Optional[str] = None
    velocidad_media:  float = Field(default=18.0, ge=0, le=120)
    headway_min:      float = Field(default=5.0,  ge=0)
    dist_paradero_km: float = Field(default=1.0,  ge=0)
    n_buses_en_ruta:  int   = Field(default=4,    ge=0)
    sentido_num:      int   = Field(default=0)      # 0=IDA/Norte/EO  1=VUELTA/Sur/OE

class ODRequest(BaseModel):
    route_id:        str   = Field(..., example="201")
    origen_lat:      float = Field(..., example=-12.0464)
    origen_lon:      float = Field(..., example=-77.0428)
    destino_lat:     float = Field(..., example=-12.1100)
    destino_lon:     float = Field(..., example=-77.0200)
    parada_origen:   Optional[str] = None
    parada_destino:  Optional[str] = None
    velocidad_media: float = Field(default=20.0, ge=1)
    timestamp:       Optional[str] = None

class RegularityRequest(BaseModel):
    route_id:        str   = Field(..., example="EXP5")
    headway_min:     float = Field(default=5.0)
    delay_min:       Optional[float] = None
    delay_min_lag1:  float = Field(default=0.0)
    velocidad_media: float = Field(default=25.0)
    timestamp:       Optional[str] = None

class AlertsRequest(BaseModel):
    route_id:         str   = Field(..., example="AN_07")
    paradero_id:      str   = Field(..., example="AVCS")
    delay_min:        float = Field(default=0.0)
    headway_min:      float = Field(default=5.0)
    velocidad_media:  float = Field(default=18.0)
    dist_paradero_km: float = Field(default=1.0)
    timestamp:        Optional[str] = None

class AllRequest(BaseModel):
    """Payload completo compatible con Lovable (incluye campos GPS SICM)."""
    imei:             Optional[int]   = None
    latitude:         float           = Field(default=-12.0464)
    longitude:        float           = Field(default=-77.0428)
    destination_lat:  float           = Field(default=-12.1100)
    destination_lon:  float           = Field(default=-77.0200)
    route_id:         str             = Field(..., example="RB")
    ts:               Optional[int]   = None
    license_plate:    str             = Field(default="")
    speed:            float           = Field(default=18.0)
    direction_id:     int             = Field(default=0)
    driver_id:        Optional[str]   = None
    tsinitialtrip:    Optional[int]   = None
    identifier:       Optional[str]   = None
    origin_stop:      str             = Field(default="")
    destination_stop: str             = Field(default="")
    target_stop:      str             = Field(default="")
    current_stop:     str             = Field(default="")
    scheduled_time:   Optional[str]   = None
    delay_min:        float           = Field(default=0.0)
    headway_min:      float           = Field(default=6.0)

class MetroCrowdingRequest(BaseModel):
    estacion:  str   = Field(..., example="Naranjal")
    hora_dec:  float = Field(..., example=8.5)

class LechuceroGapRequest(BaseModel):
    dow:             int   = Field(..., example=4, description="0=Lun … 6=Dom")
    headway_actual:  float = Field(..., example=35.0)


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": {k: {"fitted": m.is_fitted} for k, m in _models.items()},
        "timestamp": datetime.now().isoformat(),
    }

@app.get("/services")
def list_services():
    return {
        "M1_METROPOLITANO": ["RA", "RB", "RC", "RD"],
        "M2_EXPRESO":       ["EXP1","EXP2","EXP3","EXP5","EXP6","EXP7",
                             "EXP8","EXP9","EXP_10","EXP_11","EXP_12","EXP_13"],
        "M3_SUPEREXPRESO":  ["SX", "SXN"],
        "M4_ALIMENTADORA":  ["AN_01..AN_24", "AS_02", "AS_04", "AS_07", "AS_08"],
        "M5_LECHUCERO":     ["SN", "Lechucero", "LVE", "LBI"],
        "M6_CORREDOR":      ["201","204","206","209","301","303","305",
                             "336","401","404","405","406","412"],
        "M7_METRO":         ["L1", "L2"],
    }

@app.post("/predict/eta")
def predict_eta(req: ETARequest):
    model_key = classify_service(req.route_id)
    model     = _get_model(model_key)
    feats     = _time_features(req.timestamp)
    feats.update({
        "VELOCIDAD_MEDIA":   req.velocidad_media,
        "HEADWAY_MIN":       req.headway_min,
        "DIST_PARADERO_KM":  req.dist_paradero_km,
        "N_BUSES_EN_RUTA":   req.n_buses_en_ruta,
        "SENTIDO_NUM":       req.sentido_num,
        "MODELO_NUM":        MODEL_ENCODING.get(model_key, 3),
        "CONTEO_HORA":       100.0,
        "EJE_NUM":           0,
        "ZONA_NUM":          1 if req.route_id.upper().startswith("AS") else 0,
    })
    result = model.predict(feats, ruta_id=req.route_id)
    result.update({"route_id": req.route_id, "paradero_id": req.paradero_id})
    return result

@app.post("/predict/od")
def predict_od(req: ODRequest):
    model_key = classify_service(req.route_id)
    model     = _get_model(model_key)
    feats     = _time_features(req.timestamp)
    hav, man  = _haversine(req.origen_lat, req.origen_lon,
                            req.destino_lat, req.destino_lon)
    spacing   = STOP_SPACING_KM.get(model_key, 0.6)
    feats.update({
        "DIST_HAVERSINE_KM": hav,
        "DIST_MANHATTAN_KM": man,
        "N_PARADAS":         max(1, round(hav / spacing)),
        "N_ESTACIONES":      max(1, round(hav / spacing)),
        "VELOCIDAD_MEDIA":   req.velocidad_media,
        "MODELO_NUM":        MODEL_ENCODING.get(model_key, 3),
        "CONTEO_HORA":       100.0,
        "EJE_NUM":           0,
    })
    result = model.predict_od(feats,
                               orig_stop=req.parada_origen or "",
                               dest_stop=req.parada_destino or "")
    result.update({"route_id": req.route_id, "dist_km": round(hav, 2)})
    return result

@app.post("/predict/regularity")
def predict_regularity(req: RegularityRequest):
    model_key = classify_service(req.route_id)
    model     = _get_model(model_key)
    feats     = _time_features(req.timestamp)
    feats.update({
        "HEADWAY_MIN":     req.headway_min,
        "DELAY_MIN":       req.delay_min or 0.0,
        "DELAY_MIN_LAG1":  req.delay_min_lag1,
        "VELOCIDAD_MEDIA": req.velocidad_media,
        "SENTIDO_NUM":     0,
        "MODELO_NUM":      MODEL_ENCODING.get(model_key, 3),
        "CONTEO_HORA":     100.0,
        "EJE_NUM":         0,
    })
    result = model.predict_regularity(feats, delay_min=req.delay_min)
    result.update({"route_id": req.route_id, "modelo": model_key})
    return result

@app.post("/predict/alerts")
def predict_alerts(req: AlertsRequest):
    model_key = classify_service(req.route_id)
    model     = _get_model(model_key)
    feats     = _time_features(req.timestamp)
    feats.update({
        "VELOCIDAD_MEDIA":   req.velocidad_media,
        "HEADWAY_MIN":       req.headway_min,
        "DIST_PARADERO_KM":  req.dist_paradero_km,
        "N_BUSES_EN_RUTA":   3,
        "DELAY_MIN":         req.delay_min,
        "DELAY_MIN_LAG1":    req.delay_min,
        "SENTIDO_NUM":       0,
        "MODELO_NUM":        MODEL_ENCODING.get(model_key, 3),
        "CONTEO_HORA":       100.0,
        "EJE_NUM":           0,
        "ZONA_NUM":          0,
    })
    eta_res = model.predict(feats, ruta_id=req.route_id)
    reg_res = model.predict_regularity(feats, delay_min=req.delay_min)
    alerts  = _build_alerts(eta_res, reg_res, model_key,
                             req.route_id, req.paradero_id,
                             req.delay_min, req.headway_min)
    return {
        "route_id":    req.route_id,
        "paradero_id": req.paradero_id,
        "modelo":      model_key,
        "alertas":     alerts,
        "sin_alertas": len(alerts) == 0,
        "eta_min":     eta_res.get("eta_min"),
        "estado_regularidad": reg_res.get("estado"),
        "confianza_eta": eta_res.get("confianza"),
    }

@app.post("/predict/all")
def predict_all(req: AllRequest):
    """Endpoint principal de Lovable — GPS payload → 3 predicciones + alertas."""
    model_key = classify_service(req.route_id)
    model     = _get_model(model_key)

    ts = None
    if req.scheduled_time:
        from datetime import date
        ts = f"{date.today().isoformat()}T{req.scheduled_time}"
    elif req.ts:
        ts = datetime.fromtimestamp(req.ts / 1000).isoformat()

    feats = _time_features(ts)
    hav, man = _haversine(req.latitude, req.longitude,
                           req.destination_lat, req.destination_lon)
    spacing  = STOP_SPACING_KM.get(model_key, 0.6)
    eje_num  = {"EO": 0, "NS_NORTE": 1, "NS_SUR": 2}.get(
        get_corredor_axis(req.route_id), 0)

    feats.update({
        "VELOCIDAD_MEDIA":   req.speed,
        "HEADWAY_MIN":       req.headway_min,
        "DIST_PARADERO_KM":  1.0,
        "N_BUSES_EN_RUTA":   3,
        "DELAY_MIN":         req.delay_min,
        "DELAY_MIN_LAG1":    req.delay_min,
        "DIST_HAVERSINE_KM": hav,
        "DIST_MANHATTAN_KM": man,
        "N_PARADAS":         max(1, round(hav / spacing)),
        "N_ESTACIONES":      max(1, round(hav / spacing)),
        "SENTIDO_NUM":       req.direction_id,
        "MODELO_NUM":        MODEL_ENCODING.get(model_key, 3),
        "CONTEO_HORA":       100.0,
        "EJE_NUM":           eje_num,
        "ZONA_NUM":          1 if req.route_id.upper().startswith("AS") else 0,
    })

    eta_res = model.predict(feats, ruta_id=req.route_id)
    od_res  = model.predict_od(feats, orig_stop=req.origin_stop,
                                dest_stop=req.destination_stop)
    reg_res = model.predict_regularity(feats, delay_min=req.delay_min)
    alerts  = _build_alerts(eta_res, reg_res, model_key,
                             req.route_id, req.target_stop,
                             req.delay_min, req.headway_min)

    return {
        "route_id":    req.route_id,
        "modelo":      model_key,
        "target_stop": req.target_stop,
        "eta":         eta_res,
        "od":          od_res,
        "regularity":  reg_res,
        "alertas":     alerts,
        "dist_km":     round(hav, 2),
        "timestamp":   datetime.now().isoformat(),
    }

@app.post("/predict/metro/crowding")
def metro_crowding(req: MetroCrowdingRequest):
    """Alerta de alta demanda en estación del metro."""
    model = _get_model("M7_METRO")
    alert = model.crowding_alert(req.hora_dec, req.estacion)
    return {"estacion": req.estacion, "hora_dec": req.hora_dec, **alert}

@app.post("/predict/lechucero/gap")
def lechucero_gap(req: LechuceroGapRequest):
    """Alerta de 'sin bus nocturno hace mucho tiempo'."""
    model = _get_model("M5_LECHUCERO")
    alert = model.alert_gap(req.dow, req.headway_actual)
    return alert
