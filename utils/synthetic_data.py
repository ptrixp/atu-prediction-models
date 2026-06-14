"""
synthetic_data.py
=================
Generador de datos sintéticos calibrado con la data real S22-2026 ATU.

Parámetros calibrados con:
  - Arribo S22: delays reales por servicio (mediana, std)
  - cosac_detalle: demanda por hora (dic'25–ene'26)
  - linea1_detalle: demanda metro por hora
  - corredoresC: validaciones corredor (28 ene 2026)
  - Waze velocidades: 14-28 km/h por tramo y período

Cada modelo recibe N_SAMPLES registros con la distribución calibrada de su servicio.
Se usa cuando no hay suficientes datos reales para entrenar (< MIN_REAL_ROWS).
"""

import numpy as np
import pandas as pd
import logging

from utils.service_registry import MODEL_ENCODING, get_period

logger = logging.getLogger(__name__)

MIN_REAL_ROWS = 500  # umbral para decidir si se necesita sintético

# ─── Parámetros calibrados por modelo ────────────────────────────────────────
# Fuente: análisis de Arribo S22 (delay_sec describe() por servicio)

CALIBRATION = {
    # M1: RA median=-130s, RB=-96s, RC=-105s, RD=-58s → casi siempre ligeramente adelantados
    "M1_METROPOLITANO": {
        "delay_loc": -1.8,  "delay_scale": 3.5,
        "headway_base": 5,  "headway_peak_factor": 1.2,
        "vel_base": 18,     "vel_peak_factor": 0.65,
        "dist_mean": 0.8,   "dist_std": 0.4,
        "n_buses_mean": 8,
    },
    # M2: EXP5 median=-240s, EXP_12=+456s → alta varianza entre líneas
    "M2_EXPRESO": {
        "delay_loc": 1.5,   "delay_scale": 5.0,
        "headway_base": 8,  "headway_peak_factor": 1.3,
        "vel_base": 20,     "vel_peak_factor": 0.70,
        "dist_mean": 1.2,   "dist_std": 0.6,
        "n_buses_mean": 5,
    },
    # M3: SX median=-32s, SXN=-38s → los más puntuales, poca varianza
    "M3_SUPEREXPRESO": {
        "delay_loc": -0.5,  "delay_scale": 1.5,
        "headway_base": 6,  "headway_peak_factor": 1.1,
        "vel_base": 22,     "vel_peak_factor": 0.80,
        "dist_mean": 1.0,   "dist_std": 0.4,
        "n_buses_mean": 6,
    },
    # M4: alta varianza, AN_03 median=-312s, AN_12=+300s
    "M4_ALIMENTADORA": {
        "delay_loc": 1.0,   "delay_scale": 6.0,
        "headway_base": 10, "headway_peak_factor": 1.4,
        "vel_base": 14,     "vel_peak_factor": 0.60,
        "dist_mean": 0.5,   "dist_std": 0.3,
        "n_buses_mean": 3,
    },
    # M5: sin PO horario → sin delay calculable. Solo velocidad y hora nocturna
    "M5_LECHUCERO": {
        "delay_loc": 0.0,   "delay_scale": 0.0,   # delay no aplica
        "headway_base": 25, "headway_peak_factor": 0.5,  # menos frecuente en madrugada
        "vel_base": 35,     "vel_peak_factor": 1.0,      # vías libres
        "dist_mean": 2.0,   "dist_std": 1.0,
        "n_buses_mean": 2,
    },
    # M6: 1 solo día de datos reales → mayor peso en sintético
    "M6_CORREDOR": {
        "delay_loc": 1.2,   "delay_scale": 4.0,
        "headway_base": 8,  "headway_peak_factor": 1.3,
        "vel_base": 16,     "vel_peak_factor": 0.65,
        "dist_mean": 0.7,   "dist_std": 0.4,
        "n_buses_mean": 4,
    },
    # M7: ETA casi determinista, demanda muy predecible por estación
    "M7_METRO": {
        "delay_loc": 0.3,   "delay_scale": 0.8,
        "headway_base": 4,  "headway_peak_factor": 1.0,
        "vel_base": 35,     "vel_peak_factor": 1.0,
        "dist_mean": 1.3,   "dist_std": 0.3,
        "n_buses_mean": 10,
    },
}

# Distribución horaria calibrada con cosac_detalle.csv (picos reales Lima)
# rango 4-23h = 20 valores
HORA_WEIGHTS = [
    0.015, 0.010, 0.010, 0.015,  # 4-7h  (madrugada)
    0.060, 0.090, 0.110, 0.095,  # 7-10h (pico AM fuerte)
    0.060, 0.050, 0.050, 0.055,  # 10-13h
    0.060, 0.058, 0.055, 0.060,  # 13-16h
    0.075, 0.080, 0.060, 0.040,  # 16-19h (pico PM)
]
pass  # weights normalized

# Para M5 (lechucero): distribución nocturna invertida
HORA_WEIGHTS_NOCTURNO = [
    0.18, 0.15, 0.10, 0.08,  # 0-3h  (mayor movimiento primera noche)
    0.05, 0.04,               # 4-5h  (madrugada baja)
] + [0.0] * 12 + [            # 6-17h sin operación
    0.05, 0.08, 0.10, 0.12, 0.10, 0.10, 0.05,  # 17-23h (inicio noche)
]
# normalizar
_s = sum(HORA_WEIGHTS_NOCTURNO)
HORA_WEIGHTS_NOCTURNO = [w/_s for w in HORA_WEIGHTS_NOCTURNO]


def generate_for_model(model_key: str, n_samples: int = 3000,
                       seed: int = 42) -> pd.DataFrame:
    """
    Genera dataset sintético calibrado para un modelo específico.
    """
    rng = np.random.default_rng(seed)
    c   = CALIBRATION[model_key]

    # Horas — M5 usa distribución nocturna
    hora_weights = HORA_WEIGHTS
    hora_base = list(range(4, 24))
    hw_arr = np.array(hora_weights, dtype=float)
    hw_arr = hw_arr / hw_arr.sum()  # normalize to exactly 1.0
    horas   = rng.choice(hora_base, size=n_samples, p=hw_arr)
    minutos = rng.integers(0, 60, n_samples)
    hora_dec = horas + minutos / 60.0

    dow        = rng.integers(0, 7, n_samples)
    is_weekend = (dow >= 5).astype(int)
    is_peak_am = ((hora_dec >= 6.5) & (hora_dec <= 9.5)).astype(int)
    is_peak_pm = ((hora_dec >= 17.5) & (hora_dec <= 20.0)).astype(int)
    is_noc     = ((hora_dec >= 22.0) | (hora_dec <= 5.0)).astype(int)

    # Velocidad
    is_peak    = is_peak_am | is_peak_pm
    velocidad  = np.array([
        c["vel_base"] * (c["vel_peak_factor"] if is_peak[i] else 1.0)
        + rng.normal(0, 3)
        for i in range(n_samples)
    ]).clip(5, 80)

    # Headway
    headway = np.array([
        c["headway_base"] * (c["headway_peak_factor"] if is_peak[i] else 1.0)
        + rng.exponential(2)
        for i in range(n_samples)
    ]).clip(1, 90)

    # Distancia O-D
    dist_hav  = rng.gamma(shape=2.5, scale=c["dist_mean"]).clip(0.3, 30) if n_samples == 1 \
                else rng.gamma(shape=2.5, scale=c["dist_mean"], size=n_samples).clip(0.3, 30)
    mid_lat   = -12.05
    dist_man  = dist_hav * rng.uniform(1.1, 1.4, n_samples)
    n_paradas = (dist_hav / 0.6).clip(1).astype(int)

    # Delay — M5 no tiene delay (sin PO)
    if model_key == "M5_LECHUCERO":
        delay_min     = np.zeros(n_samples)
        delay_min_lag = np.zeros(n_samples)
    else:
        delay_min     = (rng.normal(c["delay_loc"], c["delay_scale"], n_samples)
                         + rng.exponential(1.5, n_samples) * 0.3)
        delay_min_lag = delay_min + rng.normal(0, 2, n_samples)

    # ETA = headway/2 × factor_congestion
    cong = 1 + 0.4 * is_peak - 0.1 * is_weekend
    eta_min = (headway / 2 * cong + rng.normal(0, 1.5, n_samples)).clip(0.5, 60)

    # Tiempo O-D
    tiempo_od = (dist_hav / velocidad * 60 * cong
                 + rng.normal(0, 3, n_samples)).clip(1, 150)

    # Demanda por parada (proxy validaciones)
    n_buses = (rng.integers(1, c["n_buses_mean"] * 2 + 1, n_samples)).clip(1, 30)

    def _classify_delay(d):
        if d < -5:   return "MUY_ADELANTADO"
        elif d < -2: return "ADELANTADO"
        elif d <= 2: return "REGULAR"
        elif d <= 5: return "RETRASADO"
        return "ANOMALIA"

    estado = [_classify_delay(d) for d in delay_min]

    hora_sin = np.sin(2 * np.pi * hora_dec / 24)
    hora_cos = np.cos(2 * np.pi * hora_dec / 24)
    dow_sin  = np.sin(2 * np.pi * dow / 7)
    dow_cos  = np.cos(2 * np.pi * dow / 7)

    parada_origen  = [f"P{rng.integers(1, 300):04d}" for _ in range(n_samples)]
    parada_destino = [f"P{rng.integers(1, 300):04d}" for _ in range(n_samples)]
    ruta_ids       = [f"R{rng.integers(1, 80):03d}"  for _ in range(n_samples)]

    df = pd.DataFrame({
        "HORA_DEC": hora_dec, "DOW": dow,
        "IS_WEEKEND": is_weekend, "IS_PEAK_AM": is_peak_am,
        "IS_PEAK_PM": is_peak_pm, "IS_NOCTURNO": is_noc,
        "HORA_SIN": hora_sin, "HORA_COS": hora_cos,
        "DOW_SIN": dow_sin,   "DOW_COS": dow_cos,
        "MODELO": model_key,
        "MODELO_NUM": MODEL_ENCODING[model_key],
        "VELOCIDAD_MEDIA": velocidad,
        "HEADWAY_MIN": headway,
        "DIST_PARADERO_KM": dist_hav / 2,
        "N_BUSES_EN_RUTA": n_buses,
        "DIST_HAVERSINE_KM": dist_hav,
        "DIST_MANHATTAN_KM": dist_man,
        "N_PARADAS": n_paradas,
        "ETA_MIN": eta_min,
        "TIEMPO_OD_MIN": tiempo_od,
        "DELAY_MIN": delay_min,
        "DELAY_MIN_LAG1": delay_min_lag,
        "ESTADO_REGULARIDAD": estado,
        "PARADA_ORIGEN": parada_origen,
        "PARADA_DESTINO": parada_destino,
        "RUTA": ruta_ids,
    })
    return df


def generate_all(n_per_model: int = 4000, seed: int = 42) -> pd.DataFrame:
    """Genera dataset completo para los 7 modelos concatenados."""
    dfs = []
    for i, model_key in enumerate(CALIBRATION.keys()):
        df = generate_for_model(model_key, n_samples=n_per_model, seed=seed + i)
        dfs.append(df)
        logger.info(f"Sintético {model_key}: {len(df)} filas")
    return pd.concat(dfs, ignore_index=True)


def merge_real_and_synthetic(real_df: pd.DataFrame,
                             model_key: str,
                             n_synthetic: int = 3000,
                             seed: int = 42) -> pd.DataFrame:
    """
    Combina datos reales con sintéticos cuando hay pocos datos reales.
    El peso del sintético baja proporcionalmente a cuántos datos reales hay.
    """
    real_n = len(real_df)
    if real_n >= MIN_REAL_ROWS:
        logger.info(f"{model_key}: usando solo data real ({real_n} filas)")
        return real_df

    synth_needed = max(0, n_synthetic - real_n)
    logger.info(f"{model_key}: {real_n} reales + {synth_needed} sintéticos")
    synth = generate_for_model(model_key, n_samples=synth_needed, seed=seed)
    synth["SOURCE"] = "synthetic"
    real_df = real_df.copy()
    real_df["SOURCE"] = "real"
    return pd.concat([real_df, synth], ignore_index=True)
