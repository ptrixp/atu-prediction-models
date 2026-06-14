"""
feature_engineering.py
=======================
Pipeline de ingesta y feature engineering para los 7 modelos ATU.

Fuentes reales disponibles:
  - Arribo S22-2026.xlsx          → M1, M2, M3, M4 (delay vs PO)
  - cosac_detalle.csv             → M1, M2, M3 (validaciones por estación/hora)
  - linea1_detalle.csv            → M7 (validaciones metro por estación/hora)
  - corredoresC.csv               → M6 (validaciones corredor por parada/hora)
  - GPS SICM (transmisión real)   → todos (velocidad, posición, headway)
  - velocidades_por_tramo (Waze)  → todos (velocidad de tramo)

Esquemas de columnas por fuente (ver diccionario_de_datos.txt):
  GPS:        PLACA, LATITUD, LONGITUD, VELOCIDAD, IMEI_GPS, FECHA_HORA_TRACK,
              RUTA, SENTIDO, ALTITUD, ORIENTACION, FECHORA_INI_VIAJE,
              DNI_CONDUCTOR, IS_TRAMA_DESFASADA, IDENTIFICADOR
  Arribo:     Fecha, ID, Código del Conductor, Código de Servicio,
              Ruta / Servicio, Estacion de Parada, Sentido, Ejecutado, Programado
  Validaciones COSAC/Metro: fecha, hora, perfil, precio, n_estacion, desc_estacion, conteo
  Validaciones Corredores:  id, fecha, hora, placa, perfil, linea, precio,
                            nomb_parada, desc_parada, sentido, conteo
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2

from utils.service_registry import classify_service, MODEL_ENCODING, get_corredor_axis, get_period

logger = logging.getLogger(__name__)


# ─── Loaders por fuente ──────────────────────────────────────────────────────

def load_arribo(filepath: str) -> pd.DataFrame:
    """
    Carga Arribo S22-2026.xlsx.
    Columnas: Fecha, ID, Código del Conductor, Código de Servicio,
              Ruta / Servicio, Estacion de Parada, Sentido, Ejecutado, Programado
    """
    df = pd.read_excel(filepath)
    df.columns = [c.strip() for c in df.columns]

    df["Ejecutado"]  = pd.to_datetime(df["Ejecutado"].astype(str),  format="mixed", errors="coerce")
    df["Programado"] = pd.to_datetime(df["Programado"].astype(str), format="mixed", errors="coerce")
    df["DELAY_SEC"]  = (df["Ejecutado"] - df["Programado"]).dt.total_seconds()
    df["DELAY_MIN"]  = df["DELAY_SEC"] / 60

    df["HORA_DEC"] = df["Ejecutado"].dt.hour + df["Ejecutado"].dt.minute / 60
    df["DOW"]      = df["Ejecutado"].dt.dayofweek
    df["FECHA_DT"] = pd.to_datetime(df["Fecha"], format="mixed", errors="coerce")

    # Extraer prefijo de servicio limpio
    df["SERVICIO"] = (
        df["Ruta / Servicio"].str.split(" - ").str[0].str.strip()
    )
    df["MODELO"] = df["SERVICIO"].apply(classify_service)
    df["MODELO_NUM"] = df["MODELO"].map(MODEL_ENCODING).fillna(3).astype(int)

    # Sentido como binario
    df["SENTIDO_NUM"] = df["Sentido"].map(
        {"Norte": 0, "Sur": 1, "IDA": 0, "VUELTA": 1, "NS": 0, "SN": 1,
         "EO": 0, "OE": 1}
    ).fillna(0).astype(int)

    logger.info(f"Arribo cargado: {len(df)} filas | modelos: {df['MODELO'].value_counts().to_dict()}")
    return df


def load_validaciones_cosac(filepath: str) -> pd.DataFrame:
    """cosac_detalle.csv — fecha, hora, perfil, precio, n_estacion, desc_estacion, conteo"""
    df = pd.read_csv(filepath, encoding="latin1")
    df.columns = [c.strip().lower() for c in df.columns]
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["hora"]  = pd.to_numeric(df["hora"], errors="coerce")
    df["conteo"] = pd.to_numeric(df["conteo"], errors="coerce").fillna(0)
    df["MODELO"] = "M1_METROPOLITANO"  # COSAC = M1/M2/M3 agrupados para demanda
    logger.info(f"COSAC validaciones: {len(df)} filas | estaciones: {df['desc_estacion'].nunique()}")
    return df


def load_validaciones_metro(filepath: str) -> pd.DataFrame:
    """linea1_detalle.csv — fecha, hora, perfil, precio, n_estacion, desc_estacion, conteo"""
    df = pd.read_csv(filepath, encoding="latin1")
    df.columns = [c.strip().lower() for c in df.columns]
    df["fecha"]  = pd.to_datetime(df["fecha"], errors="coerce")
    df["hora"]   = pd.to_numeric(df["hora"], errors="coerce")
    df["conteo"] = pd.to_numeric(df["conteo"], errors="coerce").fillna(0)
    df["MODELO"] = "M7_METRO"
    logger.info(f"Metro L1 validaciones: {len(df)} filas | estaciones: {df['desc_estacion'].nunique()}")
    return df


def load_validaciones_corredores(filepath: str) -> pd.DataFrame:
    """corredoresC.csv — id;fecha;hora;placa;perfil;linea;precio;nomb_parada;desc_parada;sentido;conteo"""
    df = pd.read_csv(filepath, sep=";", encoding="latin1")
    df.columns = [c.strip().lower() for c in df.columns]
    df["fecha"]  = pd.to_datetime(df["fecha"], errors="coerce")
    df["hora"]   = pd.to_numeric(df["hora"], errors="coerce")
    df["conteo"] = pd.to_numeric(df["conteo"], errors="coerce").fillna(0)
    df["MODELO"] = "M6_CORREDOR"
    df["EJE"]    = df["linea"].apply(get_corredor_axis)
    df["SENTIDO_NUM"] = df["sentido"].map(
        {"EO": 0, "OE": 1, "NS": 0, "SN": 1, "IDA": 0, "VUELTA": 1}
    ).fillna(0).astype(int)
    logger.info(f"Corredores validaciones: {len(df)} filas | líneas: {sorted(df['linea'].unique())}")
    return df


def load_gps(filepath: str) -> pd.DataFrame:
    """
    GPS transmisión SICM.
    Columnas: PLACA, LATITUD, LONGITUD, VELOCIDAD, IMEI_GPS, FECHA_HORA_TRACK,
              RUTA, SENTIDO, ALTITUD, ORIENTACION, FECHA_TRACK_DIA,
              FECHORA_INI_VIAJE, DNI_CONDUCTOR, ORIGEN, IS_TRAMA_DESFASADA, IDENTIFICADOR
    También acepta el formato del hackaton (imei, latitude, longitude, route_id, ts, speed, etc.)
    """
    df = pd.read_csv(filepath, low_memory=False)
    df.columns = [c.upper().strip() for c in df.columns]

    # Normalizar nombres entre formato SICM y hackaton GPS
    rename = {
        "LATITUDE": "LATITUD", "LONGITUDE": "LONGITUD",
        "SPEED": "VELOCIDAD", "ROUTE_ID": "RUTA",
        "LICENSE_PLATE": "PLACA",
    }
    df.rename(columns={k: v for k, v in rename.items() if k in df.columns}, inplace=True)

    # Timestamp: FECHA_HORA_TRACK (SICM) o TS en milisegundos (hackaton)
    if "FECHA_HORA_TRACK" in df.columns:
        df["TS_DT"] = pd.to_datetime(df["FECHA_HORA_TRACK"], errors="coerce")
    elif "TS" in df.columns:
        df["TS_DT"] = pd.to_datetime(df["TS"], unit="ms", errors="coerce")
    else:
        df["TS_DT"] = pd.NaT

    # Filtrar tramas desfasadas si existe la columna
    if "IS_TRAMA_DESFASADA" in df.columns:
        df = df[df["IS_TRAMA_DESFASADA"] != 1]

    df["MODELO"]     = df["RUTA"].apply(classify_service)
    df["MODELO_NUM"] = df["MODELO"].map(MODEL_ENCODING).fillna(3).astype(int)
    df["HORA_DEC"]   = df["TS_DT"].dt.hour + df["TS_DT"].dt.minute / 60
    df["DOW"]        = df["TS_DT"].dt.dayofweek
    df["VELOCIDAD"]  = pd.to_numeric(df["VELOCIDAD"], errors="coerce").fillna(0).clip(0, 120)

    logger.info(f"GPS cargado: {len(df)} filas")
    return df


# ─── Feature Engineering compartido ─────────────────────────────────────────

def build_time_features(df: pd.DataFrame, hora_col: str = "HORA_DEC",
                        dow_col: str = "DOW") -> pd.DataFrame:
    """Agrega features temporales cíclicas."""
    h = df[hora_col].fillna(8.0)
    d = df[dow_col].fillna(0)
    df["IS_WEEKEND"] = (d >= 5).astype(int)
    df["IS_PEAK_AM"] = ((h >= 6.5) & (h <= 9.5)).astype(int)
    df["IS_PEAK_PM"] = ((h >= 17.5) & (h <= 20.0)).astype(int)
    df["IS_NOCTURNO"]= ((h >= 22.0) | (h <= 5.0)).astype(int)
    df["HORA_SIN"]   = np.sin(2 * np.pi * h / 24)
    df["HORA_COS"]   = np.cos(2 * np.pi * h / 24)
    df["DOW_SIN"]    = np.sin(2 * np.pi * d / 7)
    df["DOW_COS"]    = np.cos(2 * np.pi * d / 7)
    return df


def compute_headway(df: pd.DataFrame, group_cols: list,
                    time_col: str = "TS_DT") -> pd.DataFrame:
    """Headway entre buses consecutivos en mismo grupo de ruta/paradero."""
    df = df.sort_values(group_cols + [time_col])
    df["HEADWAY_MIN"] = (
        df.groupby(group_cols)[time_col]
          .diff()
          .dt.total_seconds() / 60
    )
    return df


def compute_delay_from_arribo(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula delay en minutos desde Arribo (Ejecutado - Programado)."""
    if "DELAY_MIN" not in df.columns:
        df["DELAY_MIN"] = df["DELAY_SEC"] / 60
    df["DELAY_MIN_LAG1"] = df.groupby(
        ["SERVICIO", "Estacion de Parada"]
    )["DELAY_MIN"].shift(1).fillna(0)

    # Clasificación de regularidad (igual que modelo anterior)
    def _classify(d):
        if pd.isna(d):   return "REGULAR"
        if d < -5:       return "MUY_ADELANTADO"
        elif d < -2:     return "ADELANTADO"
        elif d <= 2:     return "REGULAR"
        elif d <= 5:     return "RETRASADO"
        return "ANOMALIA"

    df["ESTADO_REGULARIDAD"] = df["DELAY_MIN"].apply(_classify)
    return df


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    p1, p2 = radians(lat1), radians(lat2)
    dp = radians(lat2 - lat1)
    dl = radians(lon2 - lon1)
    a  = sin(dp/2)**2 + cos(p1)*cos(p2)*sin(dl/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))


def build_od_dist(lat1, lon1, lat2, lon2):
    dist_hav = haversine_km(lat1, lon1, lat2, lon2)
    mid_lat  = radians((lat1 + lat2) / 2)
    dist_man = abs(lat2 - lat1) * 111 + abs(lon2 - lon1) * 111 * cos(mid_lat)
    return dist_hav, dist_man


# ─── Demanda por parada/estación (desde validaciones) ────────────────────────

def build_demand_features(val_df: pd.DataFrame,
                          hora_col: str = "hora",
                          conteo_col: str = "conteo") -> pd.DataFrame:
    """
    Agrega features de demanda horaria desde archivos de validaciones.
    Retorna df con: HORA, CONTEO_MEAN, CONTEO_STD, CONTEO_RANK.
    """
    agg = (val_df.groupby(hora_col)[conteo_col]
           .agg(CONTEO_MEAN="mean", CONTEO_STD="std", CONTEO_SUM="sum")
           .reset_index()
           .rename(columns={hora_col: "HORA"}))
    agg["CONTEO_STD"]  = agg["CONTEO_STD"].fillna(0)
    agg["CONTEO_RANK"] = agg["CONTEO_SUM"].rank(pct=True)
    return agg
