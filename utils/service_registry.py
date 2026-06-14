"""
service_registry.py
===================
Clasificador central de servicios ATU → 7 modelos.
Basado en data real: Arribo S22-2026, corredoresC, cosac_detalle, linea1_detalle.

Modelos:
  M1_METROPOLITANO : RA, RB, RC, RD  (Regular A-D, corredor troncal COSAC)
  M2_EXPRESO       : EXP1-EXP_13     (Expresos del Metropolitano)
  M3_SUPEREXPRESO  : SX, SXN         (Super Expreso y Super Expreso Norte)
  M4_ALIMENTADORA  : AN_xx, AS_xx    (Alimentadoras Norte y Sur)
  M5_LECHUCERO     : SN, LVE, LBI   (Servicio Nocturno / Lechucero — sin PO horario)
  M6_CORREDOR      : 201-412         (Corredores Complementarios 200x/300x/400x)
  M7_METRO         : L1, L2          (Líneas de Metro)
"""

SERVICE_MODEL_MAP = {
    # M1 — Metropolitano Troncal
    "RA": "M1_METROPOLITANO", "RB": "M1_METROPOLITANO",
    "RC": "M1_METROPOLITANO", "RD": "M1_METROPOLITANO",
    # M2 — Expresos
    "EXP1": "M2_EXPRESO",  "EXP2": "M2_EXPRESO",  "EXP3": "M2_EXPRESO",
    "EXP5": "M2_EXPRESO",  "EXP6": "M2_EXPRESO",  "EXP7": "M2_EXPRESO",
    "EXP8": "M2_EXPRESO",  "EXP9": "M2_EXPRESO",
    "EXP_10": "M2_EXPRESO","EXP_11": "M2_EXPRESO",
    "EXP_12": "M2_EXPRESO","EXP_13": "M2_EXPRESO",
    # M3 — Super Expreso
    "SX": "M3_SUPEREXPRESO", "SXN": "M3_SUPEREXPRESO",
    # M5 — Lechucero / Nocturno
    "SN": "M5_LECHUCERO",
    # M7 — Metro
    "L1": "M7_METRO", "L2": "M7_METRO",
    "LIN1": "M7_METRO", "LIN2": "M7_METRO",
}

# Prefijos para clasificación dinámica (orden importa: más específico primero)
PREFIX_MODEL_MAP = [
    (("SXN",),                             "M3_SUPEREXPRESO"),
    (("SX",),                              "M3_SUPEREXPRESO"),
    (("EXP",),                             "M2_EXPRESO"),
    (("RA", "RB", "RC", "RD"),             "M1_METROPOLITANO"),
    (("TM", "MC", "COSAC"),                "M1_METROPOLITANO"),
    (("AN_", "AS_"),                       "M4_ALIMENTADORA"),
    (("SN", "LECH", "LVE", "LBI"),        "M5_LECHUCERO"),
    (("L1", "L2", "LIN"),                  "M7_METRO"),
    (("C0",),                              "M6_CORREDOR"),
]

# Líneas numéricas corredores (corredoresC.csv: 201, 204, 206, 209, 301...)
CORREDOR_NUMERIC_RANGES = [(200, 499), (3180, 3180)]

# Codificación numérica para features ML
MODEL_ENCODING = {
    "M1_METROPOLITANO": 0,
    "M2_EXPRESO":       1,
    "M3_SUPEREXPRESO":  2,
    "M4_ALIMENTADORA":  3,
    "M5_LECHUCERO":     4,
    "M6_CORREDOR":      5,
    "M7_METRO":         6,
}

# Sub-eje para M6 (corredoresC: sentido real de la ruta)
CORREDOR_AXIS = {
    201: "EO", 204: "EO", 206: "EO", 209: "EO",
    301: "NS_NORTE", 303: "NS_NORTE", 305: "NS_NORTE",
    336: "NS_NORTE", 357: "NS_NORTE", 370: "NS_NORTE",
    371: "NS_NORTE", 3180: "NS_NORTE",
    401: "NS_SUR", 404: "NS_SUR", 405: "NS_SUR",
    406: "NS_SUR", 412: "NS_SUR",
}

# Parámetros históricos por modelo (fallback cuando el modelo no está entrenado)
HISTORICAL_ETA = {
    "M1_METROPOLITANO": {"PEAK": 4.5,  "OFF_PEAK": 3.0,  "NOCTURNO": 6.0},
    "M2_EXPRESO":       {"PEAK": 6.0,  "OFF_PEAK": 4.0,  "NOCTURNO": 8.0},
    "M3_SUPEREXPRESO":  {"PEAK": 3.5,  "OFF_PEAK": 2.5,  "NOCTURNO": 5.0},
    "M4_ALIMENTADORA":  {"PEAK": 10.0, "OFF_PEAK": 8.0,  "NOCTURNO": 15.0},
    "M5_LECHUCERO":     {"PEAK": 20.0, "OFF_PEAK": 15.0, "NOCTURNO": 12.0},
    "M6_CORREDOR":      {"PEAK": 8.0,  "OFF_PEAK": 6.0,  "NOCTURNO": 10.0},
    "M7_METRO":         {"PEAK": 4.0,  "OFF_PEAK": 3.0,  "NOCTURNO": 99.0},
}

# Velocidades medias por modelo y período (km/h)
SPEED_BY_MODEL = {
    "M1_METROPOLITANO": {"PEAK": 18.0, "OFF_PEAK": 26.0, "NOCTURNO": 32.0},
    "M2_EXPRESO":       {"PEAK": 20.0, "OFF_PEAK": 28.0, "NOCTURNO": 35.0},
    "M3_SUPEREXPRESO":  {"PEAK": 22.0, "OFF_PEAK": 30.0, "NOCTURNO": 38.0},
    "M4_ALIMENTADORA":  {"PEAK": 14.0, "OFF_PEAK": 20.0, "NOCTURNO": 28.0},
    "M5_LECHUCERO":     {"PEAK": 30.0, "OFF_PEAK": 35.0, "NOCTURNO": 40.0},
    "M6_CORREDOR":      {"PEAK": 16.0, "OFF_PEAK": 24.0, "NOCTURNO": 30.0},
    "M7_METRO":         {"PEAK": 35.0, "OFF_PEAK": 35.0, "NOCTURNO": 99.0},
}

# Umbrales de alerta ETA por modelo (minutos)
ETA_ALERT_THRESHOLD = {
    "M1_METROPOLITANO": 10,
    "M2_EXPRESO":       12,
    "M3_SUPEREXPRESO":  8,
    "M4_ALIMENTADORA":  20,
    "M5_LECHUCERO":     30,
    "M6_CORREDOR":      15,
    "M7_METRO":         6,
}

# Distancia media entre paradas por modelo (km)
STOP_SPACING_KM = {
    "M1_METROPOLITANO": 0.5,
    "M2_EXPRESO":       0.8,
    "M3_SUPEREXPRESO":  1.0,
    "M4_ALIMENTADORA":  0.4,
    "M5_LECHUCERO":     1.5,
    "M6_CORREDOR":      0.6,
    "M7_METRO":         1.3,
}


def classify_service(route_id: str) -> str:
    """
    Clasifica un route_id en uno de los 7 modelos.
    Acepta códigos de Arribo (RA, EXP1, AN_07) y GPS (TM101, L1, 201).
    """
    rid = str(route_id).upper().strip()

    # Lookup exacto
    if rid in SERVICE_MODEL_MAP:
        return SERVICE_MODEL_MAP[rid]

    # Corredores numéricos
    try:
        n = int(rid)
        for lo, hi in CORREDOR_NUMERIC_RANGES:
            if lo <= n <= hi:
                return "M6_CORREDOR"
    except ValueError:
        pass

    # Prefijos textuales (orden importa)
    for prefixes, model in PREFIX_MODEL_MAP:
        for p in prefixes:
            if rid.startswith(p):
                return model

    # Fallback: alimentadora (mayor cantidad de rutas sin prefijo conocido)
    return "M4_ALIMENTADORA"


def get_corredor_axis(linea) -> str:
    try:
        return CORREDOR_AXIS.get(int(linea), "NS_NORTE")
    except (ValueError, TypeError):
        return "NS_NORTE"


def get_period(hora_dec: float) -> str:
    if 6.5 <= hora_dec <= 9.5 or 17.5 <= hora_dec <= 20.0:
        return "PEAK"
    elif hora_dec >= 22.0 or hora_dec <= 5.0:
        return "NOCTURNO"
    return "OFF_PEAK"
