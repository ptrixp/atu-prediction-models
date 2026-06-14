# ATU Prediction Models v2.0 — 7 Modelos

Sistema de predicción de transporte público para la red ATU de Lima.
Arquitectura refactorizada con un modelo especializado por tipo de servicio.

## Arquitectura

| Modelo | Servicios | Lógica | Data real |
|---|---|---|---|
| **M1_METROPOLITANO** | RA, RB, RC, RD | GBM + demanda COSAC | Arribo S22, cosac_detalle |
| **M2_EXPRESO** | EXP1–EXP_13 | GBM por línea EXP | Arribo S22, cosac_detalle |
| **M3_SUPEREXPRESO** | SX, SXN | GBM ligero (baja varianza) | Arribo S22 |
| **M4_ALIMENTADORA** | AN_xx, AS_xx | GBM por zona (Norte/Sur) | Arribo S22 |
| **M5_LECHUCERO** | SN, Lechucero | Regresión lineal + alerta de gap | GPS nocturno |
| **M6_CORREDOR** | 201–412 | GBM por eje EO/NS | corredoresC.csv |
| **M7_METRO** | L1, L2 | ETA determinista + crowding | linea1_detalle |

## Instalación

```bash
pip install -r requirements.txt
```

## Entrenamiento

```bash
# Solo con datos sintéticos (calibrados con parámetros reales S22):
python train_models.py

# Con datos reales:
python train_models.py \
  --arribo   /ruta/Arribo_de_la_Semana_22-2026.xlsx \
  --cosac    /ruta/cosac_detalle.csv \
  --metro    /ruta/linea1_detalle.csv \
  --corredores /ruta/corredoresC.csv \
  --output   ./trained
```

## API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Endpoints principales

```
POST /predict/eta          → ETA del bus al paradero
POST /predict/od           → Tiempo de viaje O-D
POST /predict/regularity   → Estado REGULAR/RETRASADO/ANOMALIA/SIN_PO
POST /predict/alerts       → Alertas combinadas (una sola llamada desde Lovable)
POST /predict/all          → ETA + OD + Regularidad + Alertas (payload GPS completo)
POST /predict/metro/crowding → Alerta crowding en estación metro
POST /predict/lechucero/gap  → Alerta "sin bus nocturno"
GET  /health               → Estado de los 7 modelos
GET  /services             → Mapa de servicios → modelo
```

### Ejemplo payload `/predict/all` (formato GPS SICM / Lovable)

```json
{
  "imei": 123456789,
  "latitude": -12.0464,
  "longitude": -77.0428,
  "destination_lat": -12.1100,
  "destination_lon": -77.0200,
  "route_id": "EXP5",
  "speed": 20.0,
  "direction_id": 0,
  "target_stop": "NA06",
  "delay_min": 2.5,
  "headway_min": 7.0
}
```

## Notas importantes por modelo

### M5 Lechucero
- **No tiene modelo de regularidad** (sin PO horario fijo).
- El endpoint `/predict/regularity` retorna `estado: "SIN_PO"`.
- Usar `/predict/lechucero/gap` para alertas nocturnas.

### M7 Metro
- ETA es **determinista** (distancia / velocidad fija 35 km/h).
- Usar `/predict/metro/crowding` para alertas de alta demanda por estación.
- No aparece en Arribo S22 — tiene sistema de control propio.

### M6 Corredor
- Solo 1 día de data real (28 ene 2026).
- El entrenamiento usa mayoritariamente datos sintéticos calibrados.
- Incluye sub-modelo por eje: EO (este-oeste), NS_NORTE, NS_SUR.

## Estructura de archivos

```
atu_models/
├── models/
│   ├── base_model.py          # Clase base para los 7 modelos
│   ├── m1_metropolitano.py    # M1: RA/RB/RC/RD + COSAC
│   └── m2_to_m7.py            # M2 al M7
├── api/
│   └── main.py                # FastAPI con todos los endpoints
├── utils/
│   ├── service_registry.py    # Clasificador route_id → modelo
│   ├── feature_engineering.py # Loaders y feature engineering por fuente
│   └── synthetic_data.py      # Generador sintético calibrado S22
├── trained/                   # Modelos .pkl entrenados
├── train_models.py            # Script de entrenamiento
└── requirements.txt
```
