"""
train_models.py
===============
Entrena los 7 modelos ATU usando data real disponible + sintética calibrada.

Flujo:
  1. Intenta cargar datos reales (Arribo, validaciones, GPS)
  2. Si hay pocos datos reales para un modelo → completa con sintéticos calibrados
  3. Entrena los 7 modelos
  4. Guarda .pkl + metadata en ./trained/

Uso:
  python train_models.py
  python train_models.py --arribo ruta/al/Arribo.xlsx
                         --cosac ruta/cosac_detalle.csv
                         --metro ruta/linea1_detalle.csv
                         --corredores ruta/corredoresC.csv
"""

import os, sys, json, pickle, logging, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

from models.m1_metropolitano import M1MetropolitanoModel
from models.m2_to_m7 import (M2ExpresoModel, M3SuperExpresoModel,
                               M4AlimentadoraModel, M5LechuceroModel,
                               M6CorredorModel, M7MetroModel)
from utils.service_registry import MODEL_ENCODING, get_corredor_axis
from utils.feature_engineering import (load_arribo, load_validaciones_cosac,
                                        load_validaciones_metro,
                                        load_validaciones_corredores,
                                        build_time_features, compute_delay_from_arribo)
from utils.synthetic_data import (generate_for_model, generate_all,
                                   merge_real_and_synthetic, MIN_REAL_ROWS)


MODELS = {
    "M1_METROPOLITANO": M1MetropolitanoModel,
    "M2_EXPRESO":       M2ExpresoModel,
    "M3_SUPEREXPRESO":  M3SuperExpresoModel,
    "M4_ALIMENTADORA":  M4AlimentadoraModel,
    "M5_LECHUCERO":     M5LechuceroModel,
    "M6_CORREDOR":      M6CorredorModel,
    "M7_METRO":         M7MetroModel,
}


def build_arribo_features(arribo_df: pd.DataFrame) -> pd.DataFrame:
    """Transforma Arribo en features listas para entrenamiento."""
    df = compute_delay_from_arribo(arribo_df)
    df = build_time_features(df, hora_col="HORA_DEC", dow_col="DOW")

    # Columnas requeridas por modelos
    df["VELOCIDAD_MEDIA"]  = 18.0   # se actualiza con GPS cuando esté disponible
    df["HEADWAY_MIN"]      = 5.0
    df["DIST_PARADERO_KM"] = 1.0
    df["N_BUSES_EN_RUTA"]  = 4
    df["DIST_HAVERSINE_KM"]= 3.0
    df["DIST_MANHATTAN_KM"]= 4.0
    df["N_PARADAS"]        = 6
    df["SENTIDO_NUM"]      = df.get("SENTIDO_NUM", pd.Series(0, index=df.index))
    df["MODELO_NUM"]       = df["MODELO"].map(MODEL_ENCODING).fillna(3).astype(int)
    df["CONTEO_HORA"]      = 100.0
    df["EJE_NUM"]          = 0
    df["ZONA_NUM"]         = df["SERVICIO"].apply(
        lambda s: 1 if str(s).upper().startswith("AS") else 0)
    df["N_ESTACIONES"]     = 10

    # ETA desde arribo: usamos headway / 2 como proxy (no tenemos GPS en este dataset)
    df["ETA_MIN"] = (df["HEADWAY_MIN"] / 2).clip(0.5, 60)
    df["TIEMPO_OD_MIN"] = df["DIST_HAVERSINE_KM"] / df["VELOCIDAD_MEDIA"] * 60

    df["PARADA_ORIGEN"]  = ""
    df["PARADA_DESTINO"] = ""
    df["RUTA"]           = df["SERVICIO"]
    return df


def main(arribo_path: str = None, cosac_path: str = None,
         metro_path: str = None, corredores_path: str = None,
         output_dir: str = "./trained", n_synthetic: int = 4000):

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── 1. Cargar datos reales ────────────────────────────────────────────────
    arribo_df     = None
    cosac_df      = None
    metro_df      = None
    corredores_df = None

    if arribo_path and Path(arribo_path).exists():
        logger.info(f"Cargando Arribo: {arribo_path}")
        arribo_df = load_arribo(arribo_path)
        arribo_df = build_arribo_features(arribo_df)
        logger.info(f"Arribo procesado: {len(arribo_df)} filas")

    if cosac_path and Path(cosac_path).exists():
        logger.info(f"Cargando COSAC validaciones: {cosac_path}")
        cosac_df = load_validaciones_cosac(cosac_path)

    if metro_path and Path(metro_path).exists():
        logger.info(f"Cargando Metro validaciones: {metro_path}")
        metro_df = load_validaciones_metro(metro_path)

    if corredores_path and Path(corredores_path).exists():
        logger.info(f"Cargando Corredores validaciones: {corredores_path}")
        corredores_df = load_validaciones_corredores(corredores_path)
        # Agregar EJE_NUM para M6
        corredores_df["EJE_NUM"] = corredores_df["linea"].apply(
            lambda l: {"EO": 0, "NS_NORTE": 1, "NS_SUR": 2}.get(get_corredor_axis(l), 0))

    # ── 2. Generar sintéticos globales si no hay ninguna fuente real ──────────
    if arribo_df is None:
        logger.info("Sin datos reales. Generando dataset sintético completo...")
        arribo_df = generate_all(n_per_model=n_synthetic)

    # ── 3. Entrenar cada modelo ───────────────────────────────────────────────
    trained_models = {}
    meta = {
        "version": "2.0.0",
        "trained_at": datetime.now().isoformat(),
        "n_models": 7,
        "models": {},
    }

    for model_key, ModelClass in MODELS.items():
        logger.info(f"\n{'='*50}\nEntrenando {model_key}\n{'='*50}")
        model = ModelClass()

        # Subset de datos reales para este modelo
        if arribo_df is not None and "MODELO" in arribo_df.columns:
            real_sub = arribo_df[arribo_df["MODELO"] == model_key].copy()
        else:
            real_sub = pd.DataFrame()

        # Combinar con sintético según cantidad de datos reales
        train_df = merge_real_and_synthetic(real_sub, model_key,
                                             n_synthetic=n_synthetic)

        # Cargar demanda de validaciones si aplica
        if model_key == "M1_METROPOLITANO" and cosac_df is not None:
            model.load_demand(cosac_df)

        if model_key == "M6_CORREDOR" and corredores_df is not None:
            model.load_demand(corredores_df)
            # Enriquecer train_df con features de corredor
            if "EJE_NUM" not in train_df.columns:
                train_df["EJE_NUM"] = 0
            if "SENTIDO_NUM" not in train_df.columns:
                train_df["SENTIDO_NUM"] = 0

        if model_key == "M7_METRO" and metro_df is not None:
            model.load_demand(metro_df)

        model.fit(train_df)

        pkl_path = str(out / f"{model_key.lower()}.pkl")
        model.save(pkl_path)

        meta["models"][model_key] = {
            "pkl": f"{model_key.lower()}.pkl",
            "n_real": len(real_sub),
            "n_train": len(train_df),
            "is_fitted": model.is_fitted,
            "eta_mae": getattr(model, "_eta_mae", getattr(model, "eta_mae", None)),
            "od_mae":  getattr(model, "_od_mae",  getattr(model, "od_mae",  None)),
        }
        trained_models[model_key] = model
        logger.info(f"✓ {model_key} guardado en {pkl_path}")

    # ── 4. Metadata ───────────────────────────────────────────────────────────
    with open(out / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"\n✓ Metadata guardada en {out / 'model_metadata.json'}")

    logger.info(f"\n{'='*50}")
    logger.info(f"ENTRENAMIENTO COMPLETO — {len(trained_models)} modelos en {out}")
    for k, v in meta["models"].items():
        eta_mae = v["eta_mae"]
        logger.info(f"  {k}: n_real={v['n_real']} | n_train={v['n_train']} | ETA_MAE={eta_mae}")
    logger.info(f"{'='*50}\n")
    return trained_models


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento 7 modelos ATU")
    parser.add_argument("--arribo",     default=None, help="Ruta al Arribo S22 .xlsx")
    parser.add_argument("--cosac",      default=None, help="Ruta a cosac_detalle.csv")
    parser.add_argument("--metro",      default=None, help="Ruta a linea1_detalle.csv")
    parser.add_argument("--corredores", default=None, help="Ruta a corredoresC.csv")
    parser.add_argument("--output",     default="./trained")
    parser.add_argument("--n_synthetic",type=int, default=4000)
    args = parser.parse_args()
    main(
        arribo_path=args.arribo,
        cosac_path=args.cosac,
        metro_path=args.metro,
        corredores_path=args.corredores,
        output_dir=args.output,
        n_synthetic=args.n_synthetic,
    )
