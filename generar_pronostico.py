"""
Genera el pronostico de siniestros viales para el dia siguiente.

Que hace:
1. Carga el historico base (data/historico/dataset_siniestros_cali_limpio.csv)
2. Agrega cualquier archivo nuevo depositado en data/nuevos/ (CSV o Excel,
   con al menos una columna 'fecha' y una fila por siniestro)
3. Reconstruye la serie diaria (numero de siniestros por dia)
4. Entrena ARIMA(4,1,1) -- el orden que dio mejor resultado en la
   validacion del notebook original -- y pronostica el dia siguiente
5. Calcula un rango operativo al 80% usando los errores historicos del
   modelo, y escribe todo en docs/prediccion.json para que el dashboard
   (docs/index.html) lo lea

Se pensó para correr una vez al día vía GitHub Actions, pero funciona
igual ejecutándolo a mano: `python scripts/generar_pronostico.py`
"""

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

RAIZ = Path(__file__).resolve().parent.parent
HISTORICO_DIR = RAIZ / "data" / "historico"
NUEVOS_DIR = RAIZ / "data" / "nuevos"
SALIDA = RAIZ / "docs" / "prediccion.json"

ORDEN_ARIMA = (4, 1, 1)          # mismo orden validado en el notebook
DIAS_ENTRENAMIENTO_MIN = 365     # no tiene sentido pronosticar con menos de un año
DIAS_HISTORICO_EN_JSON = 30      # cuantos dias recientes mandamos al dashboard


def cargar_serie_base():
    """
    Carga la serie diaria historica ya agregada (una fila por dia, no por
    siniestro). Se guarda asi -- y no como el detalle crudo -- para que el
    archivo pese poco y quepa comodo en GitHub.
    """
    archivo_base = HISTORICO_DIR / "serie_diaria_historica.csv"
    if not archivo_base.exists():
        raise SystemExit(f"No se encontro {archivo_base}")

    base = pd.read_csv(archivo_base)
    base["fecha"] = pd.to_datetime(base["fecha"])
    return base.set_index("fecha")["total_siniestros"]


def cargar_conteos_nuevos():
    """
    Lee cada archivo en data/nuevos/ (detalle: una fila por siniestro,
    igual que llega del reporte diario) y devuelve el conteo por fecha.
    """
    partes = []
    for archivo in sorted(NUEVOS_DIR.glob("*")):
        if archivo.suffix.lower() == ".csv":
            partes.append(pd.read_csv(archivo, low_memory=False))
        elif archivo.suffix.lower() in (".xlsx", ".xls"):
            partes.append(pd.read_excel(archivo))

    if not partes:
        return pd.Series(dtype="int64")

    df = pd.concat(partes, ignore_index=True)
    if "codigo" in df.columns:
        df = df.drop_duplicates(subset="codigo")

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.dropna(subset=["fecha"])

    return df.groupby(df["fecha"].dt.normalize()).size()


def construir_serie_diaria():
    """Combina la serie base con los conteos nuevos y rellena huecos con 0."""
    base = cargar_serie_base()
    nuevos = cargar_conteos_nuevos()

    # Los dias que llegan en data/nuevos/ actualizan/extienden la serie base
    combinada = nuevos.combine_first(base).sort_index()

    rango_completo = pd.date_range(combinada.index.min(), combinada.index.max(), freq="D")
    combinada = combinada.reindex(rango_completo, fill_value=0)
    combinada.index.name = "fecha"

    return combinada.to_frame("total_siniestros")


def calcular_rango_operativo(serie, orden, dias_evaluacion=90):
    """
    Mide el error del modelo en los ultimos `dias_evaluacion` dias
    (walk-forward: predice un dia, incorpora el valor real, sigue) para
    calibrar un rango del 80% que sea representativo de la situacion actual.
    """
    if len(serie) < DIAS_ENTRENAMIENTO_MIN + dias_evaluacion:
        return None, None, None

    entrenamiento = serie["total_siniestros"].iloc[: -dias_evaluacion].astype(float)
    evaluacion = serie["total_siniestros"].iloc[-dias_evaluacion:].astype(float)

    modelo = ARIMA(entrenamiento, order=orden, enforce_stationarity=False, enforce_invertibility=False)
    ajuste = modelo.fit()

    predicciones = []
    resultado_actual = ajuste
    for valor_real in evaluacion:
        pred = resultado_actual.forecast(steps=1).iloc[0]
        predicciones.append(pred)
        resultado_actual = resultado_actual.append([valor_real], refit=False)

    errores = np.abs(evaluacion.values - np.array(predicciones))
    mae = float(np.mean(errores))
    error_p80 = float(np.percentile(errores, 80))

    baseline = evaluacion.shift(7).dropna()
    real_baseline = evaluacion.loc[baseline.index]
    mae_baseline = float(np.mean(np.abs(real_baseline.values - baseline.values))) if len(baseline) else None
    mejora_pct = round(100 * (mae_baseline - mae) / mae_baseline, 1) if mae_baseline else None

    return mae, error_p80, mejora_pct


def generar():
    serie = construir_serie_diaria()

    if len(serie) < DIAS_ENTRENAMIENTO_MIN:
        raise SystemExit(
            f"Solo hay {len(serie)} dias de datos; se necesitan al menos "
            f"{DIAS_ENTRENAMIENTO_MIN} para entrenar el modelo."
        )

    serie_completa = serie["total_siniestros"].astype(float)
    ultima_fecha = serie.index.max()
    fecha_pronostico = ultima_fecha + pd.Timedelta(days=1)

    modelo_final = ARIMA(
        serie_completa, order=ORDEN_ARIMA,
        enforce_stationarity=False, enforce_invertibility=False,
    )
    ajuste_final = modelo_final.fit()
    forecast = ajuste_final.get_forecast(steps=1)
    prediccion = float(forecast.predicted_mean.iloc[0])
    intervalo_95 = forecast.conf_int(alpha=0.05)
    lim_inf_95 = max(0.0, float(intervalo_95.iloc[0, 0]))
    lim_sup_95 = float(intervalo_95.iloc[0, 1])

    mae, error_p80, mejora_pct = calcular_rango_operativo(serie, ORDEN_ARIMA)
    if error_p80 is not None:
        rango_min = max(0, round(prediccion - error_p80))
        rango_max = round(prediccion + error_p80)
    else:
        # No hay suficiente historial reciente para recalibrar: se usa el 95%
        rango_min, rango_max = round(lim_inf_95), round(lim_sup_95)

    historico_reciente = serie.tail(DIAS_HISTORICO_EN_JSON)
    promedio_30d = round(float(historico_reciente["total_siniestros"].mean()), 1)

    salida = {
        "fecha_pronostico": fecha_pronostico.strftime("%Y-%m-%d"),
        "prediccion": round(prediccion),
        "rango_min": rango_min,
        "rango_max": rango_max,
        "intervalo_95_min": round(lim_inf_95, 1),
        "intervalo_95_max": round(lim_sup_95, 1),
        "mae_modelo": round(mae, 2) if mae is not None else None,
        "mejora_vs_baseline_pct": mejora_pct,
        "promedio_30d": promedio_30d,
        "actualizado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "historico": [
            {"fecha": fecha.strftime("%Y-%m-%d"), "total": int(total)}
            for fecha, total in zip(historico_reciente.index, historico_reciente["total_siniestros"])
        ],
    }

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text(json.dumps(salida, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Pronostico para {salida['fecha_pronostico']}: {salida['prediccion']} siniestros")
    print(f"Guardado en {SALIDA}")


if __name__ == "__main__":
    generar()
