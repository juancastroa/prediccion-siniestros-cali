"""Genera el pronostico diario y los datos del dashboard de siniestros viales."""

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

ORDEN_ARIMA = (4, 1, 1)
DIAS_ENTRENAMIENTO_MIN = 365
DIAS_EVALUACION_RECIENTE = 90
DIAS_RECIENTES = 30
INICIO_TEST_OFICIAL = pd.Timestamp("2024-01-01")
FIN_TEST_OFICIAL = pd.Timestamp("2024-11-02")


def cargar_serie_base():
    """Carga la serie diaria historica ya agregada."""
    archivo_base = HISTORICO_DIR / "serie_diaria_historica.csv"
    if not archivo_base.exists():
        raise SystemExit(f"No se encontro {archivo_base}")

    base = pd.read_csv(archivo_base)
    base["fecha"] = pd.to_datetime(base["fecha"])
    return base.set_index("fecha")["total_siniestros"]


def cargar_conteos_nuevos():
    """Agrega los archivos diarios opcionales recibidos en data/nuevos."""
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
    """Combina base y novedades, incluyendo dias sin siniestros como cero."""
    base = cargar_serie_base()
    nuevos = cargar_conteos_nuevos()
    combinada = nuevos.combine_first(base).sort_index()
    rango_completo = pd.date_range(combinada.index.min(), combinada.index.max(), freq="D")
    combinada = combinada.reindex(rango_completo, fill_value=0)
    combinada.index.name = "fecha"
    return combinada.to_frame("total_siniestros")


def ejecutar_walk_forward(entrenamiento, evaluacion):
    """Pronostica cada fecha una sola vez antes de incorporar su valor real."""
    modelo = ARIMA(
        entrenamiento.astype(float),
        order=ORDEN_ARIMA,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    resultado = modelo.fit()

    # El filtro de Kalman entrega el pronostico de un paso para cada fecha
    # usando solo observaciones anteriores. Es equivalente a actualizar el
    # estado fecha por fecha, pero evita volver a construir el modelo cientos
    # de veces durante la actualizacion diaria.
    completa = pd.concat([entrenamiento.astype(float), evaluacion.astype(float)])
    resultado_completo = resultado.apply(completa, refit=False)
    inicio = len(entrenamiento)
    predicciones = resultado_completo.predict(start=inicio, end=inicio + len(evaluacion) - 1)

    return [
        {
            "fecha": fecha.strftime("%Y-%m-%d"),
            "real": int(valor_real),
            "prediccion": float(prediccion),
            "error_absoluto": abs(float(valor_real) - float(prediccion)),
        }
        for (fecha, valor_real), prediccion in zip(evaluacion.astype(float).items(), predicciones)
    ]


def resumir_evaluacion(filas, baseline):
    """Calcula metricas de una evaluacion sin usar etiquetas de accuracy."""
    reales = np.array([fila["real"] for fila in filas], dtype=float)
    predicciones = np.array([fila["prediccion"] for fila in filas], dtype=float)
    errores = np.abs(reales - predicciones)
    baseline = baseline.astype(float).to_numpy()
    errores_baseline = np.abs(reales - baseline)
    mae = float(np.mean(errores))
    mae_baseline = float(np.mean(errores_baseline))

    return {
        "dias": len(filas),
        "arima": {
            "mae": mae,
            "rmse": float(np.sqrt(np.mean((reales - predicciones) ** 2))),
        },
        "baseline_semana_anterior": {
            "mae": mae_baseline,
            "rmse": float(np.sqrt(np.mean(errores_baseline**2))),
        },
        "reduccion_mae_pct": 100 * (mae_baseline - mae) / mae_baseline,
        "porcentaje_error_absoluto_hasta": [
            {"limite": limite, "porcentaje": float(np.mean(errores <= limite) * 100)}
            for limite in range(1, 6)
        ],
        "error_absoluto_p80": float(np.percentile(errores, 80)),
    }


def evaluar_test_oficial(serie):
    """Evalua exclusivamente del 2024-01-01 al 2024-11-02 (307 dias)."""
    if serie.index.max() < FIN_TEST_OFICIAL:
        raise SystemExit("La serie no contiene todo el periodo oficial hasta 2024-11-02.")

    entrenamiento = serie.loc[: INICIO_TEST_OFICIAL - pd.Timedelta(days=1), "total_siniestros"]
    evaluacion = serie.loc[INICIO_TEST_OFICIAL:FIN_TEST_OFICIAL, "total_siniestros"]
    if len(evaluacion) != 307:
        raise SystemExit(f"El test oficial debe tener 307 dias y tiene {len(evaluacion)}.")

    filas = ejecutar_walk_forward(entrenamiento, evaluacion)
    # Usa el valor observado siete dias antes, tambien para el inicio del test.
    baseline = serie["total_siniestros"].shift(7).loc[evaluacion.index]
    metricas = resumir_evaluacion(filas, baseline)
    metricas.update(
        {
            "periodo_inicio": INICIO_TEST_OFICIAL.strftime("%Y-%m-%d"),
            "periodo_fin": FIN_TEST_OFICIAL.strftime("%Y-%m-%d"),
            "predicciones": filas,
        }
    )
    return metricas


def evaluar_periodo_reciente(serie, dias=DIAS_EVALUACION_RECIENTE):
    """Mide el comportamiento reciente sin mezclarlo con el test oficial."""
    if len(serie) < DIAS_ENTRENAMIENTO_MIN + dias:
        return None

    entrenamiento = serie["total_siniestros"].iloc[:-dias]
    evaluacion = serie["total_siniestros"].iloc[-dias:]
    filas = ejecutar_walk_forward(entrenamiento, evaluacion)
    baseline = serie["total_siniestros"].shift(7).loc[evaluacion.index]
    metricas = resumir_evaluacion(filas, baseline)
    metricas.update(
        {
            "periodo_inicio": evaluacion.index.min().strftime("%Y-%m-%d"),
            "periodo_fin": evaluacion.index.max().strftime("%Y-%m-%d"),
        }
    )
    # La tabla y la grafica corresponden solo al test oficial fijo.
    metricas.pop("porcentaje_error_absoluto_hasta")
    return metricas


def construir_historico(serie, prediccion):
    """Prepara serie y agregados para las vistas historicas del dashboard."""
    diario = serie["total_siniestros"]
    reciente = diario.tail(DIAS_RECIENTES)
    por_anio = diario.groupby(diario.index.year).agg(["sum", "mean"])
    dias_semana = ["Lunes", "Martes", "Miercoles", "Jueves", "Viernes", "Sabado", "Domingo"]
    por_dia_semana = diario.groupby(diario.index.dayofweek).mean()
    p25, p75 = np.percentile(reciente.to_numpy(), [25, 75])
    estado = "normal"
    if prediccion < p25:
        estado = "bajo"
    elif prediccion > p75:
        estado = "alto"

    primeros_15 = float(reciente.iloc[:15].mean())
    ultimos_15 = float(reciente.iloc[-15:].mean())
    return {
        "serie_diaria": [
            {"fecha": fecha.strftime("%Y-%m-%d"), "total": int(total)}
            for fecha, total in diario.items()
        ],
        "por_anio": [
            {"anio": int(anio), "total": int(fila["sum"]), "promedio_diario": float(fila["mean"])}
            for anio, fila in por_anio.iterrows()
        ],
        "por_dia_semana": [
            {"dia": dias_semana[indice], "promedio": float(valor)}
            for indice, valor in por_dia_semana.items()
        ],
        "ultimos_30_dias": [
            {"fecha": fecha.strftime("%Y-%m-%d"), "total": int(total)}
            for fecha, total in reciente.items()
        ],
        "promedio_30_dias": float(reciente.mean()),
        "tendencia_30_dias": {
            "promedio_primeros_15_dias": primeros_15,
            "promedio_ultimos_15_dias": ultimos_15,
            "variacion_promedio": ultimos_15 - primeros_15,
        },
        "clasificacion_pronostico": {
            "estado": estado,
            "percentil_25": float(p25),
            "percentil_75": float(p75),
        },
    }


def generar():
    serie = construir_serie_diaria()
    if len(serie) < DIAS_ENTRENAMIENTO_MIN:
        raise SystemExit(f"Solo hay {len(serie)} dias de datos; se necesitan {DIAS_ENTRENAMIENTO_MIN}.")

    test_oficial = evaluar_test_oficial(serie)
    evaluacion_reciente = evaluar_periodo_reciente(serie)
    serie_completa = serie["total_siniestros"].astype(float)
    ultima_fecha = serie.index.max()
    fecha_pronostico = ultima_fecha + pd.Timedelta(days=1)

    ajuste_final = ARIMA(
        serie_completa,
        order=ORDEN_ARIMA,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit()
    prediccion_central = float(ajuste_final.forecast(steps=1).iloc[0])
    error_p80 = evaluacion_reciente["error_absoluto_p80"] if evaluacion_reciente else test_oficial["error_absoluto_p80"]
    rango_min = max(0, round(prediccion_central - error_p80))
    rango_max = round(prediccion_central + error_p80)
    historico = construir_historico(serie, prediccion_central)

    salida = {
        "modelo": {"nombre": "ARIMA", "orden": [4, 1, 1], "horizonte_dias": 1},
        "actualizado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "datos_disponibles_hasta": ultima_fecha.strftime("%Y-%m-%d"),
        "pronostico": {
            "fecha": fecha_pronostico.strftime("%Y-%m-%d"),
            "prediccion_central": prediccion_central,
            "prediccion_redondeada": round(prediccion_central),
            "rango_operativo": {
                "min": rango_min,
                "max": rango_max,
                "percentil_error_absoluto": 80,
                "error_absoluto_percentil": error_p80,
            },
        },
        "evaluacion_oficial_2024": test_oficial,
        "evaluacion_reciente_90_dias": evaluacion_reciente,
        "historico": historico,
    }

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text(json.dumps(salida, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Pronostico para {salida['pronostico']['fecha']}: {prediccion_central:.2f} ({round(prediccion_central)})")
    print(f"Test oficial: {test_oficial['dias']} dias, MAE {test_oficial['arima']['mae']:.3f}, RMSE {test_oficial['arima']['rmse']:.3f}")
    print(f"Guardado en {SALIDA}")


if __name__ == "__main__":
    generar()
