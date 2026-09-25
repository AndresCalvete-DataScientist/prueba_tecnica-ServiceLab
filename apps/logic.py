"""
Lógica de negocio de la app — sin nada de Streamlit ni Plotly, para poder
probarla de forma aislada. `app.py` importa estas funciones y se encarga
solo de la interfaz.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- rutas
CARPETA = Path(__file__).parent
RUTA_MODELO = "../models/modelo_implementation_success.joblib"
RUTA_ENCODER = CARPETA / "encoder_categorico.joblib"
RUTA_CATALOGO = CARPETA / "../data/outputs/2B_catalog_priorizado.csv"

# ---------------------------------------------------------------- constantes
AVG_HOURLY_COST = 30_000  # COP por hora, mismo valor usado en el EDA y el entrenamiento
PESO_BENEFICIO = 0.30
PESO_PROBABILIDAD = 0.70

CAMPOS_FORMULARIO = [
    "department", "process_name", "monthly_volume",
    "avg_minutes", "manual_share", "data_availability", "process_complexity",
]

COLUMNAS_TABLA = [
    "process_id", "department", "process_name", "monthly_volume", "avg_minutes",
    "manual_share", "data_availability", "process_complexity",
    "estimated_annual_benefit", "implementation_success", "success_probability",
    "priority_score",
]


# ---------------------------------------------------------------- carga
def cargar_artefactos():
    """Carga el modelo, el encoder y el catálogo ya priorizado. Sin caché aquí:
    la capa de Streamlit (`app.py`) decide cómo cachear estas llamadas."""
    modelo = joblib.load(RUTA_MODELO)
    encoder = joblib.load(RUTA_ENCODER)
    catalogo = pd.read_csv(RUTA_CATALOGO)
    return modelo, encoder, catalogo


def opciones_categoricas(encoder):
    """Departamentos y procesos válidos, en el mismo orden que usa el encoder."""
    departamentos, procesos = encoder.categories_
    return list(departamentos), list(procesos)


# ---------------------------------------------------------------- feature engineering
def calcular_features_derivadas(df):
    """Aplica las fórmulas del EDA. `df` debe tener las 5 columnas base
    (monthly_volume, avg_minutes, manual_share, data_availability, process_complexity).
    `manual_share` se espera ya en escala 0-1.
    """
    df = df.copy()
    df["annual_hours"] = df["monthly_volume"] * df["avg_minutes"] * 12 / 60
    df["automation_potential"] = (
        0.50 * df["manual_share"]
        + 0.30 * (df["data_availability"] / 5)
        + 0.20 * (1 - df["process_complexity"] / 5)
    )
    df["recoverable_hours"] = df["annual_hours"] * df["automation_potential"]
    df["estimated_annual_benefit"] = df["recoverable_hours"] * AVG_HOURLY_COST
    df["feasibility_score"] = (
        0.60 * (df["data_availability"] / 5)
        + 0.40 * (1 - df["process_complexity"] / 5)
    )
    df["log_annual_hours"] = np.log1p(df["annual_hours"])
    return df


def construir_matriz_modelo(df, modelo, encoder):
    """Codifica department/process_name y devuelve las columnas en el orden
    exacto que espera el modelo (`modelo.feature_names_in_`)."""
    df = df.copy()
    df[["department", "process_name"]] = encoder.transform(df[["department", "process_name"]])
    return df[list(modelo.feature_names_in_)]


# ---------------------------------------------------------------- predicción de un proceso nuevo
def validar_entrada_formulario(datos):
    """Valida los 5 campos base antes de predecir. Devuelve una lista de errores (vacía si todo bien)."""
    errores = []
    if not (datos["monthly_volume"] > 0):
        errores.append("monthly_volume debe ser mayor que 0.")
    if not (datos["avg_minutes"] > 0):
        errores.append("avg_minutes debe ser mayor que 0.")
    if not (0 <= datos["manual_share"] <= 100):
        errores.append("manual_share debe estar entre 0 y 100.")
    if datos["data_availability"] not in (1, 2, 3, 4, 5):
        errores.append("data_availability debe ser un entero de 1 a 5.")
    if datos["process_complexity"] not in (1, 2, 3, 4, 5):
        errores.append("process_complexity debe ser un entero de 1 a 5.")
    return errores


def predecir_nuevo_proceso(datos_formulario, modelo, encoder, catalogo):
    """
    datos_formulario: dict con los mismos campos que la encuesta de Forms
    (department, process_name, monthly_volume, avg_minutes,
    manual_share [0-100], data_availability [1-5], process_complexity [1-5]).

    Devuelve (fila_resultado: dict, catalogo_con_nuevo: DataFrame) donde
    catalogo_con_nuevo es el catálogo completo + el proceso nuevo, ya
    priorizado y ordenado — útil para mostrar en qué posición queda.
    """
    fila = pd.DataFrame([datos_formulario]).copy()
    fila["manual_share"] = fila["manual_share"] / 100  # 0-100 -> 0-1, igual que en la encuesta

    fila = calcular_features_derivadas(fila)
    X = construir_matriz_modelo(fila, modelo, encoder)

    probabilidad = float(modelo.predict_proba(X)[:, 1][0])
    prediccion = int(modelo.predict(X)[0])

    fila["process_id"] = "NUEVO"
    fila["implementation_success"] = prediccion
    fila["success_probability"] = probabilidad

    combinado = pd.concat([catalogo, fila], ignore_index=True)
    combinado = calcular_priority_score(combinado)
    combinado = combinado.sort_values("priority_score", ascending=False).reset_index(drop=True)

    posicion = int(combinado.index[combinado["process_id"] == "NUEVO"][0]) + 1
    fila_resultado = combinado.loc[combinado["process_id"] == "NUEVO"].iloc[0].to_dict()
    fila_resultado["posicion"] = posicion
    fila_resultado["total"] = len(combinado)

    return fila_resultado, combinado


# ---------------------------------------------------------------- priority_score
def calcular_priority_score(df):
    """Normaliza estimated_annual_benefit (min-max sobre `df`) y calcula priority_score.
    Se recalcula cada vez que cambia el conjunto (por ejemplo, al añadir un proceso nuevo)
    para que la normalización siga reflejando el rango vigente."""
    df = df.copy()
    beneficio = df["estimated_annual_benefit"]
    rango = beneficio.max() - beneficio.min()
    if rango == 0:
        df["normalized_estimated_annual_benefit"] = 0.0
    else:
        df["normalized_estimated_annual_benefit"] = (beneficio - beneficio.min()) / rango
    df["priority_score"] = (
        PESO_BENEFICIO * df["normalized_estimated_annual_benefit"]
        + PESO_PROBABILIDAD * df["success_probability"]
    )
    return df


# ---------------------------------------------------------------- filtros de la lista
def filtrar_catalogo(catalogo, exito, departamentos, procesos, top_n):
    """
    exito: "Todos" | "Exitoso" | "No exitoso"
    departamentos, procesos: listas (vacías = sin filtrar por esa columna)
    top_n: int o None (None = mostrar todos)
    """
    df = catalogo.copy()
    if exito == "Exitoso":
        df = df[df["implementation_success"] == 1]
    elif exito == "No exitoso":
        df = df[df["implementation_success"] == 0]

    if departamentos:
        df = df[df["department"].isin(departamentos)]
    if procesos:
        df = df[df["process_name"].isin(procesos)]

    df = df.sort_values("priority_score", ascending=False)
    if top_n is not None:
        df = df.head(top_n)
    return df.reset_index(drop=True)
