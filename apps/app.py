"""
App de priorización de procesos para automatización con IA.

Dos funciones principales:
1. Explorar el catálogo ya priorizado, con filtros y visualizaciones.
2. Evaluar un proceso nuevo (mismos campos que la encuesta de Forms) y ver
   dónde quedaría clasificado dentro de la lista completa.
"""

import plotly.express as px
import streamlit as st

import logic

st.set_page_config(
    page_title="Priorización de procesos con IA",
    page_icon="📊",
    layout="wide",
)


# ---------------------------------------------------------------- carga (cacheada)
@st.cache_resource
def _cargar_modelo_y_encoder():
    modelo, encoder, _ = logic.cargar_artefactos()
    return modelo, encoder


@st.cache_data
def _cargar_catalogo():
    _, _, catalogo = logic.cargar_artefactos()
    return catalogo


modelo, encoder = _cargar_modelo_y_encoder()
catalogo = _cargar_catalogo()
DEPARTAMENTOS, PROCESOS = logic.opciones_categoricas(encoder)

ETIQUETAS_EXITO = {0: "No exitoso", 1: "Exitoso"}

st.title("📊 Priorización de procesos para automatización con IA")
st.caption(
    "Catálogo de procesos puntuado con un modelo de Random Forest que predice "
    "`implementation_success`, combinado con el beneficio anual estimado en un "
    "`priority_score`."
)

tab_catalogo, tab_formulario = st.tabs(["📋 Catálogo priorizado", "🆕 Evaluar proceso nuevo"])


# ================================================================== TAB 1
with tab_catalogo:
    st.subheader("Filtros")
    col_f1, col_f2, col_f3, col_f4 = st.columns([1, 1.4, 1.4, 1])

    with col_f1:
        filtro_exito = st.selectbox("Resultado", ["Todos", "Exitoso", "No exitoso"])
    with col_f2:
        filtro_departamentos = st.multiselect("Departamento", DEPARTAMENTOS)
    with col_f3:
        filtro_procesos = st.multiselect("Nombre del proceso", PROCESOS)
    with col_f4:
        opcion_top = st.selectbox("Mostrar", ["Top 10", "Top 50", "Top 100", "Todos"], index=0)

    top_n = None if opcion_top == "Todos" else int(opcion_top.split()[1])
    df_filtrado = logic.filtrar_catalogo(
        catalogo, filtro_exito, filtro_departamentos, filtro_procesos, top_n
    )

    st.markdown(f"**{len(df_filtrado)}** procesos en la selección actual (de {len(catalogo)} en total).")

    # --------------------------- tabla
    tabla = (
        df_filtrado[logic.COLUMNAS_TABLA]
        .assign(
            implementation_success=lambda d: d["implementation_success"].map(ETIQUETAS_EXITO),
            manual_share=lambda d: (d["manual_share"] * 100).round(0),
        )
        .rename(columns={
            "process_id": "ID", "department": "Departamento", "process_name": "Proceso",
            "monthly_volume": "Volumen/mes", "avg_minutes": "Min. prom.",
            "manual_share": "% manual", "data_availability": "Disp. datos",
            "process_complexity": "Complejidad", "estimated_annual_benefit": "Beneficio anual (COP)",
            "implementation_success": "Resultado", "success_probability": "Prob. éxito",
            "priority_score": "Priority score",
        })
    )
    st.dataframe(
        tabla,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Beneficio anual (COP)": st.column_config.NumberColumn(format="$%,.0f"),
            "Prob. éxito": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
            "Priority score": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f"),
            "% manual": st.column_config.NumberColumn(format="%.0f%%"),
        },
    )

    st.download_button(
        "⬇️ Descargar selección (CSV)",
        df_filtrado.to_csv(index=False).encode("utf-8"),
        file_name="catalogo_filtrado.csv",
        mime="text/csv",
    )

    # --------------------------- visualizaciones
    st.subheader("Visualizaciones de la selección actual")

    if df_filtrado.empty:
        st.info("No hay procesos que cumplan estos filtros.")
    else:
        col_v1, col_v2 = st.columns(2)

        with col_v1:
            beneficio_por_depto = (
                df_filtrado.groupby("department", as_index=False)["estimated_annual_benefit"]
                .sum().sort_values("estimated_annual_benefit", ascending=True)
            )
            fig_beneficio = px.bar(
                beneficio_por_depto, x="estimated_annual_benefit", y="department", orientation="h",
                labels={"estimated_annual_benefit": "Beneficio anual estimado (COP)", "department": "Departamento"},
                title="Beneficio anual estimado por departamento",
            )
            fig_beneficio.update_layout(yaxis_title=None)
            st.plotly_chart(fig_beneficio, use_container_width=True)

        with col_v2:
            conteo_depto = df_filtrado["department"].value_counts().reset_index()
            conteo_depto.columns = ["department", "procesos"]
            fig_conteo = px.pie(
                conteo_depto, names="department", values="procesos", hole=0.45,
                title="Procesos por departamento",
            )
            st.plotly_chart(fig_conteo, use_container_width=True)

        fig_dispersión = px.scatter(
            df_filtrado, x="success_probability", y="estimated_annual_benefit",
            color="priority_score", size="monthly_volume", hover_name="process_id",
            hover_data={"department": True, "process_name": True},
            labels={
                "success_probability": "Probabilidad de éxito",
                "estimated_annual_benefit": "Beneficio anual estimado (COP)",
                "priority_score": "Priority score",
            },
            color_continuous_scale="viridis",
            title="Probabilidad de éxito vs. beneficio anual estimado",
        )
        st.plotly_chart(fig_dispersión, use_container_width=True)


# ================================================================== TAB 2
with tab_formulario:
    st.subheader("Evaluar un proceso nuevo")
    st.caption(
        "Ingresa los mismos datos que se piden en la encuesta de Forms. "
        "El modelo calcula la probabilidad de éxito y el `priority_score`, "
        "y muestra en qué posición quedaría dentro del catálogo completo."
    )

    with st.form("formulario_proceso_nuevo"):
        col1, col2 = st.columns(2)
        with col1:
            department = st.selectbox("Departamento", DEPARTAMENTOS)
            process_name = st.selectbox("Tipo de proceso", PROCESOS)
            monthly_volume = st.number_input(
                "¿Cuántas veces se ejecuta el proceso al mes?", min_value=1, value=200, step=1
            )
            avg_minutes = st.number_input(
                "¿Cuántos minutos toma cada ejecución?", min_value=0.1, value=20.0, step=0.5
            )
        with col2:
            manual_share = st.slider(
                "¿Qué porcentaje del proceso se realiza manualmente?", 0, 100, 70
            )
            data_availability = st.select_slider(
                "¿Qué tan disponible y estructurada está la información necesaria? (1 = baja, 5 = alta)",
                options=[1, 2, 3, 4, 5], value=3,
            )
            process_complexity = st.select_slider(
                "¿Qué tan complejo es el proceso? (1 = baja, 5 = alta)",
                options=[1, 2, 3, 4, 5], value=3,
            )

        enviado = st.form_submit_button("🔮 Predecir y priorizar", use_container_width=True)

    if enviado:
        datos = dict(
            department=department, process_name=process_name,
            monthly_volume=monthly_volume, avg_minutes=avg_minutes,
            manual_share=manual_share, data_availability=data_availability,
            process_complexity=process_complexity,
        )
        errores = logic.validar_entrada_formulario(datos)

        if errores:
            for e in errores:
                st.error(e)
        else:
            resultado, catalogo_con_nuevo = logic.predecir_nuevo_proceso(
                datos, modelo, encoder, catalogo
            )

            es_exitoso = resultado["implementation_success"] == 1
            st.divider()
            col_r1, col_r2, col_r3 = st.columns(3)
            col_r1.metric(
                "Resultado predicho",
                "✅ Exitoso" if es_exitoso else "⚠️ No exitoso",
            )
            col_r2.metric("Probabilidad de éxito", f"{resultado['success_probability']:.1%}")
            col_r3.metric(
                "Posición en el catálogo",
                f"{resultado['posicion']} / {resultado['total']}",
            )

            col_r4, col_r5 = st.columns(2)
            col_r4.metric("Priority score", f"{resultado['priority_score']:.3f}")
            col_r5.metric(
                "Beneficio anual estimado",
                f"${resultado['estimated_annual_benefit']:,.0f} COP",
            )

            st.caption(
                "El `priority_score` se recalculó sobre el catálogo completo + este proceso, "
                "así que la normalización del beneficio puede moverse ligeramente si este "
                "proceso queda fuera del rango que ya tenían los demás."
            )

            # ubicar visualmente el proceso nuevo dentro de la distribución completa
            catalogo_con_nuevo["es_nuevo"] = catalogo_con_nuevo["process_id"] == "NUEVO"
            fig_ubicacion = px.scatter(
                catalogo_con_nuevo, x="success_probability", y="estimated_annual_benefit",
                color="es_nuevo", size="monthly_volume",
                color_discrete_map={True: "#C44E52", False: "#4C72B0"},
                labels={
                    "success_probability": "Probabilidad de éxito",
                    "estimated_annual_benefit": "Beneficio anual estimado (COP)",
                    "es_nuevo": "Proceso nuevo",
                },
                title="Dónde queda el proceso nuevo frente al resto del catálogo",
            )
            st.plotly_chart(fig_ubicacion, use_container_width=True)
