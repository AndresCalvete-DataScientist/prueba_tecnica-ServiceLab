# Priorización de procesos con IA — app de Streamlit

App con dos funciones:

1. **Catálogo priorizado** — explora `catalog_priorizado.csv` con filtros (éxito/fracaso,
   departamento, proceso, Top N) y tres visualizaciones en Plotly.
2. **Evaluar proceso nuevo** — un formulario con los mismos campos de la encuesta de
   Forms; al enviarlo, predice `implementation_success` y `success_probability` con el
   modelo, calcula su `priority_score` y muestra en qué posición quedaría dentro del
   catálogo completo.

## Archivos del proyecto

```
app.py                              # interfaz (Streamlit + Plotly)
logic.py                            # lógica pura (carga, features, predicción, filtros)
requirements.txt
modelo_implementation_success.joblib   # el Random Forest ya entrenado
encoder_categorico.joblib              # OrdinalEncoder de department/process_name,
                                        # ajustado sobre historical (no venía guardado
                                        # junto al modelo, así que se generó aparte)

```

## Probarla en tu computador

```bash
python -m venv venv
source venv/bin/activate          # en Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```