# =============================
# Librerías
# =============================
import pandas as pd
import matplotlib.pyplot as plt
import ast, io, contextlib, re, json, subprocess
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import ollama

# =============================
# Configuración del modelo
# =============================
llm = "qwen2.5:7b"
subprocess.run(["ollama", "pull", llm], check=False)
print(f"🚀 Modelo cargado: {llm}")

# =============================
# Carga directa del dataset
# =============================
DATA_PATH = "datos_limpios.csv"

try:
    dtf = pd.read_csv(DATA_PATH)
    datetime_col = next((col for col in dtf.columns if any(x in col.lower() for x in ["fecha", "date", "hora"])), None)
    if datetime_col is None:
        raise ValueError("No se encontró columna de fecha/hora.")

    dtf[datetime_col] = pd.to_datetime(dtf[datetime_col], errors="coerce")
    dtf.set_index(datetime_col, inplace=True)
    dtf.sort_index(inplace=True)
    dtf = dtf.asfreq("15T")  # frecuencia fija de 15 minutos

    print(f"✅ Dataset '{DATA_PATH}' cargado correctamente ({len(dtf)} filas)")
    print(f"   Índice temporal: {datetime_col} (frecuencia 15 min)")
    print(f"   Columnas disponibles: {list(dtf.columns)}\n")
    print("📋 Primeras filas del dataset:")
    print(dtf.head(), "\n")

except Exception as e:
    print(f"❌ Error al cargar {DATA_PATH}: {e}")
    exit()

# =============================
# Herramientas
# =============================

def final_answer(text: str) -> str:
    return text

def is_valid_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False

def sanitize_code(code: str) -> str:
    code = code.replace("df[", "dtf[")
    if code.count("(") > code.count(")"):
        code += ")" * (code.count("(") - code.count(")"))
    if code.count("[") > code.count("]"):
        code += "]" * (code.count("[") - code.count("]"))
    if code.count('"') % 2 != 0:
        code += '"'
    if code.count("'") % 2 != 0:
        code += "'"
    return code

def code_exec(code: str) -> str:
    output = io.StringIO()
    code = sanitize_code(code.strip())
    if not code.startswith("print(") or not code.endswith(")"):
        return "Error: usa print(...) para mostrar resultados."
    if not is_valid_python(code):
        return "Error: código incompleto o inválido."
    with contextlib.redirect_stdout(output):
        try:
            exec(code, globals())
        except Exception as e:
            print(f"Error: {e}")
    return output.getvalue()

def plot_data(columns=None, start_date=None, end_date=None, title="Gráfico de datos"):
    try:
        df = dtf.copy()
        if columns:
            cols_validas = [c for c in columns if c in df.columns]
            df = df[cols_validas]
        if start_date and end_date:
            df = df.loc[start_date:end_date]
        elif start_date:
            df = df.loc[start_date]
        df.plot(figsize=(12, 5), linestyle="--")
        plt.title(title)
        plt.xlabel("Tiempo (15 min)")
        plt.ylabel("Potencia [MW]")
        plt.grid(True)
        plt.show()
        return f"Gráfico generado con columnas {columns or list(df.columns)}."
    except Exception as e:
        return f"Error al graficar: {e}"

def predict_data(model="prophet", column=None, horizon=1, end_date=None):
    try:
        df = dtf.copy()
        if column not in df.columns:
            return f"Error: columna no válida. Columnas disponibles: {list(df.columns)}"

        df = df[[column]].dropna().reset_index()
        df.columns = ["ds", "y"]
        freq = pd.infer_freq(df["ds"].head(10)) or "15min"
        freq_minutes = int(re.findall(r"\d+", freq)[0]) if "min" in freq else 1440

        last_date = df["ds"].max()

        if end_date:
            target_date = pd.to_datetime(end_date)
            delta_minutes = (target_date - last_date).total_seconds() / 60
            if delta_minutes <= 0:
                return f"La fecha {end_date} ya está dentro del rango de datos."
            steps = int(delta_minutes / freq_minutes)
        else:
            steps = horizon * int(24 * 60 / freq_minutes)

        if model.lower() == "prophet":
            m = Prophet(daily_seasonality=True)
            m.fit(df)
            future = m.make_future_dataframe(periods=steps, freq=freq)
            forecast = m.predict(future)
            forecast_pred = forecast[forecast["ds"] > last_date][["ds", "yhat"]]
            if end_date:
                forecast_pred = forecast_pred[forecast_pred["ds"] <= target_date]

            plt.figure(figsize=(10, 5))
            plt.plot(forecast_pred["ds"], forecast_pred["yhat"], "o-", color="darkorange", label="Predicción (Prophet)")
            plt.title(f"Predicción Prophet para {column}" + (f" hasta {end_date}" if end_date else f" ({horizon} día{'s' if horizon>1 else ''})"))
            plt.xlabel("Fecha")
            plt.ylabel(column)
            plt.grid(True)
            plt.legend()
            plt.show()

            print("\n📈 Predicciones futuras:\n")
            print(forecast_pred.to_string(index=False))
            return f"Predicción Prophet completada ({len(forecast_pred)} puntos mostrados)."

        elif model.lower() == "arima":
            df.set_index("ds", inplace=True)
            model_fit = ARIMA(df["y"], order=(2, 1, 2)).fit()
            future_dates = pd.date_range(last_date, periods=steps+1, freq=freq)[1:]
            forecast = model_fit.forecast(steps=steps)
            forecast_df = pd.DataFrame({"ds": future_dates, "yhat": forecast})
            if end_date:
                forecast_df = forecast_df[forecast_df["ds"] <= target_date]

            plt.figure(figsize=(10, 5))
            plt.plot(forecast_df["ds"], forecast_df["yhat"], "o-", color="steelblue", label="Predicción (ARIMA)")
            plt.title(f"Predicción ARIMA para {column}" + (f" hasta {end_date}" if end_date else f" ({horizon} día{'s' if horizon>1 else ''})"))
            plt.xlabel("Fecha")
            plt.ylabel(column)
            plt.grid(True)
            plt.legend()
            plt.show()

            print("\n📈 Predicciones futuras:\n")
            print(forecast_df.to_string(index=False))
            return f"Predicción ARIMA completada ({len(forecast_df)} puntos mostrados)."

        else:
            return "Error: modelo no reconocido. Usa 'prophet' o 'arima'."
    except Exception as e:
        return f"Error durante la predicción: {e}"

# =============================
# Diccionario de herramientas
# =============================
dic_tools = {
    "final_answer": final_answer,
    "code_exec": code_exec,
    "plot_data": plot_data,
    "predict_data": predict_data
}

# =============================
# Ejecutor de herramientas
# =============================
def use_tool(agent_res, dic_tools):
    msg = agent_res["message"]
    res, t_name = "", ""
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        for tool in msg.tool_calls:
            t_name = tool["function"]["name"]
            raw_args = tool["function"]["arguments"]
            try:
                t_inputs = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except:
                t_inputs = raw_args
            if f := dic_tools.get(t_name):
                try:
                    res = f(**t_inputs) if isinstance(t_inputs, dict) else f(t_inputs)
                except Exception as e:
                    res = f"Error ejecutando {t_name}: {e}"
    elif msg.get("content", ""):
        res = msg["content"]
    return {"res": res, "tool_used": t_name}

# =============================
# Bucle principal
# =============================
prompt = """
Eres un Analista de Datos especializado en series de tiempo eléctricas.

Contexto del dataset:
- El archivo `datos_limpios.csv` ya está cargado en memoria como `dtf`.
- Representa datos reales de consumo eléctrico con frecuencia de 15 minutos.
- Columnas disponibles:
  • `MW`: Potencia eléctrica medida (en megavatios).
  • `MW_P`: Potencia predicha (en megavatios).
- El índice temporal (`fechaHora`) está en formato datetime y define los intervalos de 15 minutos.

Tu objetivo:
Ayudar al usuario a analizar, visualizar y predecir los datos de `dtf`.

Herramientas disponibles:
1. **code_exec** → Ejecuta código Python con `print(...)` (por ejemplo, cálculos, estadísticas, correlaciones).
2. **plot_data** → Genera gráficos (por ejemplo, tendencias diarias o comparaciones MW vs MW_P).
3. **predict_data** → Predice valores futuros usando Prophet o ARIMA (por ejemplo, los próximos días de MW).
4. **final_answer** → Explica resultados o responde en lenguaje natural.

Restricciones:
- No cargues ni reemplaces el dataset (ya está cargado como `dtf`).
- No inventes columnas, datos o archivos.
- Siempre usa los nombres reales de columnas.
- Usa comillas dobles en nombres de columnas, ej: `dtf["MW"]`.
- Nunca uses `pd.read_csv()` dentro de `code_exec`.
- Si el usuario pide una fecha o rango que no existe, informa el error con `final_answer`.

Ejemplos de comportamiento esperado:
- “¿Cuál es el promedio de MW?” → usa `code_exec` con `print(dtf["MW"].mean())`.
- “Grafica el 2024-09-06” → usa `plot_data` con `columns=["MW"]`, `start_date="2024-09-06"`, `end_date="2024-09-06"`.
- “Predice MW para los próximos 2 días con Prophet” → usa `predict_data` con `{"model": "prophet", "column": "MW", "horizon": 2}`.
- “¿Qué columnas tiene el archivo?” → usa solo `final_answer` describiendo las columnas.

Tu prioridad es responder de forma útil, precisa y sin inventar resultados.
"""


messages = [{"role": "system", "content": prompt}]
print("💬 Agente centrado en 'datos_limpios.csv'. Escribe tus consultas o 'quit' para salir.\n")

while True:
    q = input("🙂 > ")
    if q.lower() == "quit":
        break
    messages.append({"role": "user", "content": q})
    available_tools = {
        "final_answer": {"type": "function", "function": {"name": "final_answer", "description": "Respuesta textual"}},
        "code_exec": {"type": "function", "function": {"name": "code_exec", "description": "Ejecuta código con print()"}},
        "plot_data": {"type": "function", "function": {"name": "plot_data", "description": "Grafica columnas"}},
        "predict_data": {"type": "function", "function": {"name": "predict_data", "description": "Predicción con Prophet o ARIMA"}}
    }
    res = ollama.chat(model=llm, messages=messages, tools=[v for v in available_tools.values()], format="json")
    dic_res = use_tool(res, dic_tools)
    print("👽 >", dic_res["res"], "\n")
    messages.append({"role": "assistant", "content": dic_res["res"]})
