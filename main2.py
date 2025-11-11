# =============================
# Librerías
# =============================
import pandas as pd
import matplotlib.pyplot as plt
import ast, io, contextlib, re, json, subprocess
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import ollama
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os

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



import os

def predict_data(model="prophet", column=None, horizon=None, end_date=None, save_dir="predicciones"):
    """
    Predice valores de 'column' usando Prophet o SARIMA y guarda automáticamente un CSV con:
    columnas = ['fecha', 'MW_pred'].
    
    Parámetros:
      - model: 'prophet' o 'arima'
      - column: nombre de la columna (ej. 'MW')
      - horizon: días de predicción (int o texto como '2 días'). Si es None y no hay end_date -> 1 día (96 pasos)
      - end_date: fecha final 'YYYY-MM-DD' para predicción hasta ese día
      - save_dir: carpeta donde se guardará el CSV (por defecto 'predicciones')
    """
    try:
        df = dtf.copy()
        if column is None or column not in df.columns:
            return f"Error: debes especificar una columna válida. Columnas disponibles: {list(df.columns)}"

        # Datos base
        df = df[[column]].dropna().reset_index()
        df.columns = ["ds", "y"]
        freq = "15min"
        last_date = df["ds"].max()

        # -------------------------------
        # Determinar horizonte de predicción (pasos)
        # -------------------------------
        steps = 96  # default 1 día
        if horizon:
            if isinstance(horizon, str):
                nums = re.findall(r"\d+", horizon)
                days = int(nums[0]) if nums else 1
                steps = max(1, days * 96)
            elif isinstance(horizon, int):
                # Interpretamos como días
                steps = max(1, horizon * 96)
        if end_date and (not horizon or isinstance(horizon, str) and re.match(r"^\d{4}-\d{2}-\d{2}$", horizon)):
            try:
                target_date = pd.to_datetime(end_date)
                delta = target_date - last_date
                if delta.total_seconds() <= 0:
                    return f"La fecha {end_date} ya está incluida en los datos."
                steps = max(1, int(delta.total_seconds() / (15 * 60)))
            except Exception as e:
                return f"Error interpretando end_date: {e}"

        # -------------------------------
        # Modelos
        # -------------------------------
        model = model.lower()

        if model == "prophet":
            m = Prophet(daily_seasonality=True)
            m.fit(df)

            future = m.make_future_dataframe(periods=steps, freq=freq)
            forecast = m.predict(future)
            forecast_pred = forecast[forecast["ds"] > last_date][["ds", "yhat"]]
            forecast_pred = forecast_pred.rename(columns={"ds": "fecha", "yhat": "MW_pred"})

            # Gráfico solo de predicciones
            plt.figure(figsize=(10, 5))
            plt.plot(forecast_pred["fecha"], forecast_pred["MW_pred"], "o-", label="Predicción (Prophet)")
            plt.title(f"Predicción Prophet para {column} ({steps} pasos, {steps//96} día(s))")
            plt.xlabel("Fecha")
            plt.ylabel(column)
            plt.grid(True)
            plt.legend()
            plt.show()

            out_df = forecast_pred.copy()

        elif model == "arima":
            # Reducimos a últimos ~30 días si hay demasiados datos (memoria)
            df_use = df.tail(96 * 30) if len(df) > 10000 else df.copy()
            df_use.set_index("ds", inplace=True)

            sarimax = SARIMAX(df_use["y"], order=(2, 1, 2), seasonal_order=(1, 0, 1, 96)).fit(disp=False)

            future_dates = pd.date_range(last_date, periods=steps + 1, freq=freq)[1:]
            forecast_vals = sarimax.forecast(steps=steps)
            forecast_df = pd.DataFrame({"fecha": future_dates, "MW_pred": forecast_vals})

            # Gráfico
            plt.figure(figsize=(10, 5))
            plt.plot(forecast_df["fecha"], forecast_df["MW_pred"], "o-", label="Predicción (SARIMA)")
            plt.title(f"Predicción SARIMA para {column} ({steps} pasos, {steps//96} día(s))")
            plt.xlabel("Fecha")
            plt.ylabel(column)
            plt.grid(True)
            plt.legend()
            plt.show()

            out_df = forecast_df.copy()

        else:
            return "Error: modelo no reconocido. Usa 'prophet' o 'arima'."

        # -------------------------------
        # Guardado automático a CSV
        # -------------------------------
        os.makedirs(save_dir, exist_ok=True)
        # Rango para nombre de archivo
        start_str = pd.to_datetime(out_df["fecha"].min()).strftime("%Y%m%d_%H%M")
        end_str   = pd.to_datetime(out_df["fecha"].max()).strftime("%Y%m%d_%H%M")
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

        filename = f"pred_{model}_{column}_{start_str}_to_{end_str}_{timestamp}.csv"
        out_path = os.path.join(save_dir, filename)

        # Guardar solo dos columnas solicitadas
        out_df[["fecha", "MW_pred"]].to_csv(out_path, index=False)

        # Mostrar tabla (en consola) y confirmar ruta
        pd.set_option("display.max_rows", None)
        print("\n📈 Predicciones futuras:\n")
        print(out_df[["fecha", "MW_pred"]].to_string(index=False))
        print(f"\n💾 Guardado en: {out_path}")

        return f"Predicción {model.upper()} completada ({len(out_df)} puntos). Archivo: {out_path}"

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
def normalize_predict_args(t_inputs):
    """
    Corrige y normaliza argumentos comunes en predict_data.
    - Si 'horizon' parece una fecha, lo convierte a 'end_date'.
    - Si no se especifica horizon, usa 96 pasos (1 día de 15 min).
    - Convierte 'horizon': '1 day' → 96.
    - Limita horizon a 288 (máx. 3 días).
    """
    if not isinstance(t_inputs, dict):
        return t_inputs

    # Ignora campos inventados por el modelo
    for k in ["forecast_type", "prediction_type", "days_ahead"]:
        t_inputs.pop(k, None)

    # horizon recibido como fecha → end_date
    if "horizon" in t_inputs and isinstance(t_inputs["horizon"], str):
        if re.match(r"^\d{4}-\d{2}-\d{2}$", t_inputs["horizon"]):
            t_inputs["end_date"] = t_inputs.pop("horizon")
        else:
            # "1 day", "2 días" → convertir a pasos de 96 por día
            num = re.findall(r"\d+", t_inputs["horizon"])
            t_inputs["horizon"] = int(num[0]) * 96 if num else 96

    # date → end_date
    if "date" in t_inputs and "end_date" not in t_inputs:
        t_inputs["end_date"] = t_inputs.pop("date")

    # Si no se especifica horizon, usar 96 pasos (1 día)
    if "horizon" not in t_inputs:
        t_inputs["horizon"] = 96

    # Si horizon es entero pero muy grande, limitar
    if isinstance(t_inputs.get("horizon"), int) and t_inputs["horizon"] > 288:
        print(f"⚠️  Horizon demasiado grande ({t_inputs['horizon']} pasos). Se limitará a 288.")
        t_inputs["horizon"] = 288

    # Normaliza modelo
    if "model" in t_inputs and isinstance(t_inputs["model"], str):
        t_inputs["model"] = t_inputs["model"].lower()

    return t_inputs




def use_tool(agent_res: dict, dic_tools: dict) -> dict:
    """
    Ejecuta las herramientas solicitadas por el modelo.
    - Soporta tanto tool_calls formales como JSON plano en content.
    - Incluye normalización automática de argumentos.
    """
    msg = agent_res["message"]
    res, t_name, t_inputs = "", "", ""

    # ✅ Caso 1: tool_calls formales (cuando Ollama estructura la llamada)
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        for tool in msg.tool_calls:
            t_name = tool["function"]["name"]
            raw_args = tool["function"]["arguments"]

            # Parsear argumentos
            try:
                t_inputs = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except Exception:
                t_inputs = raw_args

            # 🔧 Normalización según la herramienta
            if t_name == "load_csv":
                t_inputs = normalize_csv_args(t_inputs)
            elif t_name == "plot_data":
                t_inputs = normalize_plot_args(t_inputs)
            elif t_name == "code_exec":
                if isinstance(t_inputs, dict):
                    code = t_inputs.get("code", "")
                    t_inputs = {"code": code}
            elif t_name == "predict_data":
                t_inputs = normalize_predict_args(t_inputs)
                if isinstance(t_inputs, dict):
                    t_inputs.setdefault("model", "prophet")
                    t_inputs.setdefault("horizon", 7)
                    if "column" not in t_inputs or t_inputs["column"] not in dtf.columns:
                        t_inputs["column"] = list(dtf.columns)[0]
            elif t_name == "final_answer" and isinstance(t_inputs, dict) and "final_answer" in t_inputs:
                t_inputs = {"text": t_inputs["final_answer"]}

            # Ejecutar herramienta
            if f := dic_tools.get(t_name):
                print(f"🔧 > {t_name} -> Inputs: {t_inputs}")
                try:
                    t_output = f(**t_inputs) if isinstance(t_inputs, dict) else f(t_inputs)
                except Exception as e:
                    cols = list(dtf.columns) if 'dtf' in globals() else 'No hay dataset cargado'
                    t_output = f"Error ejecutando {t_name}: {e}. Columnas disponibles: {cols}"
                print(f"📊 Resultado:\n{t_output}\n")
                res = t_output
            else:
                print(f"🤬 > {t_name} -> NotFound")

    # ✅ Caso 2: JSON plano en msg.content (ej: {"name": ..., "arguments": {...}})
    elif msg.get("content", "") and msg["content"].strip().startswith("{"):
        try:
            tool_call = json.loads(msg["content"])
            t_name = tool_call.get("name", "")
            t_inputs = tool_call.get("arguments", {})

            # Normalizar predict_data si aplica
            if t_name == "predict_data":
                t_inputs = normalize_predict_args(t_inputs)

            if f := dic_tools.get(t_name):
                print(f"🔧 > {t_name} -> Inputs: {t_inputs}")
                res = f(**t_inputs) if isinstance(t_inputs, dict) else f(t_inputs)
                print(f"📊 Resultado:\n{res}\n")
            else:
                res = f"Herramienta {t_name} no encontrada."
        except Exception as e:
            res = f"⚠️ Error al interpretar JSON: {e}\nContenido: {msg.get('content')}"

    # ✅ Caso 3: mensaje normal (texto plano sin tool)
    elif msg.get("content", ""):
        res = msg["content"]
        print(f"💬 {res}")

    return {"res": res, "tool_used": t_name, "inputs_used": t_inputs}


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
