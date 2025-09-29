# =============================
# Librerías
# =============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import random
import string
import io
import contextlib
import re
import subprocess
import json

# =============================
# Conexión con Ollama
# =============================
import ollama

llm = "llama3.2:3b"  # Modelo (se puede cambiar por llama, mistral, qwen, etc.) mistral:7b
subprocess.run(["ollama", "pull", llm], check=False)

# =============================
# Herramienta para leer CSV
# =============================
def load_csv(path: str) -> str:
    global dtf
    try:
        dtf = pd.read_csv(path)

        # Detectar automáticamente la columna de fechas
        datetime_col = None
        for col in dtf.columns:
            if "fecha" in col.lower() or "date" in col.lower() or "time" in col.lower():
                datetime_col = col
                break

        if datetime_col is None:
            return f"Error: no se encontró ninguna columna de fechas en {path}. Columnas: {list(dtf.columns)}"

        # Convertir a datetime y usar como índice
        dtf[datetime_col] = pd.to_datetime(dtf[datetime_col])
        dtf.set_index(datetime_col, inplace=True)

        print(dtf.head())
        return f"Archivo {path} cargado correctamente con {dtf.shape[0]} filas. Índice temporal: {datetime_col}. Columnas: {list(dtf.columns)}"
    except Exception as e:
        return f"Error al cargar el archivo: {e}"

tool_load_csv = {
  'type': 'function',
  'function': {
    'name': 'load_csv',
    'description': 'Carga un archivo CSV con series de tiempo de energía. El DataFrame se llama dtf.',
    'parameters': {
      'type': 'object',
      'required': ['path'],
      'properties': {
        'path': {'type':'string', 'description':'ruta del archivo CSV a cargar'}
      }
    }
  }
}

# =============================
# Herramienta para respuesta final
# =============================
def final_answer(text:str) -> str:
    return text

tool_final_answer = {
  'type': 'function',
  'function': {
    'name': 'final_answer',
    'description': 'Devuelve una respuesta en lenguaje natural al usuario',
    'parameters': {
      'type': 'object',
      'required': ['text'],
      'properties': {
        'text': {'type':'string', 'description':'respuesta en lenguaje natural'}
      }
    }
  }
}

# =============================
# Herramienta para ejecutar código
# =============================
def sanitize_code(code: str) -> str:
    # Forzar uso de dtf en lugar de df
    code = code.replace("df[", "dtf[")

    # Reemplazar funciones sueltas por métodos de pandas
    code = re.sub(r"mean\((dtf\[.+?\])\)", r"\1.mean()", code)
    code = re.sub(r"max\((dtf\[.+?\])\)", r"\1.max()", code)
    code = re.sub(r"min\((dtf\[.+?\])\)", r"\1.min()", code)
    code = re.sub(r"sum\((dtf\[.+?\])\)", r"\1.sum()", code)

    return code

def code_exec(code: str) -> str:
    code = sanitize_code(code)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        try:
            exec(code, globals())
        except Exception as e:
            print(f"Error: {e}")
    return output.getvalue()

tool_code_exec = {
  'type':'function',
  'function':{
    'name': 'code_exec',
    'description': 'Ejecuta código Python. Siempre usar print() para mostrar la salida.',
    'parameters': {
      'type': 'object', 
      'required': ['code'],
      'properties': {
        'code': {'type':'str', 'description':'código Python a ejecutar'},
      }
    }
  }
}


# =============================
# Herramienta para graficar datos
# =============================
def normalize_plot_args(t_inputs):
    """
    Normaliza los argumentos para plot_data y corrige errores comunes del modelo.
    """
    if not isinstance(t_inputs, dict):
        return t_inputs
    
    # Asegurar que "columns" sea lista
    if "columns" in t_inputs:
        if isinstance(t_inputs["columns"], str):
            try:
                # Convierte "['MW']" → ["MW"]
                t_inputs["columns"] = json.loads(t_inputs["columns"].replace("'", '"'))
            except Exception:
                # Si falla, convierte a lista simple
                t_inputs["columns"] = [t_inputs["columns"]]
        elif t_inputs["columns"] is None:
            t_inputs["columns"] = []
    
    return t_inputs


def plot_data(columns=None, start_date=None, end_date=None, title="Gráfico de datos"):
    """
    Genera un gráfico con matplotlib a partir del DataFrame dtf.
    columns: lista de columnas a graficar. Si es None, grafica todas.
    start_date, end_date: rango de fechas opcional.
    title: título del gráfico.
    """
    try:
        df = dtf.copy()

        # Validar columnas
        if columns:
            columnas_validas = [c for c in columns if c in df.columns]
            if not columnas_validas:
                return f"Error: ninguna de las columnas {columns} existe en dtf. Columnas disponibles: {list(df.columns)}"
            df = df[columnas_validas]

        # Filtrar por fechas
        if start_date and end_date:
            if start_date == end_date:
                # Filtrar solo el día exacto
                df = df.loc[start_date]
            else:
                # Filtrar rango de fechas
                df = df.loc[start_date:end_date]
        elif start_date:
            # Filtrar un único día
            df = df.loc[start_date]

        # Graficar
        df.plot(figsize=(12,5), linestyle="--")
        plt.title(title)
        plt.xlabel("Índice (ej: tiempo o filas)")
        plt.ylabel("Valores")
        plt.grid(True)
        plt.show()

        return f"Gráfico generado con columnas {columns or list(dtf.columns)}."
    except Exception as e:
        return f"Error al graficar: {e}"


tool_plot_data = {
  'type': 'function',
  'function': {
    'name': 'plot_data',
    'description': 'Genera un gráfico del dataset dtf usando matplotlib.',
    'parameters': {
      'type': 'object',
      'properties': {
        'columns': {
          'type': 'array',
          'items': {'type': 'string'},
          'description': 'Lista de columnas a graficar. Si no se da, se grafican todas.'
        },
        'start_date': {
          'type':'string',
          'description':'Fecha de inicio en formato YYYY-MM-DD (opcional)'
        },
        'end_date': {
          'type':'string',
          'description':'Fecha de fin en formato YYYY-MM-DD (opcional)'
        },
        'title': {
          'type':'string',
          'description':'Título del gráfico'
        }
      }
    }
  }
}



# =============================
# Diccionario de herramientas
# =============================
dic_tools = {
    "load_csv": load_csv,
    "final_answer": final_answer,
    "code_exec": code_exec,
    "plot_data": plot_data
}

# =============================
# Ejecutor de herramientas
# =============================


def use_tool(agent_res: dict, dic_tools: dict) -> dict:
    msg = agent_res["message"]
    res, t_name, t_inputs = "", "", ""

    if hasattr(msg, "tool_calls") and msg.tool_calls:
        for tool in msg.tool_calls:
            t_name = tool["function"]["name"]
            raw_args = tool["function"]["arguments"]

            # 👇 Parsear inputs de manera segura
            try:
                t_inputs = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except Exception:
                t_inputs = raw_args

            # 👇 Normalizar argumentos de load_csv
            if t_name == "load_csv":
                t_inputs = normalize_csv_args(t_inputs)
                if t_inputs is None:
                    res = "Error: argumentos inválidos para load_csv."
                    print(res)
                    return {"res": res, "tool_used": t_name, "inputs_used": t_inputs}

            # 👇 Normalizar argumentos de plot_data
            if t_name == "plot_data":
                t_inputs = normalize_plot_args(t_inputs)

            # 👇 Mapear casos especiales de final_answer
            if t_name == "final_answer" and isinstance(t_inputs, dict) and "final_answer" in t_inputs:
                t_inputs = {"text": t_inputs["final_answer"]}

            # 🔧 Ejecutar herramienta
            if f := dic_tools.get(t_name):
                print('🔧 >', f"\x1b[1;31m{t_name} -> Inputs: {t_inputs}\x1b[0m")
                try:
                    if isinstance(t_inputs, dict):
                        t_output = f(**t_inputs)
                    else:
                        t_output = f(t_inputs)
                except Exception as e:
                    # Feedback al modelo en caso de error
                    t_output = f"Error ejecutando {t_name}: {e}. Columnas disponibles: {list(dtf.columns) if 'dtf' in globals() else 'No hay dataset cargado'}"
                print(t_output)
                res = t_output
            else:
                print('🤬 >', f"\x1b[1;31m{t_name} -> NotFound\x1b[0m")

    # 👇 Si el mensaje trae contenido normal (texto), devolverlo como respuesta
    if msg.get("content", "") != "":
        res = msg["content"]
        t_name, t_inputs = "", ""

    return {"res": res, "tool_used": t_name, "inputs_used": t_inputs}


def normalize_csv_args(t_inputs):
    """
    Normaliza los argumentos que llegan a load_csv para que siempre terminen
    como {"path": "archivo.csv"} válido.
    Si no se puede normalizar, devuelve None.
    """
    # Caso directo: {"path": "archivo.csv"}
    if isinstance(t_inputs, dict) and "path" in t_inputs and isinstance(t_inputs["path"], str):
        candidate = t_inputs["path"].strip()
        # Caso especial: viene como "{'path': 'datos_limpios.csv'}"
        if candidate.startswith("{") and "path" in candidate:
            try:
                inner = json.loads(candidate.replace("'", '"'))
                return {"path": inner["path"]}
            except Exception:
                return None
        # Caso válido normal
        if candidate not in ["{", "}", ")", ""]:
            return {"path": candidate}
        return None

    # Caso anidado: {"path": {"value": "archivo.csv"}} o similar
    if isinstance(t_inputs, dict) and "path" in t_inputs and isinstance(t_inputs["path"], dict):
        inner = t_inputs["path"]
        if "value" in inner:
            return {"path": inner["value"]}
        if "file_path" in inner:
            return {"path": inner["file_path"]}
        if "path" in inner:
            return {"path": inner["path"]}

    # Caso string plano: "archivo.csv"
    if isinstance(t_inputs, str):
        candidate = t_inputs.strip().strip("{}()")
        if candidate != "":
            return {"path": candidate}
        return None

    # Si no se reconoce → error
    return None





# =============================
# Bucle principal del agente
# =============================
def run_agent(llm, messages, available_tools):
    tool_used, local_memory = '', ''
    used_code_exec = False   # 👈 bandera para saber si ya se usó code_exec

    while tool_used != 'final_answer':
        try:
            agent_res = ollama.chat(
                model=llm, 
                messages=messages, 
                format="json", 
                tools=[v for v in available_tools.values()]
            )

            dic_res = use_tool(agent_res, dic_tools)
            res, tool_used, inputs_used = dic_res["res"], dic_res["tool_used"], dic_res["inputs_used"]

            # 👀 Marcar cuando se use code_exec
            if tool_used == "code_exec":
                used_code_exec = True  

            # 🚨 Evitar final_answer sin code_exec antes
            if tool_used == "final_answer" and not used_code_exec:
                print("⚠️ > El modelo intentó responder sin calcular. Reintentando con code_exec...")
                messages.append({
                    "role": "user", 
                    "content": "Debes usar code_exec para calcular antes de dar la respuesta final."
                })
                tool_used = ""  # 👈 forzar otra vuelta
                continue

        except Exception as e:
            print("⚠️ >", e)
            res = f"Intenté usar {tool_used} pero falló. Intentaré otra cosa."
            messages.append({"role":"assistant", "content":res})

        # 👇 memoria local de uso de herramientas
        if tool_used not in ['','final_answer']:
            local_memory += f"\nTool used: {tool_used}.\nInput used: {inputs_used}.\nOutput: {res}"
            messages.append({"role":"assistant", "content":local_memory})
            available_tools.pop(tool_used, None)
            if len(available_tools) == 1:
                messages.append({"role":"user", "content":"ahora activa la herramienta final_answer."})

        if tool_used == '':
            break

    return res

# =============================
prompt = '''
Eres un Analista de Datos especializado en series de tiempo o tabulares.
Tu tarea es responder cualquier consulta del usuario sobre el dataset `dtf`.

📌 Reglas del dataset:
- El dataset se llama siempre dtf.
- Las columnas y datos pueden variar según el CSV cargado.
- Antes de hacer cálculos o gráficos, identifica las columnas disponibles en dtf (usa dtf.columns) y utiliza exactamente esos nombres.
- Usa métodos de pandas para cálculos, transformaciones y filtrados.
- Usa comillas dobles para los nombres de columnas (ejemplo: dtf["columna"]).
- Siempre muestra los resultados con print() (excepto gráficos).
- No inventes columnas ni datasets.
- Nunca uses pd.read_csv() dentro de code_exec; el dataset ya está cargado en dtf.

📌 Reglas de uso de herramientas:
- Para cálculos, estadísticas, transformaciones o filtrados:
  1. Usa code_exec para ejecutar el cálculo en Python.
  2. Después de code_exec, usa final_answer para explicar el resultado en lenguaje natural.
- Para gráficos: usa exclusivamente la herramienta plot_data.
- Nunca uses final_answer directamente para preguntas de cálculo o gráficos; siempre después de code_exec o plot_data.
- Si no hay datos cargados, indica que debe usarse load_csv.
- Todo bloque de código debe ser sintácticamente completo y ejecutable en Python.
- Nunca dejes funciones o paréntesis sin cerrar (ejemplo: plt.title("..."), dtf["columna"]).

📌 Reglas de herramientas (OBLIGATORIAS):
- load_csv: solo para cargar un archivo. Formato: {"path": "archivo.csv"}.
  Ejemplo: {"path": "datos_limpios.csv"}.
  No uses parámetros extra como "code", "value", "file_path".
- code_exec: exclusivamente para ejecutar código Python. Formato: {"code": "..."}.
- final_answer: exclusivamente para dar la respuesta final en lenguaje natural. Formato: {"text": "..."}.
- plot_data: exclusivamente para graficar datos. Formato:
  {"columns": ["col1","col2"], "start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD", "title": "título"}.
  - "columns" siempre debe ser una lista JSON de strings. Ejemplo: ["MW","MW_P"]. Nunca como string "['MW']".
  - Si no se indican columnas, se grafican todas.
  - "start_date" y "end_date" son opcionales (si no se dan, se usa todo el rango disponible).
  - Para un día específico, usar start_date = end_date = "YYYY-MM-DD".
  - "title" debe ser siempre un texto descriptivo del gráfico.

📌 Reglas de gráficos (cuando uses plot_data):
- Usa siempre nombres reales de las columnas en dtf (consulta dtf.columns).
- Si el usuario pide varias columnas: inclúyelas en "columns" como lista.
- Si no especifica columnas: grafica todas.
- El parámetro title debe describir el gráfico claramente (ejemplo: "Consumo de energía en 2024-09-05").
- Nunca generes manualmente código matplotlib en code_exec; los gráficos se hacen solo con plot_data.
- Si el usuario pide un único día (ej: "solo el día 2024-09-06"), debes poner start_date = end_date = "2024-09-06".
- Nunca uses rangos amplios (ej: start_date="2024-09-01", end_date="2024-09-06") si el usuario pidió un único día.


📌 Ejemplos de uso de plot_data:
- {"columns": ["MW"], "start_date": "2024-09-05", "end_date": "2024-09-05", "title": "MW en 2024-09-05"}
- {"columns": ["MW","MW_P"], "start_date": "2024-09-01", "end_date": "2024-09-30", "title": "MW vs MW_P en septiembre 2024"}
- {"columns": [], "title": "Todas las columnas disponibles"}


📌 Reglas específicas para code_exec:
- El parámetro "code" siempre debe ser un bloque de Python **completo y válido**.
- Siempre debe contener una instrucción print(...) correctamente cerrada.
- Nunca generes código incompleto, paréntesis abiertos o comillas sin cerrar.
- Nunca uses múltiples líneas en code_exec, solo una única instrucción.
- Ejemplo válido: {"code": "print(dtf[\"MW\"].max())"}
- Ejemplo inválido: {"code": "print(dtf["}  ❌

'''


messages = [{"role":"system", "content":prompt}]

# =============================
# Chat interactivo
# =============================
while True:
    q = input("🙂 > ")
    if q.lower() == "quit":
        break
    messages.append({"role":"user", "content":q})

    available_tools = {
        "load_csv": tool_load_csv,
        "final_answer": tool_final_answer,
        "code_exec": tool_code_exec,
        "plot_data": tool_plot_data
    }
    res = run_agent(llm, messages, available_tools)

    print("👽 >", f"\x1b[1;30m{res}\x1b[0m")
    messages.append({"role":"assistant", "content":res})
