#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, io, json, contextlib, subprocess
import random, string
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ======== Configuración ========
LLM_MODEL = "llama3.2:1b"   # puedes cambiar a "llama3" o "mistral"
ARCHIVOS = Path("archivos"); ARCHIVOS.mkdir(exist_ok=True)

# ======== Utilidades ========
def ensure_ollama_model(model: str) -> None:
    """Hace 'ollama pull <model>' sin ruido ni problemas de encoding en Windows."""
    try:
        # En Windows, silenciamos stdout/err para evitar UnicodeDecodeError en consola cp1252
        subprocess.run(
            ["ollama", "pull", model],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        # No detenemos la app si no se puede hacer pull (puede que ya exista o el daemon no esté activo)
        pass

def has_langchain_duckduckgo():
    try:
        from langchain_community.tools import DuckDuckGoSearchResults  # noqa: F401
        return True
    except Exception:
        return False

# ======== Datos de ejemplo (timeseries corta) ========
np.random.seed(1)
length = 30
ts = pd.DataFrame(
    data=np.random.randint(0, 15, size=length),
    columns=["y"],
    index=pd.date_range(start="2023-01-01", freq="MS", periods=length).strftime("%Y-%m"),
)
# Evita bloquear terminal; guarda la figura como archivo
fig = ts.plot(kind="bar", figsize=(10, 3), legend=False).get_figure()
fig.tight_layout()
fig_path = ARCHIVOS / "timeseries_preview.png"
fig.savefig(fig_path, dpi=150)
plt.close(fig)

dtf = ts.reset_index().rename(columns={"index": "date"})
data_preview = "\n".join([str(row) for row in dtf.to_dict(orient="records")[:5]])

SYSTEM_PROMPT = f"""
You are a data analysis agent. The dataset 'dtf' is already loaded and contains monthly sales:
Preview (first rows):
{data_preview}

Rules:
- Prefer concise, actionable insights.
- If you need to run Python on 'dtf' or 'ts', call the tool 'code_exec'.
- If you need web context, call 'search_web'.
- When done, call 'final_answer'.
"""

MEMORY_HINT = "The dataset already exists as 'dtf' and 'ts'. Don't create a new one."

# ======== Herramientas ========
# Estado compartido para code_exec
_EXEC_STATE = {"dtf": dtf, "ts": ts, "pd": pd, "np": np, "plt": plt, "ARCHIVOS": ARCHIVOS}

def final_answer(text: str) -> str:
    return text

def code_exec(code: str) -> str:
    """
    Ejecuta Python con estado persistente.
    Usa print(...) para devolver resultados.
    Puede guardar en ARCHIVOS.
    """
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        try:
            exec(code, _EXEC_STATE, _EXEC_STATE)
        except Exception as e:
            print(f"Error: {e}")
    return output.getvalue()

if has_langchain_duckduckgo():
    from langchain_community.tools import DuckDuckGoSearchResults
    _duck = DuckDuckGoSearchResults(backend="news")
    def search_web(query: str) -> str:
        try:
            return _duck.run(query)
        except Exception as e:
            return f"[search_web error] {e}"
else:
    def search_web(query: str) -> str:
        return "[search_web desactivado] Instala langchain-community para usar DuckDuckGo."

# Especificaciones para herramientas (estilo function-calling)
tool_final_answer = {
    "type": "function",
    "function": {
        "name": "final_answer",
        "description": "Returns a natural language response to the user",
        "parameters": {
            "type": "object",
            "required": ["text"],
            "properties": {"text": {"type": "string", "description": "response"}},
        },
    },
}
tool_code_exec = {
    "type": "function",
    "function": {
        "name": "code_exec",
        "description": "Execute python code. Always use print() to produce output.",
        "parameters": {
            "type": "object",
            "required": ["code"],
            "properties": {"code": {"type": "string", "description": "code to execute"}},
        },
    },
}
tool_search_web = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "Search the web (DuckDuckGo News backend).",
        "parameters": {
            "type": "object",
            "required": ["query"],
            "properties": {"query": {"type": "string", "description": "topic to search"}},
        },
    },
}

TOOLS_IMPL = {
    "final_answer": final_answer,
    "code_exec": code_exec,
    "search_web": search_web,
}

# ======== Motor de herramientas ========
def use_tool(agent_res: dict) -> dict:
    """
    Ejecuta la(s) tool_calls provistas por el modelo y devuelve:
    {'res': <salida>, 'tool_used': <nombre>, 'inputs_used': <dict>}
    """
    res, t_name, t_inputs = "", "", {}
    msg = agent_res.get("message") or {}
    calls = msg.get("tool_calls") or []

    if calls:
        # Ejecutamos sólo la primera tool que pida el modelo (flujo simple)
        tool = calls[0]
        fn = tool.get("function", {})
        t_name = fn.get("name", "")
        t_inputs = fn.get("arguments", {})
        if isinstance(t_inputs, str):
            try:
                t_inputs = json.loads(t_inputs)
            except Exception:
                t_inputs = {}
        impl = TOOLS_IMPL.get(t_name)
        if impl:
            print(f"[tool] {t_name} -> inputs: {t_inputs}")
            res = impl(**t_inputs)
        else:
            res = f"[Tool {t_name} not found]"
    else:
        # No pidió herramienta; quizá devolvió texto directo
        content = msg.get("content")
        if content:
            res = content

    return {"res": res, "tool_used": t_name, "inputs_used": t_inputs}

# ======== Agente con Ollama ========
def run_agent(model: str, messages: list, tools_catalog: dict) -> str:
    import ollama
    available = dict(tools_catalog)  # copia
    tool_used, res = "", ""

    while tool_used != "final_answer":
        try:
            agent_res = ollama.chat(
                model=model,
                messages=messages,
                tools=[v for v in available.values()],
            )
            dic_res = use_tool(agent_res)
            res, tool_used, inputs_used = dic_res["res"], dic_res["tool_used"], dic_res["inputs_used"]
        except Exception as e:
            res = f"Error llamando al modelo: {e}"
            print(res)
            messages.append({"role": "assistant", "content": res})

        if tool_used and tool_used != "final_answer":
            # Añadimos la salida de la tool como mensaje para el siguiente paso del agente
            messages.append({
                "role": "assistant",
                "content": f"Tool used: {tool_used}, Output: {res[:1500]}"  # truncado por seguridad
            })
            # Evitamos repetir la misma tool en este ciclo simple
            available.pop(tool_used, None)
            # Si sólo queda final_answer, empujamos instrucción para cerrar
            if list(available.keys()) == ["final_answer"]:
                messages.append({"role": "user", "content": "now activate the tool final_answer."})

        if tool_used == "":
            # El modelo no pidió tool; devolvió texto final
            break

    return res

# ======== Main ========
def main():
    ensure_ollama_model(LLM_MODEL)

    # Mensajes iniciales
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("Figura de previsualización guardada en:", str(fig_path))
    print("Escribe tu consulta. Teclea 'quit' para salir.\n")

    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSaliendo.")
            break

        if q.lower() in {"quit", "exit", ":q"}:
            break

        # Alimentamos al agente
        messages.append({"role": "user", "content": q})
        messages.append({"role": "user", "content": MEMORY_HINT})

        res = run_agent(
            LLM_MODEL,
            messages,
            tools_catalog={
                "final_answer": tool_final_answer,
                "code_exec": tool_code_exec,
                "search_web": tool_search_web,
            },
        )
        print("\n" + ("-" * 60))
        print(res)
        print("-" * 60 + "\n")
        messages.append({"role": "assistant", "content": res})

if __name__ == "__main__":
    main()
