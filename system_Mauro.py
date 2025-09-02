#Librerías

#Datos 
import pandas as pd
import numpy as np

#Generar random
import random
import string

#Capturar salida
import io
import contextlib
import re

length = 1000

#Conexión con Ollama

import ollama
import subprocess



llm = "llama3.2:1b" #Modelo, se puede cambiar por llamas o Mastral
#llm = "qwen2.5" 


# Descargar (pull) el modelo antes de usarlo
subprocess.run(["ollama", "pull", llm], check=False)

stream = ollama.generate(model=llm, prompt='''can you analyze timeseries?''', stream=True)
for chunk in stream:
    print(chunk['response'], end='', flush=True) #Se pide una respuesta al modelo y se va imprimiendo el texto


# Creación de una serie temporal de datos
    
import matplotlib.pyplot as plt
from datetime import datetime

## create data
np.random.seed(1) #<--for reproducibility
length = 30
ts = pd.DataFrame(data=np.random.randint(low=0, high=15, size=length),
                  columns=['y'], 
                  index= pd.date_range(start='2023-01-01', freq='MS', periods=length).strftime('%Y-%m'))

## plot
ts.plot(kind="bar", figsize=(10,3), legend=False, color="black").grid(axis='y')

ts.head()

dtf = ts.reset_index().rename(columns={"index":"date"})
dtf.head()

data = dtf.to_dict(orient='records')
data[0:5]

str_data = "\n".join([str(row) for row in data])
str_data

str_data = "\n".join([str(row) for row in dtf.to_dict(orient='records')])

plt.show(block=False)

#Dataset de ejemplo
dtf = pd.DataFrame(data={
    'Id': [''.join(random.choices(string.ascii_letters, k=5)) for _ in range(length)],
    'Age': np.random.randint(low=18, high=80, size=length),
    'Score': np.random.uniform(low=50, high=100, size=length).round(1),
    'Status': np.random.choice(['Active','Inactive','Pending'], size=length)
})
dtf.tail()

print("Shape:", dtf.shape)
print(dtf.info())
print(dtf.describe())


#Herramientas para el agente

# Agente "cierra" respuesta final

def final_answer(text:str) -> str:
    return text

tool_final_answer = {
  'type': 'function',
  'function': {
    'name': 'final_answer',
    'description': 'Returns a natural language response to the user',
    'parameters': {
      'type': 'object',
      'required': ['text'],
      'properties': {
        'text': {'type':'string', 'description':'natural language response'}
      }
    }
  }
}
final_answer(text="hi")

# Permite ejecutar código python dinamicamente
#Devuelve texto

# Sanitizador de código antes de ejecutar
def sanitize_code(code: str) -> str:
    # Forzar uso de dtf en lugar de df
    code = code.replace("df[", "dtf[")

    # Reemplazar funciones sueltas por métodos de pandas
    code = re.sub(r"mean\((dtf\[.+?\])\)", r"\1.mean()", code)
    code = re.sub(r"max\((dtf\[.+?\])\)", r"\1.max()", code)
    code = re.sub(r"min\((dtf\[.+?\])\)", r"\1.min()", code)
    code = re.sub(r"sum\((dtf\[.+?\])\)", r"\1.sum()", code)

    # Asegurar nombres de columnas correctos según dtf
    for col in dtf.columns:
        code = code.replace(col.lower(), col).replace(col.upper(), col)

    return code


def code_exec(code: str) -> str:
    code = sanitize_code(code)  # 👈 limpiar antes de ejecutar
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        try:
            exec(code, globals())
        except Exception as e:
            print(f"Error: {e}")
    return output.getvalue()

tool_code_exec = {'type':'function', 'function':{
  'name': 'code_exec',
  'description': 'Execute python code. Use always the function print() to get the output.',
  'parameters': {'type': 'object', 
                 'required': ['code'],
                 'properties': {
                    'code': {'type':'str', 'description':'code to execute'},
}}}}

code_exec("from datetime import datetime; print(datetime.now().strftime('%H:%M'))")








# Herramienta para buscar noticias y devuelve texto
from langchain_community.tools import DuckDuckGoSearchResults

from ddgs import DDGS

def search_web(query: str) -> str:
    with DDGS() as ddgs:
        results = ddgs.news(query)
        return str(results)



tool_search_web = {'type':'function', 'function':{
  'name': 'search_web',
  'description': 'Search the web',
  'parameters': {'type': 'object', 
                 'required': ['query'],
                 'properties': {
                    'query': {'type':'str', 'description':'the topic or subject to search on the web'},
}}}}

search_web(query="nvidia")

#Diccionario de herramientas
dic_tools = {"final_answer":final_answer, "code_exec":code_exec, "search_web":search_web}


# Ejecuta la herramienta correspondiente
def use_tool(agent_res: dict, dic_tools: dict) -> dict:
    msg = agent_res["message"]
    res, t_name, t_inputs = "", "", ""

    if hasattr(msg, "tool_calls") and msg.tool_calls:
        for tool in msg.tool_calls:
            t_name, t_inputs = tool["function"]["name"], tool["function"]["arguments"]

            # 👇 Mapeo para final_answer
            if t_name == "final_answer" and "final_answer" in t_inputs:
                t_inputs = {"text": t_inputs["final_answer"]}

            if f := dic_tools.get(t_name):
                print('🔧 >', f"\x1b[1;31m{t_name} -> Inputs: {t_inputs}\x1b[0m")
                t_output = f(**t_inputs)
                print(t_output)
                res = t_output
            else:
                print('🤬 >', f"\x1b[1;31m{t_name} -> NotFound\x1b[0m")

    # Si no se usó herramienta → devolver mensaje normal
    if msg.get("content", "") != "":
        res = msg["content"]
        t_name, t_inputs = "", ""

    return {"res": res, "tool_used": t_name, "inputs_used": t_inputs}

#Bucle principal del agente, llama otras herramientas y devuelve respuestas
def run_agent(llm, messages, available_tools):
    tool_used, local_memory = '', ''
    while tool_used != 'final_answer':
        ### use tools
        try:
            agent_res = ollama.chat(model=llm, messages=messages, format="json",   # 👈 evita respuestas ambiguas
            tools=[v for v in available_tools.values()])

            dic_res = use_tool(agent_res, dic_tools)
            res, tool_used, inputs_used = dic_res["res"], dic_res["tool_used"], dic_res["inputs_used"]
        ### error
        except Exception as e:
            print("⚠️ >", e)
            res = f"I tried to use {tool_used} but didn't work. I will try something else."
            print("👽 >", f"\x1b[1;30m{res}\x1b[0m")
            messages.append( {"role":"assistant", "content":res} )
        ### update memory
        if tool_used not in ['','final_answer']:
            local_memory += f"\nTool used: {tool_used}.\nInput used: {inputs_used}.\nOutput: {res}"
            messages.append( {"role":"assistant", "content":local_memory} )
            available_tools.pop(tool_used, None)
            if len(available_tools) == 1:
                messages.append( {"role":"user", "content":"now activate the tool final_answer."} )
        ### tools not used
        if tool_used == '':
            break
    return res

# Prompt
str_data = "\n".join([str(row) for row in dtf.head(10).to_dict(orient='records')])

#PROMPT inicial y se define el rol y que herramientas tiene
prompt = f'''
You are a Data Analyst. You will be given a task to solve as best you can.

When you need to calculate something from the DataFrame, always use the existing variable dtf directly with pandas.
Do not invent new functions unless explicitly asked.

You have access to the following tools:
- tool 'final_answer' to return a text response.
- tool 'code_exec' to execute Python code.
- tool 'search_web' to search for information on the internet.

Important rules:
- The dataset already exists and is called `dtf`. Never create a new dataset or variable with another name.
- Always use **pandas methods** for calculations. 
  Examples:
    - Mean: dtf["Age"].mean()
    - Maximum: dtf["Score"].max()
    - Minimum: dtf["Age"].min()
    - Sum: dtf["Score"].sum()
- Never use standalone functions like mean(), max(), min(), sum(), etc. outside pandas.
- Always use double quotes `"` for column names.  
  Example: dtf["Score"], not dtf['Score'].
- Always wrap results you want to show with `print()`, otherwise they will not appear.
- Do not invent new functions or variables unless explicitly asked.
- The DataFrame is ALWAYS called dtf. Never use df, data, dataset, or anything else.
- All calculations must be done with dtf. Example: dtf["Age"].mean(), dtf["Score"].max().
- Never return code like "final_answer". Only valid Python code should go into code_exec.

Tool usage rules:
- If the user asks for a value, a statistic, or a calculation, you MUST always call code_exec first to compute the result.
- After executing the calculation with code_exec, you MUST use final_answer to provide a clear natural language explanation of the result.
- Do not return final_answer without running code_exec first for any numeric/statistical request.
- If the user asks for something unrelated to the DataFrame (not involving dtf), do NOT call code_exec.
  - If the answer requires external information, use search_web.
  - If the answer can be given directly (concepts, definitions, general knowledge), use final_answer.

This dataset contains credit scores for each customer of the bank. Here's the first rows:
{str_data}
'''



# Bucle del chat interativo
messages = [{"role":"system", "content":prompt}]
memory = '''
The dataset already exists and it's called 'dtf', don't create a new one.
'''
while True:
    ## User
    q = input('🙂 >')
    if q == "quit":
        break
    messages.append( {"role":"user", "content":q} )

    ## Memory
    messages.append( {"role":"user", "content":memory} )      
    
    ## Model
    available_tools = {"final_answer":tool_final_answer, "code_exec":tool_code_exec, "search_web":tool_search_web}
    res = run_agent(llm, messages, available_tools)
    
    ## Response
    print("👽 >", f"\x1b[1;30m{res}\x1b[0m")
    messages.append( {"role":"assistant", "content":res} )
    
#Resumen
#El modelo Ollama interpreta la pregunta.

#Si necesita hacer cálculos, puede usar code_exec.

#Si necesita buscar afuera, usa search_web.

#Al final devuelve la respuesta con final_answer.