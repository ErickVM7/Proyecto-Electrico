import ollama
import subprocess

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

llm = "llama3.2:1b"

# Pull el modelo
subprocess.run(["ollama", "pull", llm])

stream = ollama.generate(model=llm, prompt='''hi''', stream=True)
for chunk in stream:
    print(chunk['response'], end='', flush=True)
    
    



## create data
np.random.seed(1) #<--for reproducibility
length = 30
ts = pd.DataFrame(data=np.random.randint(low=0, high=15, size=length),
                  columns=['y'],
                  index=pd.date_range(start='2023-01-01', freq='MS', periods=length).strftime('%Y-%m'))

## plot
ts.plot(kind="bar", figsize=(10,3), legend=False, color="black").grid(axis='y')
plt.show(block=False)

dtf = ts.reset_index().rename(columns={"index":"date"})
dtf.head()

data = dtf.to_dict(orient='records')
data[0:5]

str_data = "\n".join([str(row) for row in data])
str_data

prompt = f'''
Analyze this dataset, it contains monthly sales data of an online retail product:
{str_data}
'''

messages = [{"role":"system", "content":prompt}]

while True:
    ## User
    q = input('🙂 >')
    if q == "quit":
        break
    messages.append( {"role":"user", "content":q} )
   
    ## Model
    agent_res = ollama.chat(model=llm, messages=messages, tools=[])
    res = agent_res["message"]["content"]
   
    ## Response
    print("👽 >", f"\x1b[1;30m{res}\x1b[0m")
    messages.append( {"role":"assistant", "content":res} )