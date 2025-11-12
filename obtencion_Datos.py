import requests #solicitudes HTTP
import pandas as pd
from datetime import datetime, timedelta #maneja fechas
from io import StringIO #leer archivos csv

# Parámetros base
url_base = "https://apps.grupoice.com/CenceWeb/data/sen/csv/DemandaMW" #las urls empiezan igual por que es ir iterando
intervalo = 15 #se toman cada 15 minutos

# Rango de fechas
fecha_inicio = datetime(2025, 8, 27)  # Cambiar fecha inicial
fecha_fin = datetime(2025, 8, 27)     # Cambiar fecha final

dfs = [] #lista vacia

# Iterar día por día
fecha = fecha_inicio #formato de fecha YYYYMMDD
while fecha <= fecha_fin:
    inicio = fecha.strftime("%Y%m%d")
    fin = fecha.strftime("%Y%m%d")

    url = f"{url_base}?intervalo={intervalo}&inicio={inicio}&fin={fin}"
    print(f"Descargando: {url}") #Se descarga y se construye la direccion
    
    r = requests.get(url) #se descarga en modo texto
    if r.status_code == 200 and len(r.text) > 50:  # asegurar que trajo datos
        df = pd.read_csv(StringIO(r.text))
        dfs.append(df) #se guardan los datos y si estan vacio se avisa
    else:
        print(f"Error: No se encontraron datos para {inicio}")

    fecha += timedelta(days=1) #siguiente día

# Unir todos los días en un dataset
if dfs:
    data = pd.concat(dfs, ignore_index=True)
    data.to_csv("DatosReales1dias.csv", index=False)
    print("Exito: Datos guardados en datos_limpios.csv")
else:
    print("Error: No se descargaron datos.")
