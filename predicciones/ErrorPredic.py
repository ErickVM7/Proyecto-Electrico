import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

# ==========================
# CONFIGURACIÓN
# ==========================
PATH = "./predicciones/"

archivos_reales = {
    "1dia": "DatosReales1dias.csv",
    "2dias": "DatosReales2dias.csv",
    "3dias": "DatosReales3dias.csv"
}

archivos_pred = {
    "ARIMA_2d": "Pred_ARIMA_2DIAS.csv",
    "ARIMA_3d": "Pred_ARIMA_3DIAS.csv",
    "Prophet_1d": "Pred_Prophet_1dias.csv",
    "Prophet_3d": "Pred_Prophet_3dias.csv"
}

# ==========================
# FUNCIÓN PARA CALCULAR MÉTRICAS
# ==========================
def calcular_metricas(real, pred):
    mae = mean_absolute_error(real, pred)
    rmse = np.sqrt(mean_squared_error(real, pred))
    mape = np.mean(np.abs((real - pred) / real)) * 100
    return mae, rmse, mape

# ==========================
# PROCESAMIENTO
# ==========================
resultados = []

for nombre_pred, archivo_pred in archivos_pred.items():
    # Determinar cuántos días cubre el archivo
    if "1" in archivo_pred:
        archivo_real = archivos_reales["1dia"]
    elif "2" in archivo_pred:
        archivo_real = archivos_reales["2dias"]
    else:
        archivo_real = archivos_reales["3dias"]

    # Leer archivos
    df_real = pd.read_csv(os.path.join(PATH, archivo_real))
    df_pred = pd.read_csv(os.path.join(PATH, archivo_pred))

    # Asegurar nombres consistentes
    df_real = df_real.rename(columns={"fechaHora": "fecha", "MW": "MW_real"})
    df_pred = df_pred.rename(columns={"MW_P": "MW_pred", "MW": "MW_pred"})

    # Convertir fechas
    df_real["fecha"] = pd.to_datetime(df_real["fecha"], errors="coerce")
    df_pred["fecha"] = pd.to_datetime(df_pred["fecha"], errors="coerce")

    # Unir por fecha
    df_merge = pd.merge(df_real[["fecha", "MW_real"]],
                        df_pred[["fecha", "MW_pred"]],
                        on="fecha", how="inner")

    # Calcular métricas
    mae, rmse, mape = calcular_metricas(df_merge["MW_real"], df_merge["MW_pred"])

    resultados.append({
        "Modelo": nombre_pred,
        "Archivo Real": archivo_real,
        "MAE (MW)": round(mae, 3),
        "RMSE (MW)": round(rmse, 3),
        "MAPE (%)": round(mape, 2),
        "N muestras": len(df_merge)
    })

# ==========================
# RESULTADOS
# ==========================
df_resultados = pd.DataFrame(resultados)
print("\n=== Resultados de comparación ===")
print(df_resultados)


