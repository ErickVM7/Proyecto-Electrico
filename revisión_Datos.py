import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


# Cargar archivo
data = pd.read_csv("datos_limpios.csv")

# Vista previa
print("\n Vista previa:")
print(data.head())

# Información general
print("\n Info del csv:")
data.info()

#  Revisar valores nulos por columna
print("\n Valores nulos por columna:")
print(data.isnull().sum())

#  Revisar duplicados
print("\n Número de filas duplicadas (todas las columnas):", data.duplicated().sum())

# Verificar columna fechaHora
if "fechaHora" in data.columns:
    data["fechaHora"] = pd.to_datetime(data["fechaHora"], errors="coerce")
    print("Duplicados solo en fechaHora:", data.duplicated(subset=["fechaHora"]).sum())

    # Revisar rango de fechas
    print("\n Rango de fechas:")
    print("Mínima:", data["fechaHora"].min())
    print("Máxima:", data["fechaHora"].max())

#  Estadísticas básicas
print("\n Estadísticas descriptivas:")
print(data.describe())

#  Porcentaje de valores faltantes
faltantes = data.isnull().mean() * 100
print("\n Porcentaje de valores faltantes por columna (%):")
print(faltantes)

# Ajustes para graficar
# Renombrar columnas
data = data.rename(columns={"MW": "demanda_mw", "MW_P": "demanda_pronosticada"})

# Poner fechaHora como índice
data = data.set_index("fechaHora")

# Seleccionar un día específico para graficar
fecha = "2025-08-26"
data_dia = data.loc[fecha]

# Graficar
plt.figure(figsize=(12,5))
plt.plot(data_dia.index, data_dia["demanda_mw"], marker="o", label="Demanda MW")
if "demanda_pronosticada" in data.columns:
    plt.plot(data_dia.index, data_dia["demanda_pronosticada"], linestyle="--", label="Demanda Pronosticada")

plt.title(f"Demanda eléctrica - {fecha}")
plt.xlabel("Hora")
plt.ylabel("MW")
plt.legend()
plt.grid(True)
plt.show()


# Calcular correlación entre demanda_mw y demanda_pronosticada
corr_value = data["demanda_mw"].corr(data["demanda_pronosticada"])
print(f"Correlación entre demanda_mw y demanda_pronosticada: {corr_value:.3f}")




# Calcular errores
data["error"] = data["demanda_pronosticada"] - data["demanda_mw"]
data["error_abs"] = data["error"].abs()
data["error_pct"] = (data["error_abs"] / data["demanda_mw"]) * 100

# Métricas globales
mae = mean_absolute_error(data["demanda_mw"], data["demanda_pronosticada"])
rmse = np.sqrt(mean_squared_error(data["demanda_mw"], data["demanda_pronosticada"]))
mape = data["error_pct"].mean()

print("\n=== Evaluación del modelo ===")
print(f"MAE  (Error Medio Absoluto): {mae:.3f} MW")
print(f"RMSE (Raíz del Error Cuadrático Medio): {rmse:.3f} MW")
print(f"MAPE (Error Porcentual Medio): {mape:.2f}%")

# Gráfico de error
plt.figure(figsize=(12,5))
plt.plot(data.index, data["error"], label="Error (MW_P - MW)", color="red")
plt.axhline(0, color="black", linestyle="--")
plt.title("Error de Predicción en el Tiempo")
plt.xlabel("Fecha")
plt.ylabel("Error (MW)")
plt.legend()
plt.grid(True)
plt.show()

# Gráfico comparativo real vs predicho
plt.figure(figsize=(12,5))
plt.plot(data.index, data["demanda_mw"], label="Real", alpha=0.7)
plt.plot(data.index, data["demanda_pronosticada"], label="Predicción", linestyle="--", alpha=0.8)
plt.title("Demanda Real vs Predicha")
plt.xlabel("Fecha")
plt.ylabel("MW")
plt.legend()
plt.grid(True)
plt.show()
