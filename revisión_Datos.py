import matplotlib.pyplot as plt
import pandas as pd

# Cargar archivo
data = pd.read_csv("datos_limpios.csv")

# Vista previa
print("\n Vista previa:")
print(data.head())

# Información general
print("\n Info del csv:")
data.info()

# 1. Revisar valores nulos por columna
print("\n Valores nulos por columna:")
print(data.isnull().sum())

# 2. Revisar duplicados
print("\n Número de filas duplicadas (todas las columnas):", data.duplicated().sum())

# 3. Verificar columna fechaHora
if "fechaHora" in data.columns:
    data["fechaHora"] = pd.to_datetime(data["fechaHora"], errors="coerce")
    print("Duplicados solo en fechaHora:", data.duplicated(subset=["fechaHora"]).sum())

    # Revisar rango de fechas
    print("\n Rango de fechas:")
    print("Mínima:", data["fechaHora"].min())
    print("Máxima:", data["fechaHora"].max())

# 4. Estadísticas básicas
print("\n Estadísticas descriptivas:")
print(data.describe())

# 5. Porcentaje de valores faltantes
faltantes = data.isnull().mean() * 100
print("\n Porcentaje de valores faltantes por columna (%):")
print(faltantes)

# Ajustes para graficar
# Renombrar columnas
data = data.rename(columns={"MW": "demanda_mw", "MW_P": "demanda_pronosticada"})

# Poner fechaHora como índice
data = data.set_index("fechaHora")

# 6. Seleccionar un día específico para graficar
fecha = "2024-09-26"
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
