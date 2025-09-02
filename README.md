# Proyecto Eléctrico - Sistema de análisis y predicción de series

Los archivos incluidos en el repositorio original son:

- `requirements.txt`: Especificación de las dependencias de paquetes de Python.
- `problemas.txt`: Problemas encontrados en el desarrollo del proyecto.
- `env_prueba.py`: Archivo python para comprobar la correcta instalación del entorno virtual y sus respectivas librerías.
- `system_Mauro.py`: Archivo python para ejecutar el sistema base propuesto por Mauro Di Pietro.
  
- `archivos/datos_cence`:  Datos de CENCE de los últimos tres años
- `archivos/ProyectoElectrico_PrediccionEnTiempoReal.pdf`: Enunciado y objetivo del proyecto.
 
- `LICENSE`: Licencia Creative Commons [Zero v1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/deed.es) de derechos de autor.
- `.gitignore`: Archivos y directorios que Git ignora para hacer control de versiones y para subir a un repositorio remoto. Típicamente son archivos con datos privados o específicos del sistema operativo local, que no deben reproducirse en el repositorio de otros desarrolladores.
- `_pycache_`: Directorio automático de python que almacena los archivos de código de bytes compilados.
- `.venv`: Configuración del entorno virtual

## Documentación e instrucciones del proyecto

El enunciado del proyecto y sus objetivos están disponibles en el pdf:

[ProyectoElectrico_PrediccionEnTiempoReal.pdf]

## Instrucciones para ejecución local

Para la realización del proyecto se trabajó en Visual Studio Code bajo un entorno virtual.


## Clonar el repositorio

Para comenzar, es necesario "clonar" el repositorio con sus archivos localmente. Para esto:

- Asegurarse de que Git está instalado. Es posible probar con `$ git --version`.
- Ubicarse en el directorio donde estará ubicado el proyecto, con `$ cd`.
- Clonar el proyecto con `$ git clone https://github.com/ErickVM7/Proyecto-Electrico`.
- Moverse al directorio del proyecto con `$ cd Proyecto-Electrico`.
- Si no fue hecho antes, configurar las credenciales de Git en el sistema local, con `$ git config --global user.name "Nombre Apellido"` y `$ git config --global user.email "your-email@example.com"`, de modo que quede vinculado con la cuenta de GitHub.

## Descargar e instalar Ollama

Seguir los pasos de descarga e instalación de Ollama desde su página según su sistema operativo. 
https://ollama.com/download

## Crear un ambiente virtual de Python

En una terminal, en el directorio raíz del repositorio, utilizar:

```bash
python3 -m venv env
```

donde `env` es el nombre del ambiente. Esto crea una carpeta con ese nombre.

Para activar el ambiente virtual, utilizar:

En Linux/macOS:
```bash
source env/bin/activate
```

donde `env/bin/activate` es el `PATH`. El *prompt* de la terminal cambiará para algo similar a:

```bash
base env ~/.../pipeline $
```
En Windows (CMD):
```cmd
.\env\Scripts\Activate.bat
```

En Windows (PowerShell):
```powershell
.\env\Scripts\Activate.ps1
```

Nota: si aparece un error sobre ejecución de scripts o permisos, ejecuta: 
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```


En este ambiente virtual no hay paquetes de Python instalados. Es posible verificar esto con `pip list`, que devolverá algo como:

```bash
Package    Version
---------- -------
pip        24.0
setuptools 65.5.0
```

## Instalar los paquetes necesarios para ejecutar el proyecto

Con el ambiente virtual activado, instalar los paquetes indicados en el archivo `requirements.txt`, con:

```bash
pip install -r requirements.txt
```

Para verificar la instalación, es posible usar nuevamente `pip list`, que ahora mostrará una buena cantidad de nuevos paquetes y sus dependencias. Además, se puede ejecutar el archivo `env_prueba.py` para verificar que se haya instalado correctamente el entorno virtual.

## Sistema base propuesto por Mauro Di Pietro

El artículo de Mauro Di Pietro muestra cómo construir desde cero un agente de inteligencia artificial capaz de procesar series temporales y dataframes grandes, utilizando únicamente Python y la librería Ollama para ejecutar modelos de lenguaje en local, sin depender de GPU en la nube ni de claves de API. La diferencia entre modelo y agente radica en que, el modelo interpreta los datos como texto mientras que un agente está diseñado para planificar acciones y activar herramientas para procesar datos. Por ejemplo, si tenemos una tabla convertida en cadena de caracteres, el modelo puede identificar patrones, columnas o tendencias, pero no calcula con precisión porque no ejecuta código.

En el sistema base diseñado en el árticulo, se apoya en tres funciones principales:

- code_exec → ejecuta código Python sobre el dataset ya cargado.
- search_web → realiza búsquedas rápidas (DuckDuckGo News).
- final_answer → entrega una respuesta en lenguaje natural al usuario.

El dataset inicial se genera al inicio como ejemplo y se mantiene en memoria. Esto permite que el agente lo consulte o lo transforme durante la sesión. Otra función importante es el `prompt inicial`, ya que es el punto de partida que define el comportamiento del agente. Este establece un marco de trabajo: primero se le asigna un rol específico, indicándole que debe actuar como analista de datos y responder a tareas relacionadas con series temporales o grandes tablas. Luego, se le enumeran las herramientas disponibles, explicando qué puede hacer con cada una de ellas, por ejemplo ejecutar código en Python o devolver un resultado en lenguaje natural. Además, se proporciona un contexto mínimo del dataset, mostrando algunas filas de la tabla para que el modelo reconozca la estructura de las columnas y los tipos de datos, y se le recuerda explícitamente que el dataset ya existe en memoria para evitar que intente recrearlo en cada paso. Finalmente, se incluyen instrucciones prácticas sobre cómo deben presentarse los resultados, como usar siempre la función print() al ejecutar código. De esta manera, el prompt inicial funciona como una guía que establece las reglas de interacción entre el usuario y el agente.

El artículo concluye que este enfoque democratiza el análisis de datos: cualquier persona podría interactuar con tablas y series temporales mediante lenguaje natural, mientras el agente se encarga de traducir esas peticiones en código, ejecutar operaciones y devolver resultados interpretables.

### Flujo de uso

1.  Iniciar el script
    El sistema carga el modelo, genera un dataset de ejemplo y abre un
    loop interactivo.
    
2.  Interactuar con el agente

    -   El usuario hace preguntas o solicitudes (ej: “calcula promedio”).
    -   El modelo decide si responde en texto (final_answer), ejecuta
        código (code_exec) o busca información (search_web).

3.  Recepción de resultados

    -   Si se usa final_answer: devuelve texto.
    -   Si se usa code_exec: ejecuta Python y devuelve la salida de este código
    -   Si se usa search_web: devuelve un resumen de noticias
        relevantes.


## Estado actual del proyecto

Este repositorio contiene la primera etapa del proyecto (**Tarea 1 y 2**), que corresponde a:

- Instalación y configuración del entorno virtual de Python.
- Instalación y prueba de Ollama en local.
- Implementación y prueba del sistema base propuesto por Mauro Di Pietro (`system_Mauro.py`), 
  el cual permite interactuar con un agente que procesa series de tiempo simples y dataframes en memoria.

- Obtención de datos ICE (CENCE). (`obtencionDatos.py`)
- Convertir los datos a un archivo .csv bien estructurado. (`datos_limpios.csv`)
- Estandarizar formato de fechas y valores. (`datos_limpios.csv`)
- Validar que los datos no tengan errores o vacíos.



## Para ejecutar el proyecto






