# Proyecto Eléctrico - Sistema de análisis y predicción de series

Los archivos incluidos en el repositorio original son:

- `.venv`: configuración del entorno virtual
- `requirements.txt`: especificación de las dependencias de paquetes de Python.
- `ProyectoElectrico_PrediccionEnTiempoReal.pdf`: Enunciado y objetivo del proyecto.
- `env_prueba.py`: Archivo python para comprobar la correcta instalación del entorno virtual y sus respectivas librerías.
- `datos_cence`:  Datos de CENCE de los últimos tres años
- `LICENSE`: licencia Creative Commons [Zero v1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/deed.es) de derechos de autor.
- `.gitignore`: archivos y directorios que Git ignora para hacer control de versiones y para subir a un repositorio remoto. Típicamente son archivos con datos privados o específicos del sistema operativo local, que no deben reproducirse en el repositorio de otros desarrolladores.

## Documentación e instrucciones del proyecto

El enunciado del proyecto y sus objetivos están disponibles en el pdf:

[ProyectoElectrico_PrediccionEnTiempoReal.pdf]

## Instrucciones para ejecución local

Para la realización del proyecto se trabajó en Visual Studio Code bajo un entorno virtual.


### Clonar el repositorio

Para comenzar, es necesario "clonar" el repositorio con sus archivos localmente. Para esto:

- Asegurarse de que Git está instalado. Es posible probar con `$ git --version`.
- Ubicarse en el directorio donde estará ubicado el proyecto, con `$ cd`.
- Clonar el proyecto con `$ git clone https://github.com/ErickVM7/Proyecto-Electrico`.
- Moverse al directorio del proyecto con `$ cd Proyecto-Electrico`.
- Si no fue hecho antes, configurar las credenciales de Git en el sistema local, con `$ git config --global user.name "Nombre Apellido"` y `$ git config --global user.email "your-email@example.com"`, de modo que quede vinculado con la cuenta de GitHub.

### Descargar e instalar Ollama

Seguir los pasos de descarga e instalación de Ollama desde su pagína según su sistema operativo. 
https://ollama.com/download

### Crear un ambiente virtual de Python

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

### Instalar los paquetes necesarios para ejecutar el proyecto

Con el ambiente virtual activado, instalar los paquetes indicados en el archivo `requirements.txt`, con:

```bash
pip install -r requirements.txt
```

Para verificar la instalación, es posible usar nuevamente `pip list`, que ahora mostrará una buena cantidad de nuevos paquetes y sus dependencias. Además puedes ejecturar el archivo env_prueba.py para comprobar el correcto funcionamiento del entorno.



### Para ejecutar el proyecto

- En el directorio raíz, 

