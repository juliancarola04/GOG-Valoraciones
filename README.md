# Estimador de valoraciones de videojuegos GOG

Proyecto de machine learning que estima la valoración promedio de un videojuego a partir de sus características. Incluye un notebook de análisis y entrenamiento y una aplicación web desarrollada con Streamlit.

## Acceso desde la nube

La aplicación está disponible en **[https://gog-valoraciones.streamlit.app](https://gog-valoraciones.streamlit.app)**. Podés utilizarla desde el navegador sin instalar el proyecto.

## ¿Qué hace la aplicación?

El formulario permite ingresar:

- Desarrolladora y publicadora.
- Año de lanzamiento.
- Precio base en dólares (USD).
- Restricción de edad.
- Si el videojuego es gratuito.
- Géneros y sistemas operativos compatibles.

Al presionar **Predecir Valoración**, la aplicación muestra una valoración estimada expresada sobre 5 puntos. Es necesario seleccionar al menos un género y un sistema operativo.

## Estructura del proyecto

```text
.
├── Entrenamiento/
│   ├── GOG_Entrenamiento.ipynb   # Análisis, preparación de datos y entrenamiento
│   ├── gog_games_dataset.csv    # Dataset utilizado
│   └── requirements.txt         # Dependencias del entrenamiento
├── Streamlit/
│   ├── app.py                   # Aplicación web
│   ├── modelo_gog_rf.pkl        # Pipeline entrenado para realizar predicciones
│   └── requirements.txt         # Dependencias de la aplicación
├── .gitignore
└── README.md
```

## Datos y modelo

El notebook utiliza el archivo `gog_games_dataset.csv`. Según la documentación incluida en el notebook, los datos provienen del [GOG.com Video Games Dataset de Kaggle](https://www.kaggle.com/datasets/lunthu/gog-com-video-games-dataset), creado por Victor Laputsky a partir de información de la API de GOG recopilada en junio de 2025.

El análisis contempla la limpieza y el filtrado de productos para trabajar con videojuegos, excluir títulos en desarrollo y conservar registros con al menos cinco reseñas. La variable objetivo es `overallAvgRating`.

Se comparan modelos de regresión lineal y Random Forest mediante validación cruzada de cinco particiones, utilizando MAE, RMSE y R². El modelo exportado para la aplicación es un pipeline de scikit-learn que integra el preprocesamiento y un `RandomForestRegressor` de 100 árboles.

Las predicciones son aproximaciones basadas en los datos disponibles. Características como la historia, la jugabilidad o la calidad técnica no están representadas directamente en el formulario y pueden influir en la valoración real.

## Ejecución local

Necesitás Python y pip. Ejecutá los siguientes comandos desde la carpeta raíz del proyecto.

### 1. Crear un entorno virtual

```powershell
python -m venv .venv
```

### 2. Instalar las dependencias de la aplicación

En Windows, podés utilizar directamente el Python del entorno sin activarlo:

```powershell
.\.venv\Scripts\python.exe -m pip install -r Streamlit/requirements.txt
```

### 3. Iniciar Streamlit

```powershell
.\.venv\Scripts\python.exe -m streamlit run Streamlit/app.py
```

Abrí la dirección que indique la terminal; por defecto es `http://localhost:8501`.

La aplicación busca `modelo_gog_rf.pkl` dentro de la carpeta `Streamlit`, junto a `app.py`. El archivo debe permanecer en esa ubicación para que se puedan realizar predicciones.

En Linux o macOS, reemplazá `.\.venv\Scripts\python.exe` por `.venv/bin/python`.

## Análisis y reentrenamiento

1. Instalá las dependencias de entrenamiento en el entorno virtual:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r Entrenamiento/requirements.txt
   ```

2. Abrí `Entrenamiento/GOG_Entrenamiento.ipynb` en un editor compatible con notebooks, como Visual Studio Code con la extensión Jupyter, y seleccioná el entorno `.venv` como kernel.
3. Ejecutá el notebook con `Entrenamiento` como directorio de trabajo, ya que el CSV se carga mediante una ruta relativa.
4. Ejecutá las celdas en orden. La última celda entrena el pipeline y genera `modelo_gog_rf.pkl` en el directorio de trabajo.
5. Para usar el modelo nuevo en la aplicación, copiá el archivo generado a `Streamlit/modelo_gog_rf.pkl`, reemplazando la versión anterior.

Los archivos `requirements.txt` fijan las versiones de las dependencias. Conviene mantener versiones compatibles de scikit-learn y joblib entre el entrenamiento y la aplicación para cargar el modelo serializado correctamente.

## Tecnologías utilizadas

- **Python** como lenguaje principal.
- **pandas y NumPy** para el procesamiento de datos.
- **scikit-learn** para el preprocesamiento, entrenamiento y evaluación.
- **Matplotlib** para las visualizaciones del notebook.
- **joblib** para guardar y cargar el pipeline.
- **Streamlit** para la interfaz web.

## Autor

Julián Agustín Carola, según la autoría indicada en el notebook de entrenamiento.
