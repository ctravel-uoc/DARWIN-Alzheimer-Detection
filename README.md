# Modelos predictivos e Inteligencia Artificial Explicable para la detección temprana del Alzheimer mediante el análisis de escritura digitalizada del protocolo DARWIN

<div align="center">

  <p><strong>Modelos predictivos para la detección temprana del Alzheimer basado en biomarcadores digitales y transparencia clínica (XAI)</strong></p>

  [![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?style=flat-square&logo=python)](https://www.python.org/)
  [![Machine Learning](https://img.shields.io/badge/ML-XGBoost%20%7C%20EBM-orange.svg?style=flat-square)](https://github.com/microsoft/interpret)
  [![XAI](https://img.shields.io/badge/IA_Explicable-SHAP%20%7C%20LIME%20%7C%20Anchors-brightgreen.svg?style=flat-square)](https://github.com/slundberg/shap)
  [![Status](https://img.shields.io/badge/Status-TFM_Completado-success.svg?style=flat-square)](#)

  **Trabajo de Fin de Máster (TFM)** | **Autora:** Carmen Travel Alarcón  
  *Máster en Ciencia de Datos - Universitat Oberta de Catalunya*

</div>

---

## Objetivo: Detectar lo invisible

El diagnóstico del Alzheimer suele llegar tarde. Cuando los síntomas son evidentes, el daño neurológico ya es profundo. Este proyecto nace de una idea ambiciosa: **¿Podemos encontrar el rastro de la enfermedad en la forma sutil en que movemos el bolígrafo?**

Utilizando el dataset clínico **DARWIN**, este trabajo desarrolla una herramienta de cribado que analiza "micro-indicadores" de la escritura (presión, pausas y velocidad) para identificar señales de deterioro antes de que los tests tradicionales puedan detectarlas.

### El hito: Protocolo DARWIN-11
El protocolo original pedía a los pacientes completar 25 tareas, algo que causaba fatiga y estrés en personas mayores. Este sistema ha logrado aislar las **11 tareas críticas** (denominado **DARWIN-11**), reduciendo la duración de la prueba a la mitad sin perder precisión.

> **En resumen:** Una prueba más humana, más rápida y lista para ser usada en una consulta de atención primaria.

---

## Lo que nos cuenta la Inteligencia Artificial

Este proyecto se aleja de las "cajas negras" incomprensibles. Hemos usado **IA Explicable (XAI)** para que el modelo nos explique sus razones:

1. **Planificar antes que actuar:** El biomarcador más importante no fue la velocidad de la mano, sino el **tiempo en el aire** (`air_time17`). Los pacientes con Alzheimer no escriben más despacio; simplemente necesitan más tiempo para planificar el siguiente trazo.
2. **Diferentes perfiles, diferentes pacientes:** Al agrupar las explicaciones de la IA, descubrimos que el Alzheimer no afecta a todos igual. Identificamos perfiles de "bloqueo cognitivo" frente a otros de "deterioro distribuido", abriendo la puerta a un seguimiento más personalizado.
3. **El límite de la resiliencia:** Creamos un simulador que detecta el punto exacto en el que el deterioro de un paciente sano dispara las alarmas del modelo.
4. **Pensado para el mundo real:** El sistema es tan ligero que puede ejecutarse localmente en una **tablet**, sin necesidad de servidores potentes ni conexión a internet, garantizando la privacidad absoluta de los datos.

---

## ¿Qué hay en este repositorio?

He organizado el código para que sea fácil de seguir, desde la limpieza inicial hasta la auditoría final:

```text
DARWIN_Alzheimer_Detection/
├── data/                # Aquí va el CSV del dataset DARWIN (ver sección Dataset)
├── models/              # Modelos y datos serializados entre fases
├── notebooks/           # Los notebooks, en orden:
│   ├── 01_02_EDA_Preprocesamiento.ipynb
│   ├── 04_Modelado.ipynb
│   └── 05_XAI_Simulacion.ipynb
├── src/
│   └── 03_Seleccion_Variables.py
├── requirements.txt
└── README.md
```

### Descripción de cada fase

**`01_02_EDA_Preprocesamiento.ipynb` — Fases 1 y 2**  
Exploración clínica del dataset: distribuciones, balance de clases, análisis de fatiga entre tareas y visualización de la "huella clínica" mediante radar. Incluye el preprocesamiento completo: partición estratificada 80/20 y normalización robusta con `RobustScaler`.

**`03_Seleccion_Variables.py` — Fase 3**  
Script independiente (alta carga computacional). Aplica un filtro de colinealidad de Pearson seguido de selección recursiva con validación cruzada (RFECV), reduciendo las 450 variables originales a 224 biomarcadores relevantes.

**`04_Modelado.ipynb` — Fase 4**  
Comparativa de cuatro modelos (Random Forest, Extra Trees, XGBoost y EBM) bajo distintos escenarios de reducción del protocolo. Aquí se identifica DARWIN-11 como el punto óptimo y se audita el consumo de memoria con `tracemalloc`.

**`05_XAI_Simulacion.ipynb` — Fase 5**  
Capa completa de explicabilidad: importancias globales con SHAP y EBM, curvas de dependencia parcial (PDP), fenotipos de pacientes con K-Means, auditoría de errores con Waterfall, explicaciones locales con LIME y Anchors, y simulación de deterioro progresivo.

---

## Dataset

Este proyecto utiliza el dataset **DARWIN** (*Digital Analysis of Real-time Writing for dIagNosis*), compuesto por registros de escritura digitalizada de 174 participantes (89 pacientes con Alzheimer y 85 controles sanos) en 25 tareas estructuradas.

El dataset está disponible públicamente en la siguiente publicación original:

> Cilia, N. D., De Gregorio, G., De Stefano, C., Fontanella, F., Marcelli, A., & Parziale, A. (2022). Diagnosing Alzheimer’s disease from on-line handwriting: A novel dataset and performance benchmarking. *Engineering Applications of Artificial Intelligence*, 111, 104822. https://doi.org/10.1016/j.engappai.2022.104822

El archivo CSV debe colocarse en la carpeta `data/` antes de ejecutar los notebooks.

---

## Stack tecnológico

* **Procesamiento y estadística:** `pandas`, `numpy`, `scipy`
* **Machine learning:** `scikit-learn`, `xgboost`, `interpret` (EBM)
* **Explicabilidad (XAI):** `shap`, `lime`, `anchor-exp`
* **Visualización:** `matplotlib`, `seaborn`
* **Reproducibilidad y serialización:** `joblib`
* **Auditoría de recursos:** `tracemalloc`, `time`

---

## Instalación y reproducción

**1. Clona el repositorio:**
```bash
git clone https://github.com/ctravel-uoc/DARWIN_Alzheimer_Detection
```

**2. Instala las dependencias:**
```bash
pip install -r requirements.txt
```

**3. Descarga el dataset** y colócalo en `data/DARWIN.csv` (ver enlace en sección Dataset).

**4. Ejecuta las fases en orden:**

```bash
# Fase 3 (script de consola, puede tardar varios minutos)
python src/03_Seleccion_Variables.py
```

Los notebooks `01_02`, `04` y `05` se ejecutan secuencialmente en Jupyter. Cada fase guarda sus resultados en `models/` para que la siguiente pueda usarlos.

---

> **Disclaimer médico:** *Este proyecto es un Trabajo de Fin de Máster de carácter académico. Las conclusiones, métricas y algoritmos aquí desarrollados requieren validación clínica antes de cualquier aplicación diagnóstica en entornos sanitarios reales.*
