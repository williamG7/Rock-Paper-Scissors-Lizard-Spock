# Rock, Paper, Scissors, Lizard, Spock - Clasificador de Gestos con Deep Learning

## Descripción

Proyecto de **Deep Learning** que implementa un **Clasificador de Imágenes de Gestos** usando **Redes Neuronales Convolucionales (CNN)** capaz de reconocer y clasificar 5 categorías de gestos: Piedra, Papel, Tijeras, Lagarto y Spock. 

El modelo se entrena con datasets de Google y se amplia con imágenes personalizadas. Además, incluye una implementación completa del juego **Piedra, Papel, Tijeras, Lagarto, Spock** que integra el modelo entrenado para jugar automáticamente reconociendo los gestos desde imágenes.

---

## Objetivos

1. **Cargar y explorar** el dataset de imágenes de Rock, Paper, Scissors de Google (2,520 imágenes de entrenamiento).
2. **Diseñar y entrenar** una Red Neuronal Convolucional (CNN) capaz de clasificar las 3 categorías iniciales con un **accuracy ≥ 90%**.
3. **Implementar callbacks personalizados** para optimizar el entrenamiento (`PararEnAccuracy` y `EarlyStopping`).
4. **Validar el modelo** con imágenes personalizadas cargadas desde el usuario.
5. **Ampliar el modelo** con 2 categorías adicionales: **Lagarto** y **Spock** (5 categorías totales).
6. **Reentrenar** el modelo con imágenes nuevas para alcanzar un accuracy ≥ 80% en las 5 categorías.
7. **Implementar el juego** Piedra, Papel, Tijeras, Lagarto, Spock:
   - Recibe dos inputs de imagen (jugador 1 y jugador 2)
   - El modelo clasifica cada gesto automáticamente
   - Aplica las reglas del juego
   - Determina el ganador o empate

---

## Dataset

### Datos de entrenamiento iniciales
- **Fuente (API Google):** `rps.zip` desde Google Cloud Storage
- **Link:** `https://storage.googleapis.com/learning-datasets/rps.zip`
- **Tamaño:** ~2,520 imágenes de entrenamiento (840 por categoría)
  - 840 imágenes de Piedra
  - 840 imágenes de Papel
  - 840 imágenes de Tijeras

### Datos de validación
- **Fuente (API Google):** `rps-test-set.zip`
- **Link:** `https://storage.googleapis.com/learning-datasets/rps-test-set.zip`
- **Tamaño:** ~372 imágenes de validación

### Datos extendidos
- **Categorías adicionales:** Lagarto y Spock (imágenes descargadas de internet o capturadas manualmente)
- **Total final:** 5 categorías de clasificación

### Características de las imágenes
- **Dimensión:** 150 x 150 píxeles RGB
- **Formato:** PNG
- **Contenido:** Imágenes de manos mostrando diferentes gestos

---

## Arquitectura del Modelo

```
Input (150 x 150 x 3)
    ↓
Conv2D(64, 3x3) + ReLU → MaxPooling2D(2,2)
    ↓
Conv2D(64, 3x3) + ReLU → MaxPooling2D(2,2)
    ↓
Conv2D(128, 3x3) + ReLU → MaxPooling2D(2,2)
    ↓
Conv2D(128, 3x3) + ReLU → MaxPooling2D(2,2)
    ↓
Flatten() → Dropout(0.5)
    ↓
Dense(512, ReLU)
    ↓
Dense(5, Softmax)  [Rock, Paper, Scissors, Lizard, Spock]
```

### Parámetros del modelo
- **Total de parámetros:** 3,473,475 (13.25 MB)
- **Parámetros entrenables:** 3,473,475
- **Loss:** Categorical Crossentropy
- **Optimizer:** RMSprop
- **Métrica:** Accuracy

---

## Metodología

### 1) Importación y Configuración
- Descarga de datasets desde Google Cloud Storage
- Instalación de `keras_preprocessing`
- Importación de librerías necesarias (TensorFlow, Keras, NumPy, Matplotlib)

### 2) Exploración de Datos (EDA)
```python
- Recuento de imágenes por categoría
- Visualización de muestras
- Análisis de estructura de carpetas
- Validación de convención de nombres
```

### 3) Preparación de Datos
- **Data Augmentation:** Rotación, desplazamiento, zoom, flip horizontal
- **Normalización:** Rescalado a [0, 1]
- **Splitting:** División 80-20 de train-test
- **Batch Size:** 126 imágenes por batch

### 4) Entrenamiento del Modelo
```python
# Callbacks personalizados
class PararEnAccuracy(tf.keras.callbacks.Callback):
    """Detiene el entrenamiento cuando se alcanza un accuracy objetivo"""
    def __init__(self, objetivo=0.90, monitor="val_accuracy"):
        # Implementación personalizada

# Training
history = model.fit(
    train_generator,
    epochs=25,
    validation_data=validation_generator,
    callbacks=[callback_90, early_stopping]
)
```

**Resultados del entrenamiento inicial:**
- Epoch 3: `val_accuracy = 0.957` (≥ 0.90) → **Entrenamiento detenido**
- El callback personalizado optimiza el tiempo de entrenamiento

### 5) Validación y Pruebas
- Evaluación en conjunto de validación
- Visualización de curvas de accuracy
- Pruebas con imágenes personalizadas cargadas por el usuario

### 6) Expansión a 5 Categorías
- Reconfiguración de capa de salida: `Dense(5, activation='softmax')`
- Recopilación de imágenes de Lagarto y Spock
- Reentrenamiento del modelo completo
- Target: accuracy ≥ 80%

### 7) Implementación del Juego
- Clasificación automática de dos imágenes de entrada
- Aplicación de reglas del juego extendido:
  - **Piedra:** Aplasta Tijeras y Lagarto
  - **Papel:** Cubre Piedra y desacredita Spock
  - **Tijeras:** Corta Papel y decapita Lagarto
  - **Lagarto:** Come Papel y envenena Spock
  - **Spock:** Rompe Tijeras y desintegra Piedra

---

## Tecnologías y Librerías

```python
import tensorflow as tf      # Framework de Deep Learning
import keras                 # API de alto nivel para NN
from keras_preprocessing import image  # Procesamiento de imágenes
from keras_preprocessing.image import ImageDataGenerator  # Data Augmentation

import numpy as np           # Cálculos numéricos
import matplotlib.pyplot as plt  # Visualización
import os, shutil, glob, zipfile  # Utilidades del sistema
```

**Versiones recomendadas:**
- TensorFlow ≥ 2.14
- Python ≥ 3.8
- NumPy ≥ 1.21
- Matplotlib ≥ 3.5

---

## Cómo Usar

### Opción 1: Google Colab (Recomendado ⭐)

La forma más rápida y sin necesidad de instalar dependencias:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/williamG7/Rock-Paper-Scissors-Lizard-Spock/blob/main/Rock,_Paper,_Scissors,_Lizard,_Spock_GuzmanWilliam.ipynb)

**Ventajas:**
- ✅ GPU gratuita (aceleración de entrenamiento)
- ✅ Sin instalación de dependencias
- ✅ Montaje fácil de Google Drive
- ✅ Ejecución completa en el navegador

**Pasos:**
1. Abre el notebook en Colab usando el enlace anterior
2. Ejecuta las celdas secuencialmente
3. Sigue las instrucciones de carga de imágenes personalizadas
4. El modelo se entrenará y guardará automáticamente

### Opción 2: Ejecución Local

Si prefieres ejecutar en tu máquina local:

#### Requisitos Previos
- Python 3.8+
- pip o conda
- GPU (CUDA + cuDNN) - opcional pero recomendado

#### Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/williamG7/Rock-Paper-Scissors-Lizard-Spock.git
cd Rock-Paper-Scissors-Lizard-Spock

# 2. Crear entorno virtual (opcional pero recomendado)
python -m venv env
source env/bin/activate  # En Windows: env\Scripts\activate

# 3. Instalar dependencias
pip install tensorflow>=2.14
pip install keras-preprocessing
pip install numpy matplotlib
pip install jupyter
```

#### Ejecución

```bash
# Iniciar Jupyter Notebook
jupyter notebook

# Abre el archivo:
# Rock,_Paper,_Scissors,_Lizard,_Spock_GuzmanWilliam.ipynb
```

**Nota:** El notebook descargará automáticamente los datasets de Google Cloud en su primera ejecución.

---

## Estructura del Repositorio

```
Rock-Paper-Scissors-Lizard-Spock/
│
├── Rock,_Paper,_Scissors,_Lizard,_Spock_GuzmanWilliam.ipynb  # Notebook principal
├── README.md                                                  # Documentación
├── my_model.keras                                            # Modelo entrenado (opcional)
└── .gitignore
```

---

## Resultados y Métricas

### Fase 1: Rock, Paper, Scissors (3 categorías)

| Métrica | Valor |
|---------|-------|
| Epochs | 3 (parado por callback) |
| Training Accuracy | 67.12% |
| Validation Accuracy | **95.7%** ✅ |
| Loss (Training) | 0.85 |
| Loss (Validation) | 0.15 |

### Fase 2: Extensión a 5 categorías

| Métrica | Target | Status |
|---------|--------|--------|
| Validation Accuracy | ≥ 80% | 📊 En proceso |
| Categorías | 5 | ✅ Implementado |
| Imágenes Totales | 2,500+ | ✅ Recopiladas |

---

## Funcionalidades Clave

### 1. Callbacks Personalizados
```python
class PararEnAccuracy(tf.keras.callbacks.Callback):
    """
    Detiene el entrenamiento cuando se alcanza un accuracy objetivo.
    Optimiza tiempo de entrenamiento y evita overfitting.
    """
```

### 2. Data Augmentation
- Rotación ±40°
- Desplazamiento horizontal/vertical ±20%
- Zoom ±20%
- Flip horizontal

### 3. Modelo Modular
- Fácil de extender a más categorías
- Capas configurables
- Dropout para regularización

### 4. Juego Automatizado
- Clasificación en tiempo real
- Aplicación automática de reglas
- Determinación de ganador
  
---

## Troubleshooting

### Error: `keras_preprocessing not found`
```bash
pip install keras-preprocessing
```

### Error: GPU no detectada en Colab
```python
# Verificar GPU en Colab
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

### Las imágenes subidas no se detectan
- Asegúrate de que las imágenes sean PNG o JPG
- Verifica que el tamaño sea 150x150px
- Intenta con formato RGB (3 canales)

---

## Referencias

### Videos Tutoriales
- [Instructivo del Proyecto (YouTube)](https://youtu.be/u2TjZzNuly8)
- [Reglas del Juego Extendido (YouTube)](https://youtu.be/IFurn06BDuc?t=38)

### Documentación
- [TensorFlow/Keras Docs](https://www.tensorflow.org/api_docs)
- [Redes Neuronales Convolucionales (CNN)](https://en.wikipedia.org/wiki/Convolutional_neural_network)
- [Rock-Paper-Scissors-Lizard-Spock (Wikipedia)](https://en.wikipedia.org/wiki/Rock_paper_scissors#Variants)

### Datasets
- [Google Learning Datasets](https://storage.googleapis.com/learning-datasets/)
- [Rock, Paper, Scissors Dataset](https://storage.googleapis.com/learning-datasets/rps.zip)
- [Rock, Paper, Scissors Test Set](https://storage.googleapis.com/learning-datasets/rps-test-set.zip)

---

## Autor

**William Guzmán**

- 🔗 GitHub: [@williamG7](https://github.com/williamG7)

---

## Licencia

Este proyecto está bajo licencia **MIT** y es de libre distribución con fines educativos y de aprendizaje.

Si utilizas este proyecto en tu trabajo, se agradece:
- Mencionar el repositorio original
- Citar al autor
- Enlazar a este repositorio

---

## Agradecimientos

- 📚 Google por proporcionar los datasets de libre acceso
- 🎓 Comunidad de Machine Learning por los tutoriales
- 💡 Todos los que contribuyan con mejoras

---

## Estadísticas del Proyecto

![GitHub stars](https://img.shields.io/github/stars/williamG7/Rock-Paper-Scissors-Lizard-Spock?style=social)
![GitHub forks](https://img.shields.io/github/forks/williamG7/Rock-Paper-Scissors-Lizard-Spock?style=social)
![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14+-orange?logo=tensorflow)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)

---

**¿Te gustó este proyecto? ⭐ Dale una estrella en GitHub**

