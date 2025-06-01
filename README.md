# Clasificación de madurez de plátanos y tomates

Este repositorio contiene el código para el TP2 del Grupo 4 de Inteligencia Artificial (Miércoles Noche).

## Descripción
El objetivo de este proyecto es desarrollar modelos de inteligencia artificial capaces de clasificar el estado de madurez de plátanos y tomates a partir de imágenes. Se exploran diferentes técnicas de procesamiento de imágenes y aprendizaje automático para lograr una clasificación precisa.

## Estructura del repositorio
- **src/**: Scripts de entrenamiento, preprocesamiento y utilidades.
- **logs/**: Registros de ejecución, resultados y métricas generados por los scripts.
- **dataset/**: Conjunto de datos de imágenes organizados por tipo de fruta, partición (train/test) y clase de madurez.

## Estructura del dataset
```
TP2_ClasificacionFruta_Grupo4/
├── dataset/
│   ├── bananas/
│   │   ├── train/
│   │   │   ├── inmaduro/
│   │   │   ├── maduro/
│   │   │   ├── sobre-maduro/
│   │   │   └── descomposicion/
│   │   └── test/
│   │       ├── inmaduro/
│   │       ├── maduro/
│   │       ├── sobre-maduro/
│   │       └── descomposicion/
│   └── tomates/
│       ├── train/
│       │   ├── inmaduro/
│       │   ├── maduro/
│       │   ├── sobre-maduro/
│       │   └── descomposicion/
│       └── test/
│           ├── inmaduro/
│           ├── maduro/
│           ├── sobre-maduro/
│           └── descomposicion/
```

Cada carpeta de clase debe contiene las imágenes correspondientes a ese nivel de madurez para entrenar y clasificar el modelo.

## Requisitos
- Python 3.8+
- tensorflow[and-cuda]==2.19.0
- numpy
- pillow
- scipy
- scikit-learn
- pandas

## Instrucciones de uso
1. Clonar el repositorio:
   ```bash
   git clone https://github.com/TomasAMolinari/TP2_ClasificacionFruta_Grupo4.git
   cd TP2_ClasificacionFruta_Grupo4
   ```
2. Instalar dependencias:
   ```bash
   pip install -r src/requirements.txt
   ```
3. Ejecutar el script principal para entrenamiento y evaluación:
   ```bash
   python src/cnn_madurez.py --fruit_type bananas
   ```
   o para tomates:
   ```bash
   python src/cnn_madurez.py --fruit_type tomates
   ```
   
   Opcionalmente, para usar el mejor modelo previamente entrenado (si existe):
   ```bash
   python src/cnn_madurez.py --fruit_type bananas --use_best_model
   ```
   
## Escenarios

El proyecto cuenta con tres branches principales, cada uno dedicado a un enfoque de entrenamiento diferente:
- [**Escenario 1**](https://github.com/TomasAMolinari/TP2_ClasificacionFruta_Grupo4/tree/1er-Escenario): Red Convolucional (CNN) básica para clasificación directa de imágenes.
- [**Escenario 2**](https://github.com/TomasAMolinari/TP2_ClasificacionFruta_Grupo4/tree/2do-Escenario): Perceptrón Multicapa (MLP) sobre vectores de características extraídas de las imágenes.
- [**Escenario 3**](https://github.com/TomasAMolinari/TP2_ClasificacionFruta_Grupo4/tree/3er-Escenario): CNN con ajustes de hiperparámetros y topología para analizar el impacto en la generalización.

Cada branch contiene el código y resultados correspondientes a su enfoque.

## Créditos
- Grupo 4 - Inteligencia Artificial (Miércoles Noche)
- Integrantes:
    - Molinari, Tomás Agustín
    - Mouriño, Martín Ezequiel
