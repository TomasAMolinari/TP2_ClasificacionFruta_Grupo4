# Escenario 3

Este repositorio contiene el código para el TP2 del Grupo 4 de Inteligencia Artificial (Miércoles Noche), enfocado en el entrenamiento de una **Red Convolucional (CNN) básica** para clasificar el estado de madurez de plátanos y tomates a partir de imágenes. Para este escenario se modificaron los hiperparámetros y la topología con respecto al Escenario 1, para observar como cambia el comportamiento del modelo.

## Estructura del repositorio
- **src/**: Scripts de entrenamiento, preprocesamiento y utilidades.
- **logs/**: Registros de ejecución, resultados y métricas generados por los scripts.
- **dataset/**: Conjunto de datos de imágenes organizados por tipo de fruta, partición (train/test) y clase de madurez.

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

## Créditos
- Grupo 4 - Inteligencia Artificial (Miércoles Noche)
- Integrantes:
    - Molinari, Tomás Agustín
    - Mouriño, Martín Ezequiel
