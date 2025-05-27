import os  # importar módulo para manejo de sistema de archivos (rutas, listas de directorios)
import numpy as np  # importar NumPy para cálculos numéricos y manipulación de arrays
from PIL import Image  # importar Pillow para cargar y procesar imágenes

import tensorflow as tf  # importar TensorFlow, framework de deep learning
from tensorflow.keras.models import Model  # API funcional de Keras para definir modelos
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout  # capas de red neuronal CNN y dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # herramienta para augmentación y preprocesamiento de imágenes
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping  # callbacks para visualización y detención temprana
from tensorflow.keras.optimizers import Adam  # optimizador Adam para entrenar


# ruta absoluta al directorio donde está el script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ruta absoluta al dataset
DATASET_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'dataset'))
# ruta absoluta a la carpeta de logs
LOGS_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'logs'))

# 1) Definir los 4 niveles de maduración que serán las clases de salida
MATURITY_LEVELS = [
    'inmaduro',           # fruto verde (inmadura)
    'maduro',          # fruto maduro
    'sobre-maduro',    # fruto sobre-maduro
    'descomposicion'          # fruto en descomposición
]

# 2) Función para crear generadores de entrenamiento y validación
#    Asume estructura: dataset/train/<nivel>/*.jpg
def crear_generadores(data_dir=None, img_size=(150,150), batch_size=32, val_split=0.2):
    if data_dir is None:
        data_dir = os.path.join(DATASET_DIR, 'train')
    datagen = ImageDataGenerator(
        rescale=1./255,           # normalizar píxeles a rango [0,1]
        rotation_range=40,        # rotación aleatoria hasta 40 grados
        width_shift_range=0.2,    # desplazamiento horizontal aleatorio
        height_shift_range=0.2,   # desplazamiento vertical aleatorio
        horizontal_flip=True,     # volteo horizontal aleatorio
        vertical_flip=True,       # volteo vertical aleatorio
        validation_split=val_split # porcentaje de datos para validación
    )
    train_gen = datagen.flow_from_directory(
        data_dir,                 # carpeta raíz de entrenamiento
        target_size=img_size,     # redimensionar imágenes
        batch_size=batch_size,    # tamaño de batch
        classes=MATURITY_LEVELS,  # orden y nombres de clases
        class_mode='categorical', # clasificación multiclase one-hot
        subset='training'         # partición de entrenamiento
    )
    val_gen = datagen.flow_from_directory(
        data_dir,                 # misma carpeta para validación
        target_size=img_size,     # redimensionamiento igual que entrenamiento
        batch_size=batch_size,    # mismo tamaño de batch
        classes=MATURITY_LEVELS,  # mismo orden de clases
        class_mode='categorical', # misma modalidad
        subset='validation'       # partición de validación
    )
    return train_gen, val_gen  # devolver generadores

# 3) Bloque de convolución reutilizable: 2x Conv2D + MaxPooling2D
def conv_block(x, filters):
    x = Conv2D(filters, (3,3), activation='relu', padding='same')(x)  # primera convolución 3x3
    x = Conv2D(filters, (3,3), activation='relu', padding='same')(x)  # segunda convolución 3x3
    return MaxPooling2D((2,2))(x)  # reducir dimensiones espaciales a la mitad

# 4) Función para crear y compilar el modelo CNN completo
def crear_modelo(input_shape=(150,150,3), n_classes=len(MATURITY_LEVELS)):
    inp = Input(shape=input_shape)             # definir tensor de entrada
    x = conv_block(inp, 32)                    # bloque conv con 32 filtros
    x = conv_block(x, 64)                      # bloque conv con 64 filtros
    x = conv_block(x, 128)                     # bloque conv con 128 filtros
    x = Flatten()(x)                           # aplanar salida para capa densa
    x = Dense(128, activation='relu')(x)       # capa densa intermedia con activación ReLU
    x = Dropout(0.5)(x)                        # aplicar dropout 50% para evitar overfitting
    out = Dense(n_classes, activation='softmax')(x)  # capa de salida con softmax para clasificar
    model = Model(inputs=inp, outputs=out)     # crear modelo funcional
    model.compile(
        optimizer=Adam(1e-4),                 # optimizador Adam con lr=0.0001
        loss='categorical_crossentropy',      # función de pérdida para multiclase
        metrics=['accuracy']                  # métrica de evaluación: precisión
    )
    return model  # devolver modelo compilado

# 5) Función para inferir sobre los ejemplares en carpeta de test
def inferir_test(test_dir=None, img_size=(150,150)):
    if test_dir is None:
        test_dir = os.path.join(DATASET_DIR, 'test')
    for fruta in os.listdir(test_dir):                              # iterar frutas en test
        fruta_dir = os.path.join(test_dir, fruta)                    # ruta a carpeta de fruta
        if not os.path.isdir(fruta_dir):                            # saltar si no es carpeta
            continue
        print(f"Resultados para {fruta}:")                         # encabezado para cada fruta
        label_map = {v: k for k, v in train_gen.class_indices.items()}  # índice->etiqueta desde generador
        for ejemplar in os.listdir(fruta_dir):                       # iterar ejemplares
            ejemplar_dir = os.path.join(fruta_dir, ejemplar)         # ruta a carpeta de ejemplar
            if not os.path.isdir(ejemplar_dir):                      # saltar si no es carpeta
                continue
            img_file = next((f for f in os.listdir(ejemplar_dir)     # buscar primer archivo de imagen
                             if f.lower().endswith(('.png','.jpg','.jpeg'))), None)
            if not img_file:                                       # si no hay imagen, saltar
                continue
            img = Image.open(os.path.join(ejemplar_dir, img_file)).convert('RGB')  # cargar imagen y fuerza la imagen a formato RGB (evita errores por canal alfa en imágenes RGBA)
            img = img.resize(img_size)                             # redimensionar
            arr = np.expand_dims(np.array(img)/255.0, 0)           # normalizar y agregar dimensión batch
            pred = model.predict(arr)[0]                           # predecir vector softmax
            nivel = label_map[np.argmax(pred)]                     # seleccionar etiqueta más probable
            print(f"  {ejemplar} => {nivel}")                    # mostrar resultado

# 6) Función principal: entrenar red y luego inferir en test
def main():
    global train_gen, model  # declarar variables globales usadas en inferencia
    train_gen, val_gen = crear_generadores()  # crear generadores
    model = crear_modelo()                                    # construir y compilar modelo
    model.summary()                                           # mostrar arquitectura
    tb_cb = TensorBoard(log_dir=LOGS_DIR, histogram_freq=1)     # callback para TensorBoard
    es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)  # detener temprano
    model.fit(
        train_gen,                                            # datos de entrenamiento
        validation_data=val_gen,                              # datos de validación
        epochs=10,                                            # número de épocas
        callbacks=[tb_cb, es]                                 # callbacks configurados
    )
    print("\nInferencia en test:")
    inferir_test()                              # ejecutar inferencia en test

if __name__ == '__main__':
    main()  # ejecutar main al iniciar script
