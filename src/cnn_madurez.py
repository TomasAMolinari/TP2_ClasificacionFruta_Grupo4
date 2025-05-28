import os  # manejo de archivos y rutas del sistema operativo
import numpy as np  # cálculos numéricos sobre arrays
from PIL import Image  # carga y transformación de imágenes

import tensorflow as tf  # framework de deep learning
from tensorflow.keras.models import Model  # API funcional para montar redes
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
)  # capas CNN básicas y densas
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # generación y augmentación de lotes de imagen
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping  # registro en TensorBoard y parada temprana
from tensorflow.keras.optimizers import Adam  # optimizador Adam con tasa ajustable


# BASE_DIR: carpeta donde vive este script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DATASET_DIR: ruta al directorio principal de datos
DATASET_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'dataset'))
# LOGS_DIR: destino de los logs para TensorBoard
LOGS_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'logs'))

# cuatro etiquetas de madurez usadas como clases de salida
NIVEL_DE_MADUREZ = [
    'inmaduro',
    'maduro',
    'sobre-maduro',
    'descomposicion'
]

# generadores de entrenamiento y validación (estructura esperada: train/<clase>/*.jpg)
def crear_generadores(
    data_dir=os.path.join(DATASET_DIR, 'train'),
    img_size=(150, 150),
    batch_size=32,
    val_split=0.2
):
    datagen = ImageDataGenerator(
        rescale=1./255,          # escala de píxeles a [0,1], acelerando aprendizaje
        rotation_range=40,       # rotaciones aleatorias hasta ±40°
        zoom_range=[0.75,1,25],  # zoom aleatorio entre 75% y 125%
        width_shift_range=0.2,   # desplazamientos horizontales hasta 20%
        height_shift_range=0.2,  # desplazamientos verticales hasta 20%
        horizontal_flip=True,    # inversión horizontal aleatoria
        vertical_flip=True,      # inversión vertical aleatoria
        validation_split=val_split  # 20% de datos para validación
    )

    train_gen = datagen.flow_from_directory(
        data_dir,                
        target_size=img_size,    # fuerza a las imágenes en 100×100 px
        batch_size=batch_size,   # lotes de 32 muestras
        classes=NIVEL_DE_MADUREZ, # orden de etiquetas fijo
        class_mode='categorical',# salida one-hot multiclase
        subset='training'        # partición de entrenamiento
    )

    val_gen = datagen.flow_from_directory(
        data_dir,                
        target_size=img_size,    
        batch_size=batch_size,   
        classes=NIVEL_DE_MADUREZ, 
        class_mode='categorical',
        subset='validation'      # partición de validación
    )

    return train_gen, val_gen  # listas de generadores listas para fit()

# bloque CNN reutilizable: dos convoluciones + pooling
def conv_block(x, filters):
    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)  # conv 3×3 + ReLU
    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)  # conv 3×3 + ReLU
    return MaxPooling2D((2, 2))(x)  # reduce tamaño espacial a la mitad

# ensamblado y compilación del modelo completo
def crear_modelo(input_shape=(150, 150, 3), n_classes=len(NIVEL_DE_MADUREZ)):
    inp = Input(shape=input_shape)                        # tensor de entrada 150×150×3
    x = conv_block(inp, 32)                               # bloque inicial (32 filtros)
    x = conv_block(x, 64)                                 # bloque intermedio (64 filtros)
    x = conv_block(x, 128)                                # bloque profundo (128 filtros)
    x = Flatten()(x)                                      # vectoriza mapas de características
    x = Dense(128, activation='relu')(x)                  # capa densa de 128 neuronas + ReLU
    x = Dropout(0.5)(x)                                   # dropout 50% para mitigar overfitting
    out = Dense(n_classes, activation='softmax')(x)       # salida softmax con 4 clases

    model = Model(inputs=inp, outputs=out)                # modelo definido con API funcional
    model.compile(
        optimizer=Adam(1e-4),                             # Adam con lr=0.0001
        loss='categorical_crossentropy',                  # entropía cruzada para multiclase
        metrics=['accuracy']                              # precisión como métrica principal
    )

    return model  # devuelve el modelo listo para entrenar

# inferencia sobre conjunto de prueba
def inferir_test(
    test_dir=os.path.join(DATASET_DIR, 'test'),
    img_size=(150, 150)
):
    # mapeo índice→etiqueta según entrenamiento
    label_map = {v: k for k, v in train_gen.class_indices.items()}

    for fruta in os.listdir(test_dir):                          # cada categoría en test
        fruta_dir = os.path.join(test_dir, fruta)               # ruta a carpeta de categoría
        if not os.path.isdir(fruta_dir):                        # omite archivos sueltos
            continue
        print(f"Resultados para {fruta}:")                      # título por categoría

        for ejemplar in os.listdir(fruta_dir):                  # cada muestra de la categoría
            ejemplar_dir = os.path.join(fruta_dir, ejemplar)    
            if not os.path.isdir(ejemplar_dir):                 # omite si no es subcarpeta
                continue

            # primer archivo con extensión válida
            img_file = next(
                (f for f in os.listdir(ejemplar_dir)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))),
                None
            )
            if not img_file:                                    # omite si faltan imágenes
                continue

            img = Image.open(os.path.join(ejemplar_dir, img_file)).convert('RGB')
            img = img.resize(img_size)                         # imagen a 150×150 px
            arr = np.expand_dims(np.array(img) / 255.0, 0)      # normaliza y añade dimensión batch

            pred = model.predict(arr)[0]                       # vector de probabilidades
            nivel = label_map[np.argmax(pred)]                 # etiqueta de mayor probabilidad

            print(f"  {ejemplar} => {nivel}")                  # resultado mostrado

# punto de entrada: entrenamiento y luego test
def main():
    global train_gen, model                                # accesibles en inferir_test
    train_gen, val_gen = crear_generadores()               # generadores preparados
    model = crear_modelo()                                 # modelo ensamblado y compilado
    model.summary()                                        # impresión de arquitectura

    tb_cb = TensorBoard(log_dir=LOGS_DIR, histogram_freq=1)  
    es = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )  # parada temprana tras 3 épocas sin mejora

    model.fit(
        train_gen,                                        # lote de entrenamiento
        validation_data=val_gen,                          # lote de validación
        epochs=4,                                        # entrenamiento en varias épocas
        callbacks=[tb_cb, es]                             # registro y parada temprana
    )

    print("\nInferencia en test:")
    inferir_test()                                      

if __name__ == '__main__':
    main()  