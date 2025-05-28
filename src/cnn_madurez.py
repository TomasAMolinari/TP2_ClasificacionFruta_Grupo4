import os  # importar módulo para manejo de sistema de archivos (rutas, listas de directorios)
import numpy as np  # importar NumPy para cálculos numéricos y manipulación de arrays
from PIL import Image  # importar Pillow para cargar y procesar imágenes
import datetime # Para nombrar los directorios de logs

import tensorflow as tf  # importar TensorFlow, framework de deep learning
from tensorflow.keras.models import Model  # API funcional de Keras para definir modelos
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization  # capas de red neuronal CNN y dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # herramienta para augmentación y preprocesamiento de imágenes
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau  # callbacks para visualización y detención temprana
from tensorflow.keras.optimizers import Adam  # optimizador Adam para entrenar


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

    x = Conv2D(filters, (3,3), activation='relu', padding='same')(x)  # primera convolución 3x3
    x = BatchNormalization()(x) # Añadir BatchNormalization
    x = Conv2D(filters, (3,3), activation='relu', padding='same')(x)  # segunda convolución 3x3
    x = BatchNormalization()(x) # Añadir BatchNormalization
    return MaxPooling2D((2,2))(x)  # reducir dimensiones espaciales a la mitad

# 4) Función para crear y compilar el modelo CNN completo
def crear_modelo(input_shape=(150,150,3), n_classes=len(MATURITY_LEVELS)):
    inp = Input(shape=input_shape)             # definir tensor de entrada
    x = conv_block(inp, 32)                    # bloque conv con 32 filtros
    x = conv_block(x, 64)                      # bloque conv con 64 filtros
    x = conv_block(x, 128)                     # bloque conv con 128 filtros
    x = Flatten()(x)                           # aplanar salida para capa densa
    x = Dense(128, activation='relu')(x)       # capa densa intermedia con activación ReLU
    x = BatchNormalization()(x)                # añadir BatchNormalization
    x = Dropout(0.5)(x)                        # aplicar dropout 50% para evitar overfitting
    out = Dense(n_classes, activation='softmax')(x)  # capa de salida con softmax para clasificar
    model = Model(inputs=inp, outputs=out)     # crear modelo funcional

    model.compile(
        optimizer=Adam(1e-4),                             # Adam con lr=0.0001
        loss='categorical_crossentropy',                  # entropía cruzada para multiclase
        metrics=['accuracy']                              # precisión como métrica principal
    )

    return model  # devolver modelo compilado

# 5) Función para inferir sobre los ejemplares en carpeta de test
def inferir_test(test_dir=None, img_size=(150,150)):
    if test_dir is None:
        test_dir = os.path.join(DATASET_DIR, 'test')

    # asegurarse de que train_gen esté disponible para obtener class_indices
    if 'train_gen' not in globals() or train_gen is None:
        print("Error: train_gen no está definido globalmente o no se ha inicializado. "
              "No se pueden obtener las etiquetas de clase para la inferencia.")
        return
    
    label_map = {v: k for k, v in train_gen.class_indices.items()}  # índice->etiqueta desde generador

    resultados_por_fruta = {}
    total_evaluadas = 0
    correctas_global = 0

    for fruta_tipo in os.listdir(test_dir):                              # iterar tipos de frutas en test (ej: bananas, tomates)
        fruta_tipo_dir = os.path.join(test_dir, fruta_tipo)                # ruta a carpeta de tipo de fruta
        if not os.path.isdir(fruta_tipo_dir):                            # saltar si no es carpeta
            continue
        
        resultados_por_fruta[fruta_tipo] = []

        for ejemplar_nombre in os.listdir(fruta_tipo_dir):                   # iterar ejemplares de esa fruta
            ejemplar_dir = os.path.join(fruta_tipo_dir, ejemplar_nombre)     # ruta a carpeta de ejemplar
            if not os.path.isdir(ejemplar_dir):                              # saltar si no es carpeta
                continue
            
            # Extraer etiqueta real del nombre de la carpeta del ejemplar
            etiqueta_real = None
            # Intentar encontrar una coincidencia con MATURITY_LEVELS (ordenados por longitud descendente para casos como 'sobre-maduro' vs 'maduro')
            sorted_maturity_levels = sorted(MATURITY_LEVELS, key=len, reverse=True)
            for level in sorted_maturity_levels:
                if ejemplar_nombre.endswith(f"_{level}"):
                    etiqueta_real = level
                    break

            img_file = next((f for f in os.listdir(ejemplar_dir)             # buscar primer archivo de imagen
                             if f.lower().endswith(('.png','.jpg','.jpeg'))), None)
            
            current_res = {
                'ejemplar': ejemplar_nombre,
                'etiqueta_real': etiqueta_real if etiqueta_real else 'Desconocida',
                'prediccion_madurez': 'N/A',
                'probabilidad': 0
            }

            if not img_file:
                current_res['prediccion_madurez'] = 'Error - No se encontró imagen'
                resultados_por_fruta[fruta_tipo].append(current_res)
                continue
            
            try:
                img = Image.open(os.path.join(ejemplar_dir, img_file)).convert('RGB')
                img_resized = img.resize(img_size)
                arr = np.expand_dims(np.array(img_resized)/255.0, 0)
                pred_vector = model.predict(arr)[0]
                pred_idx = np.argmax(pred_vector)
                nivel_madurez_pred = label_map[pred_idx]
                probabilidad = pred_vector[pred_idx]

                current_res['prediccion_madurez'] = nivel_madurez_pred
                current_res['probabilidad'] = float(probabilidad)

                if etiqueta_real and etiqueta_real != 'Desconocida': # Solo contar para accuracy si hay etiqueta real válida
                    total_evaluadas += 1
                    if nivel_madurez_pred == etiqueta_real:
                        correctas_global += 1
                
            except Exception as e:
                print(f"Error procesando {os.path.join(ejemplar_dir, img_file)}: {e}")
                current_res['prediccion_madurez'] = f'Error - {e}'
            
            resultados_por_fruta[fruta_tipo].append(current_res)

    # Imprimir los resultados de fo
    if not resultados_por_fruta:
        print("No se encontraron imágenes para procesar en el directorio de test.")
        return

    for fruta_tipo, ejemplares in resultados_por_fruta.items():
        print(f"\nFruta Tipo: {fruta_tipo}")
        if not ejemplares:
            print("  No se encontraron ejemplares o imágenes para este tipo de fruta.")
            continue
        for res in ejemplares:
            print(f"  Ejemplar: {res['ejemplar']:<30} | Real: {res['etiqueta_real']:<15} | Pred: {res['prediccion_madurez']:<15} (Prob: {res['probabilidad']:.2f})")
    
    print("\n--- Resumen de Evaluación en Test ---")
    if total_evaluadas > 0:
        accuracy_global = (correctas_global / total_evaluadas) * 100
        print(f"Imágenes evaluadas (con etiqueta real conocida): {total_evaluadas}")
        print(f"Predicciones correctas: {correctas_global}")
        print(f"Precisión Global en Test: {accuracy_global:.2f}%")
    else:
        print("No se pudieron evaluar imágenes con etiquetas reales conocidas.")
        print("Asegúrate de que los nombres de las carpetas de ejemplares en 'dataset/test' sigan la convención: nombre_ejemplar_etiquetaverdadera")

    print("-------------------------------------------------------")

# 6) Función principal: entrenar red y luego inferir en test
def main():
    global train_gen, model  # declarar variables globales usadas en inferencia
    train_gen, val_gen = crear_generadores()  # crear generadores
    model = crear_modelo()                                    # construir y compilar modelo
    model.summary()                                           # mostrar arquitectura
    
    # directorio de logs único para cada ejecución
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_log_dir = os.path.join(LOGS_DIR, current_time)
    os.makedirs(run_log_dir, exist_ok=True) # Crear directorio si no existe

    # Callbacks
    tb_cb = TensorBoard(log_dir=run_log_dir, histogram_freq=1)     # callback para TensorBoard
    es_cb = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)  # detener temprano
    # Guardar el mejor modelo en el directorio de logs de la ejecución actual
    best_model_filename = 'best_model.keras' # Nombre de archivo consistente
    mc_cb = ModelCheckpoint(                                  # callback para guardar el mejor modelo
        filepath=os.path.join(run_log_dir, best_model_filename),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    lr_scheduler_cb = ReduceLROnPlateau(                     # callback para reducir LR si no hay mejora
        monitor='val_loss',
        factor=0.2,  # factor por el cual se reduce la LR: new_lr = lr * factor
        patience=5,  # número de épocas sin mejora antes de reducir LR
        min_lr=1e-6, # LR mínima
        verbose=1
    )
    
    callbacks_list = [tb_cb, es_cb, mc_cb, lr_scheduler_cb]

    model.fit(
        train_gen,                                            # datos de entrenamiento
        validation_data=val_gen,                              # datos de validación
        epochs=50,                                            # número de épocas (aumentado)
        callbacks=callbacks_list                              # callbacks configurados
    )

    print("\nInferencia en test:")
    # Cargar el mejor modelo guardado para la inferencia
    print("Cargando el mejor modelo guardado para inferencia...")
    # La ruta al mejor modelo ahora incluye el subdirectorio de la ejecución
    # Para encontrar el último modelo entrenado, necesitaríamos una lógica más compleja
    # o asumir que el usuario especifica qué modelo cargar.
    # Por ahora, vamos a cargar desde la ruta donde se guardó en esta ejecución.
    best_model_path = os.path.join(run_log_dir, best_model_filename)
    if os.path.exists(best_model_path):
        model = tf.keras.models.load_model(best_model_path)
    else:
        print(f"Advertencia: No se encontró el archivo {best_model_filename} en {run_log_dir}. Usando el último modelo entrenado en memoria.")

    inferir_test()                              # ejecutar inferencia en test

if __name__ == '__main__':
    main()  