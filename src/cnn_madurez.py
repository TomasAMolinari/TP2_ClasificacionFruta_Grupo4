import os  # importar módulo para manejo de sistema de archivos (rutas, listas de directorios)
import numpy as np  # importar NumPy para cálculos numéricos y manipulación de arrays
from PIL import Image  # importar Pillow para cargar y procesar imágenes
import datetime # Para nombrar los directorios de logs
import argparse # Para manejar argumentos de línea de comandos
from sklearn.utils import class_weight # Para calcular pesos de clase

# Verbosidad de TensorFlow (0 = todos, 1 = filtrar INFO, 2 = filtrar INFO y WARNING, 3 = filtrar INFO, WARNING, y ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
    batch_size=16,
    val_split=0.2
):
    datagen = ImageDataGenerator(
        rescale=1./255,          # escala de píxeles a [0,1], acelerando aprendizaje
        rotation_range=30,       # rotaciones aleatorias hasta ±30° (reducido de 60)
        zoom_range=[0.8, 1.2],   # zoom aleatorio entre 80% y 120% (reducido de [0.15,1.85])
        width_shift_range=0.15,  # desplazamientos horizontales hasta 15% (reducido de 0.5)
        height_shift_range=0.15, # desplazamientos verticales hasta 15% (reducido de 0.5)
        horizontal_flip=True,    # inversión horizontal aleatoria
        validation_split=val_split  # 20% de datos para validación
    )

    train_gen = datagen.flow_from_directory(
        data_dir,                
        target_size=img_size,    # fuerza a las imágenes en 100×100 px
        batch_size=batch_size,   # lotes de 16 muestras
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

    x = Conv2D(filters, (3,3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)  # primera convolución 3x3
    x = BatchNormalization()(x) # Añadir BatchNormalization
    x = Conv2D(filters, (3,3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)  # segunda convolución 3x3
    x = BatchNormalization()(x) # Añadir BatchNormalization
    return MaxPooling2D((2,2))(x)  # reducir dimensiones espaciales a la mitad

# 4) Función para crear y compilar el modelo CNN completo
def crear_modelo(input_shape=(150,150,3), n_classes=len(NIVEL_DE_MADUREZ)):
    inp = Input(shape=input_shape)             # definir tensor de entrada
    x = conv_block(inp, 128)  
    x = conv_block(x, 64)                      
    x = conv_block(x, 32)                      
    x = Flatten()(x)                           # aplanar salida para capa densa
    x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)       # capa densa intermedia con activación ReLU
    x = BatchNormalization()(x)                # añadir BatchNormalization
    x = Dropout(0.4)(x)                        # aplicar dropout 40% para evitar overfitting (aumentado de 0.2)
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
            # Intentar encontrar una coincidencia con NIVEL_DE_MADUREZ (ordenados por longitud descendente para casos como 'sobre-maduro' vs 'maduro')
            sorted_maturity_levels = sorted(NIVEL_DE_MADUREZ, key=len, reverse=True)
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

    parser = argparse.ArgumentParser(description='Entrenar y evaluar una CNN para clasificación de madurez de frutas.')
    parser.add_argument('--use_best_model', action='store_true',
                        help='Si se especifica, intenta cargar el mejor modelo guardado previamente. Si no, reentrena por defecto.')
    args = parser.parse_args()

    train_gen, val_gen = crear_generadores()  # crear generadores
    model = crear_modelo()                                    # construir y compilar modelo
    model.summary()                                           # mostrar arquitectura
    
    BEST_MODEL_PATH = None
    load_model_attempted = False

    if args.use_best_model:
        print_color("Intentando cargar el mejor modelo pre-entrenado...", color="amarillo")
        all_runs = sorted(os.listdir(LOGS_DIR)) if os.path.exists(LOGS_DIR) else []
        for run in reversed(all_runs):
            candidate = os.path.join(LOGS_DIR, run, 'best_model.keras')
            if os.path.exists(candidate):
                BEST_MODEL_PATH = candidate
                break
        
        if BEST_MODEL_PATH:
            try:
                model = tf.keras.models.load_model(BEST_MODEL_PATH)
                print_color(f"Modelo pre-entrenado cargado desde: {BEST_MODEL_PATH}. Saltando entrenamiento.", color="verde")
                load_model_attempted = True
            except Exception as e:
                print_color(f"Error al cargar el modelo desde {BEST_MODEL_PATH}: {e}. Se procederá a entrenar.", color="rojo")
                BEST_MODEL_PATH = None # Asegurar que no se intente usar un modelo corrupto
        else:
            print_color("No se encontró un modelo pre-entrenado. Se procederá a entrenar.", color="amarillo")
    
    if not load_model_attempted or not BEST_MODEL_PATH:
        if not args.use_best_model:
             print_color("No se especificó --use_best_model o no se encontró/pudo cargar. Entrenando nuevo modelo...", color="magenta")
        
        # Calcular class weights
        classes = np.array(train_gen.classes) # Obtener todas las etiquetas de clase del generador
        # Asegurarse de que las etiquetas sean numéricas para compute_class_weight
        # train_gen.classes devuelve los índices de clase para cada muestra, que es lo que necesitamos
        
        class_weights_calculated = class_weight.compute_class_weight(
            'balanced',
            classes=np.unique(classes), # Clases únicas presentes en los datos
            y=classes                   # Todas las etiquetas de clase
        )
        # Crear el diccionario de class_weight para model.fit
        # train_gen.class_indices mapea nombres de clase a índices: {'descomposicion': 0, 'inmaduro': 1, ...}
        # Necesitamos mapear los pesos calculados (que están en el orden de np.unique(classes)) a los índices correctos
        class_weights_dict = dict(enumerate(class_weights_calculated))
        
        print_color(f"Pesos de clase calculados: {class_weights_dict}", color="cyan")

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_log_dir  = os.path.join(LOGS_DIR, current_time)
        os.makedirs(run_log_dir, exist_ok=True)

        # callbacks solo si vamos a entrenar
        tb_cb = TensorBoard(log_dir=run_log_dir, histogram_freq=1)
        es_cb = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
        mc_cb = ModelCheckpoint(filepath=os.path.join(run_log_dir, 'best_model.keras'), monitor='val_loss', save_best_only=True, verbose=1)
        lr_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)
        callbacks_list = [tb_cb, es_cb, mc_cb, lr_cb]

        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=10, # Ajustar epochs según sea necesario
            callbacks=callbacks_list,
            class_weight=class_weights_dict # Aplicar pesos de clase
        )
        # Guardar explícitamente el modelo después del entrenamiento si no se usó ModelCheckpoint para el mejor modelo
        # o si se quiere guardar el modelo final independientemente.
        # model.save(os.path.join(run_log_dir, 'final_model.keras')) 
        print_color("Entrenamiento completado.", color="verde")

    # siempre inferir después
    print_color("Inferencia en test:", color="cyan")
    inferir_test()                            

# Helper para imprimir con colores
def print_color(text, color="default"):
    colors = {
        "default": "[0m",
        "rojo": "[31m",
        "verde": "[32m",
        "amarillo": "[33m",
        "azul": "[34m",
        "magenta": "[35m",
        "cyan": "[36m",
    }
    end_color = colors["default"]
    start_color = colors.get(color.lower(), end_color)
    print(start_color + text + end_color)

if __name__ == '__main__':
    main()  