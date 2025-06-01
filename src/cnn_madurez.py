import os  # importar módulo para manejo de sistema de archivos (rutas, listas de directorios)
import numpy as np  # importar NumPy para cálculos numéricos y manipulación de arrays
from PIL import Image  # importar Pillow para cargar y procesar imágenes
import datetime # Para nombrar los directorios de logs
import argparse # Para manejar argumentos de línea de comandos
from sklearn.utils import class_weight # Para calcular pesos de clase
from pathlib import Path # Para manejo de rutas de forma robusta y multiplataforma
import time # Para medir la duración del entrenamiento

# Verbosidad de TensorFlow (0 = todos, 1 = filtrar INFO, 2 = filtrar INFO y WARNING, 3 = filtrar INFO, WARNING, y ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- CONFIGURACIÓN GLOBAL DE TENSORFLOW Y GPU ---
import tensorflow as tf  # importar TensorFlow, framework de deep learning

# Configurar memory growth para GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs configured for memory growth.")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(f"Error setting memory growth: {e}")

from tensorflow.keras.models import Model  # API funcional de Keras para definir modelos
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization  # capas de red neuronal CNN y dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # herramienta para augmentación y preprocesamiento de imágenes
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau  # callbacks para visualización y detención temprana
from tensorflow.keras.optimizers import Adam  # optimizador Adam para entrenar
from sklearn.metrics import confusion_matrix
import pandas as pd


# PROJECT_ROOT: carpeta donde del script
PROJECT_ROOT = Path(__file__).resolve().parent
# Rutas raíz por defecto, las rutas específicas por fruta se derivarán en main()
DATASET_ROOT_DEFAULT = (PROJECT_ROOT / ".." / "dataset").resolve()
LOGS_ROOT_DEFAULT = (PROJECT_ROOT / ".." / "logs").resolve()

# Variables globales que se establecerán en main() según el tipo de fruta
DATASET_DIR: Path = None
LOGS_DIR: Path = None
train_gen = None # Necesario globalmente para inferir_test
model = None # Necesario globalmente para inferir_test


# --- DEFINICIONES GLOBALES DE RUTAS Y CONSTANTES ---

# NIVEL_DE_MADUREZ
NIVEL_DE_MADUREZ = [
    'inmaduro',
    'maduro',
    'sobre-maduro',
    'descomposicion'
]

# --- FUNCIÓN PARA CREAR GENERADORES DE DATOS ---
# generadores de entrenamiento y validación (estructura esperada: train/<clase>/*.jpg)
def crear_generadores(
    data_dir_param=None, # Se resolverá a DATASET_DIR / 'train' si es None
    img_size=(200, 200),
    batch_size=32,
    val_split=0.2
):
    actual_data_dir = data_dir_param if data_dir_param else DATASET_DIR / 'train'
    if not actual_data_dir.exists():
        print_color(f"Directorio de datos de entrenamiento no encontrado: {actual_data_dir}", color="rojo")
        print_color("Asegúrate de que la estructura es: dataset/<tipo_fruta>/train/<clase>/imagenes...", color="amarillo")
        return None, None # Devuelve None si el directorio no existe

    datagen = ImageDataGenerator(
        rescale=1./255,             # escala de píxeles a [0,1], acelerando aprendizaje
        rotation_range=60,          # rotaciones aleatorias hasta ±60°
        zoom_range=[0.5, 1.5],      # zoom aleatorio entre 50% y 150%
        width_shift_range=0.3,     # desplazamientos horizontales hasta 30%
        height_shift_range=0.3,    # desplazamientos verticales hasta 30%
        horizontal_flip=True,       # inversión horizontal aleatoria
        vertical_flip=True,         # inversión vertical aleatoria
        validation_split=val_split  # % de datos para validación
    )

    try:
        train_generator = datagen.flow_from_directory(
            actual_data_dir,
            target_size=img_size,       # fuerza a las imágenes en 100×100 px
            batch_size=batch_size,      # lotes de 16 muestras
            classes=NIVEL_DE_MADUREZ,   # orden de etiquetas fijo
            class_mode='categorical',   # salida one-hot multiclase
            subset='training'           # partición de entrenamiento
        )

        val_generator = datagen.flow_from_directory(
            actual_data_dir,
            target_size=img_size,
            batch_size=batch_size,
            classes=NIVEL_DE_MADUREZ,
            class_mode='categorical',
            subset='validation'      # partición de validación
        )
        return train_generator, val_generator # listas de generadores listas para fit()
    except FileNotFoundError:
        print_color(f"Error: No se pudo encontrar el directorio {actual_data_dir} o está vacío.", color="rojo")
        print_color("Verifica la ruta y que contenga subdirectorios de clase.", color="amarillo")
        return None, None
    except Exception as e:
        print_color(f"Ocurrió un error inesperado al crear generadores: {e}", color="rojo")
        return None, None


# --- DEFINICIÓN DEL MODELO CNN ---
# bloque CNN reutilizable: dos convoluciones + pooling
def conv_block(x, filters):

    x = Conv2D(filters, (3,3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)  # primera convolución 3x3
    x = BatchNormalization()(x) # Añadir BatchNormalization
    x = Conv2D(filters, (3,3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)  # segunda convolución 3x3
    x = BatchNormalization()(x) # Añadir BatchNormalization
    return MaxPooling2D((2,2))(x)  # reducir dimensiones espaciales a la mitad

# 4) Función para crear y compilar el modelo CNN completo
def crear_modelo(input_shape=(200,200,3), n_classes=len(NIVEL_DE_MADUREZ)):
    inp = Input(shape=input_shape)                                                                  # definir tensor de entrada
    x = conv_block(inp, 32)
    x = conv_block(x, 64)
    x = Flatten()(x)                                                                               # aplanar salida para capa densa
    x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)       # capa densa intermedia con activación ReLU
    x = BatchNormalization()(x)                                                                    # añadir BatchNormalization                                                                          # aplicar dropout 40% para evitar overfitting (aumentado de 0.2)
    out = Dense(n_classes, activation='softmax')(x)                                                # capa de salida con softmax para clasificar
    model_cnn = Model(inputs=inp, outputs=out)                                                     # crear modelo funcional

    model_cnn.compile(
        optimizer=Adam(1e-4),                             # Adam con lr=0.0001
        loss='categorical_crossentropy',                  # entropía cruzada para multiclase
        metrics=['accuracy']                              # precisión como métrica principal
    )

    return model_cnn  # devolver modelo compilado

# --- FUNCIÓN DE INFERENCIA EN EL CONJUNTO DE TEST ---
# 5) Función para inferir sobre los ejemplares en carpeta de test
def inferir_test(test_dir_param=None, img_size=(200,200), fruit_type_arg="N/A"):
    actual_test_dir = test_dir_param if test_dir_param else DATASET_DIR / 'test'
    if not actual_test_dir.is_dir():
        print_color(f"[ERROR] Directorio de test no encontrado para {fruit_type_arg}: {actual_test_dir}", color="rojo")
        print_color(f"        Asegúrate de que la estructura es: {DATASET_DIR / 'test'}/<CLASE_REAL>/<IMAGEN_EJEMPLAR.jpg>", color="amarillo")
        return

    if 'train_gen' not in globals() or train_gen is None:
        print_color("[ERROR] train_gen no definido. No se pueden obtener etiquetas para inferencia.", color="rojo")
        return
    
    global model
    if model is None:
        print_color("[ERROR] Modelo no cargado o entrenado. No se puede inferir.", color="rojo")
        return

    label_map = {v: k for k, v in train_gen.class_indices.items()}
    stats_por_clase_real = {level: {'total': 0, 'correctas': 0, 'predicciones': {k: 0 for k in NIVEL_DE_MADUREZ}} for level in NIVEL_DE_MADUREZ}
    stats_por_clase_real['Desconocida'] = {'total': 0, 'correctas': 0, 'predicciones': {k: 0 for k in NIVEL_DE_MADUREZ}}
    
    unrecognized_dirs_warnings = {} # Para warnings de directorios no mapeados
    errores_procesamiento_general = 0
    total_imagenes_clases_conocidas = 0
    correctas_global_clases_conocidas = 0
    
    y_true = [] # Valores para la matriz de confusión
    y_pred = [] # Valores para la matriz de confusión
    for clase_dir in actual_test_dir.iterdir():
        if not clase_dir.is_dir():
            continue
        
        etiqueta_real_clase_original = clase_dir.name
        clase_para_stats = 'Desconocida'
        
        for img_file_path in clase_dir.iterdir():
            if not (img_file_path.is_file() and img_file_path.suffix.lower() in ('.png','.jpg','.jpeg')):
                continue

        if etiqueta_real_clase_original in NIVEL_DE_MADUREZ:
            clase_para_stats = etiqueta_real_clase_original
        else:
            if etiqueta_real_clase_original not in unrecognized_dirs_warnings:
                unrecognized_dirs_warnings[etiqueta_real_clase_original] = 0
        
        imagenes_en_esta_clase_dir = 0
        for img_file_path in clase_dir.iterdir():
            if not (img_file_path.is_file() and img_file_path.suffix.lower() in ('.png','.jpg','.jpeg')):
                continue
            
            imagenes_en_esta_clase_dir += 1
            stats_por_clase_real[clase_para_stats]['total'] += 1
            if clase_para_stats != 'Desconocida':
                total_imagenes_clases_conocidas +=1
            else: # Si es desconocida por el nombre del dir
                unrecognized_dirs_warnings[etiqueta_real_clase_original] +=1

            try:
                img = Image.open(img_file_path).convert('RGB')
                img_resized = img.resize(img_size)
                arr = np.expand_dims(np.array(img_resized)/255.0, 0)
                pred_vector = model.predict(arr, verbose=0)[0] 
                pred_idx = np.argmax(pred_vector)
                nivel_madurez_pred = label_map[pred_idx]
                y_true.append(NIVEL_DE_MADUREZ.index(etiqueta_real_clase_original))
                y_pred.append(NIVEL_DE_MADUREZ.index(nivel_madurez_pred))
                     
                stats_por_clase_real[clase_para_stats]['predicciones'][nivel_madurez_pred] += 1

                if clase_para_stats != 'Desconocida':
                    if nivel_madurez_pred == clase_para_stats:
                        stats_por_clase_real[clase_para_stats]['correctas'] += 1
                        correctas_global_clases_conocidas += 1
            except Exception: # Simplificado, error ya se logea si es necesario arriba o se ignora
                errores_procesamiento_general += 1 # Contar errores generales si es necesario reportarlos

    # --- INICIO DE LOGEO ---

    # 0. Advertencias de directorios no reconocidos
    if unrecognized_dirs_warnings:
        for dir_name, count in unrecognized_dirs_warnings.items():
            if count > 0: # Solo mostrar si realmente hubo imágenes en ese dir no reconocido
                 print_color(f"[WARNING] Directorio '{dir_name}' no reconocido → agrupado como 'Desconocida' ({count} imágenes).", "amarillo")
        print("") # Salto de línea después de las advertencias

    print(f"=== Resultados de Inferencia: {fruit_type_arg.upper()} ===")
    print("")

    # 1. Resumen General
    print("1. Resumen General")
    if total_imagenes_clases_conocidas > 0:
        print(f"   • Clases conocidas: {total_imagenes_clases_conocidas} imágenes")
        acc_global = (correctas_global_clases_conocidas / total_imagenes_clases_conocidas * 100) if total_imagenes_clases_conocidas > 0 else 0
        print(f"     – Correctas: {correctas_global_clases_conocidas} → Precisión global: {acc_global:.2f}%")
    else:
        print("   • Clases conocidas: 0 imágenes")

    if stats_por_clase_real['Desconocida']['total'] > 0:
        print(f"   • Imágenes 'Desconocida': {stats_por_clase_real['Desconocida']['total']}")
        pred_desconocida_resumen_parts = []
        for pred_label, count in sorted(stats_por_clase_real['Desconocida']['predicciones'].items()):
            if count > 0:
                pred_desconocida_resumen_parts.append(f"{pred_label}={count}")
        if pred_desconocida_resumen_parts:
            print(f"     – Predicciones: {', '.join(pred_desconocida_resumen_parts)}")
    print("")

    # 2. Desglose por Clase
    print("2. Desglose por Clase")
    for nivel_real_key in NIVEL_DE_MADUREZ: # Iterar en el orden definido
        stats = stats_por_clase_real[nivel_real_key]
        if stats['total'] == 0: # No mostrar clases si no hubo imágenes de test para ellas
            continue

        # Adaptar nombre para mostrar (ej. 'descomposicion' a 'descomposición')
        nombre_clase_display = nivel_real_key.replace("descomposicion", "descomposición")

        print(f"\n   • {nombre_clase_display}")
        num_predichas_clase = sum(stats['predicciones'].values())
        acc_clase = (stats['correctas'] / num_predichas_clase * 100) if num_predichas_clase > 0 else 0
        print(f"     – Total: {stats['total']} | Correctas: {stats['correctas']} ({acc_clase:.2f}%)")
        
        pred_str_parts = []
        for nivel_pred, count in sorted(stats['predicciones'].items()):
            if count > 0:
                # Adaptar nombre para mostrar en predicciones también
                nivel_pred_display = nivel_pred.replace("descomposicion", "descomposición")
                pred_str_parts.append(f"{nivel_pred_display}={count}")
        if pred_str_parts:
             print(f"     – Predicciones: {', '.join(pred_str_parts)}")
    print("")

    # 3. Predicciones para "Desconocida" (si aplica)
    if stats_por_clase_real['Desconocida']['total'] > 0:
        print("3. Predicciones para 'Desconocida'")
        print(f"   • Total: {stats_por_clase_real['Desconocida']['total']}")
        pred_desconocida_detalle_parts = []
        for pred_label, count in sorted(stats_por_clase_real['Desconocida']['predicciones'].items()):
            if count > 0:
                 # Adaptar nombre para mostrar en predicciones también
                pred_label_display = pred_label.replace("descomposicion", "descomposición")
                pred_desconocida_detalle_parts.append(f"{pred_label_display}={count}")
        if pred_desconocida_detalle_parts:
            print(f"   • Predicciones: {', '.join(pred_desconocida_detalle_parts)}")
        print("")
        
    if errores_procesamiento_general > 0:
        print_color(f"[INFO] Se produjeron {errores_procesamiento_general} errores durante el procesamiento de algunas imágenes.", "amarillo")
        print("")

    print("=== Fin de la inferencia ===")
    # ==== Matriz de confusión ====
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(NIVEL_DE_MADUREZ))))
    df_cm = pd.DataFrame(cm, index=NIVEL_DE_MADUREZ, columns=NIVEL_DE_MADUREZ)

    # --- LOG MATRIZ CONFUSION EN FORMATO TABLA ---
    print(f"\n=== Matriz de Confusión: {fruit_type_arg.upper()} ===\n")
    print("3. Matriz de Confusión\n")

    col_names_display = [c.replace('descomposicion', 'descomposición') for c in NIVEL_DE_MADUREZ]
    
    real_col_width = 14 
    pred_col_width = 14 
    total_col_width = 7  # Ancho para la columna "Total"
    aciertos_col_width = 18 # Ancho para la columna "Aciertos (%)"
    # La columna "Errores (desglose)" tomará el espacio restante, pero su inicio debe estar alineado.

    # Cabecera
    header_line = f"{'Real ↓':<{real_col_width}} {'Predichas →':<12}" 
    for col_name in col_names_display:
        header_line += f" {col_name:>{pred_col_width}}"
    
    header_line += f" | {'Total':>{total_col_width}} | {'Aciertos (%)':>{aciertos_col_width}} | Errores (desglose)"
    print(header_line)
    print("─" * len(header_line)) 

    # Filas
    for i, real_class_key in enumerate(NIVEL_DE_MADUREZ):
        real_class_display = real_class_key.replace('descomposicion', 'descomposición')
        row_str = f"{real_class_display:<{real_col_width}} {'':<12}" # Espacio para alinear bajo "Predichas ->"
        
        total_real = 0
        aciertos_real = 0
        errores_desglose_parts = []

        for j, pred_class_key in enumerate(NIVEL_DE_MADUREZ):
            valor_celda = int(df_cm.loc[real_class_key, pred_class_key])
            row_str += f" {valor_celda:>{pred_col_width}}"
            total_real += valor_celda
            if real_class_key == pred_class_key:
                aciertos_real = valor_celda
            elif valor_celda > 0:
                pred_class_display = pred_class_key.replace('descomposicion', 'descomposición')
                errores_desglose_parts.append(f"{pred_class_display}={valor_celda}")
        
        porcentaje_aciertos = (aciertos_real / total_real * 100) if total_real > 0 else 0
        aciertos_str = f"{aciertos_real} ({porcentaje_aciertos:.2f} %)"
        errores_str = ", ".join(errores_desglose_parts) if errores_desglose_parts else "-"

        # Ajustar el espaciado para las últimas columnas aquí
        row_str += f" | {total_real:>{total_col_width}} | {aciertos_str:>{aciertos_col_width}} | {errores_str}"
        print(row_str)

    print(f"\n=== Fin de la Matriz de Confusión ===\n")

# --- FUNCIÓN PRINCIPAL (MAIN) ---
# 6) Función principal: entrenar red y luego inferir en test
def main():
    global DATASET_DIR, LOGS_DIR, train_gen, model  # declarar variables globales usadas en inferencia

    parser = argparse.ArgumentParser(description='Entrenar y evaluar una CNN para clasificación de madurez de frutas.')
    parser.add_argument('--fruit_type', type=str, default='bananas', choices=['bananas', 'tomates'],
                        help='Tipo de fruta para entrenar/evaluar (ej: bananas, tomates). Default: bananas')
    parser.add_argument('--use_best_model', action='store_true',
                        help='Si se especifica, intenta cargar el mejor modelo guardado previamente para la fruta. Si no, reentrena por defecto.')
    args = parser.parse_args()
    # === CONFIGURACIÓN INICIAL BASADA EN ARGUMENTOS ===
    fruit_type_arg = args.fruit_type
    print_color(f"Procesando para el tipo de fruta: {fruit_type_arg}", color="magenta")

    # Establecer rutas específicas para la fruta
    DATASET_DIR = DATASET_ROOT_DEFAULT / fruit_type_arg
    LOGS_DIR = LOGS_ROOT_DEFAULT / fruit_type_arg

    # Crear directorios si no existen
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    train_gen, val_gen = crear_generadores()  # crear generadores (usa global DATASET_DIR)
    if train_gen is None or val_gen is None:
        print_color(f"No se pudieron crear los generadores de datos para {fruit_type_arg}. Abortando.", color="rojo")
        return

    # === CREACIÓN O CARGA DEL MODELO ===
    model = crear_modelo()                                    # construir y compilar modelo
    model.summary()                                           # mostrar arquitectura
    
    BEST_MODEL_PATH = None
    load_model_attempted = False

    if args.use_best_model:
        print_color(f"Intentando cargar el mejor modelo pre-entrenado para '{fruit_type_arg}'...", color="amarillo")
        # Buscar el modelo en el LOGS_DIR específico de la fruta
        if LOGS_DIR.exists():
            all_runs = sorted([d for d in LOGS_DIR.iterdir() if d.is_dir()], reverse=True) # Más recientes primero
            for run_dir in all_runs:
                candidate = run_dir / 'best_model.keras'
                if candidate.exists():
                    BEST_MODEL_PATH = candidate
                    break
        
        if BEST_MODEL_PATH:
            try:
                model = tf.keras.models.load_model(BEST_MODEL_PATH)
                print_color(f"Modelo pre-entrenado para '{fruit_type_arg}' cargado desde: {BEST_MODEL_PATH}. Saltando entrenamiento.", color="verde")
                load_model_attempted = True
            except Exception as e:
                print_color(f"Error al cargar el modelo desde {BEST_MODEL_PATH} para '{fruit_type_arg}': {e}. Se procederá a entrenar.", color="rojo")
                BEST_MODEL_PATH = None # Asegurar que no se intente usar un modelo corrupto
        else:
            print_color(f"No se encontró un modelo pre-entrenado para '{fruit_type_arg}'. Se procederá a entrenar.", color="amarillo")
    
    if not load_model_attempted or not BEST_MODEL_PATH:
        if not args.use_best_model: # Si no se pidió usar el mejor modelo, o si falló la carga
             print_color(f"Entrenando nuevo modelo para '{fruit_type_arg}'...", color="magenta")
        
        # === BLOQUE DE ENTRENAMIENTO DEL MODELO ===

        # Calcular class weights
        classes = np.array(train_gen.classes) 
        
        class_weights_calculated = class_weight.compute_class_weight(
            class_weight='balanced', # Corregido el nombre del argumento
            classes=np.unique(classes), 
            y=classes
        )
        class_weights_dict = {}
        for nombre, peso_np in zip(NIVEL_DE_MADUREZ, class_weights_calculated):
            class_weights_dict[nombre] = float(peso_np)

        # Ahora imprimís el dict y vas a ver {'inmaduro': 0.77, 'maduro': 1.34, ...}
        print_color(f"Pesos de clase calculados para '{fruit_type_arg}': {class_weights_dict}", "cyan")

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_log_dir  = LOGS_DIR / current_time # Usar LOGS_DIR específico de la fruta
        run_log_dir.mkdir(parents=True, exist_ok=True)

        # callbacks solo si vamos a entrenar
        tb_cb = TensorBoard(log_dir=run_log_dir, histogram_freq=1)
        es_cb = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
        best_model_filepath = run_log_dir / 'best_model.keras'
        mc_cb = ModelCheckpoint(filepath=best_model_filepath, monitor='val_loss', save_best_only=True, verbose=1)
        lr_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5, verbose=1)
        callbacks_list = [tb_cb, es_cb, mc_cb, lr_cb]

        print_color(f"Iniciando entrenamiento del modelo para {fruit_type_arg}...", color="cyan")
        
        initial_lr = model.optimizer.learning_rate.numpy() if hasattr(model.optimizer.learning_rate, 'numpy') else model.optimizer.learning_rate # O directamente 1e-4 si es fijo
        
        start_time = time.time()
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=10, # Ajustar epochs según sea necesario (aumentado para dar más margen con EarlyStopping)
            callbacks=callbacks_list,
            class_weight=class_weights_dict # Aplicar pesos de clase
        )
        end_time = time.time()
        print_color(f"Entrenamiento completado para '{fruit_type_arg}'.", color="verde")

        # --- INICIO DE LOGEO DE ENTRENAMIENTO FORMATEADO ---
        print(f"\n=== Registro de Entrenamiento: {fruit_type_arg.upper()} ===")
        
        epochs_trained = len(history.history['loss'])
        total_training_time = end_time - start_time
        avg_time_per_epoch = total_training_time / epochs_trained if epochs_trained > 0 else 0
        batches_per_epoch = len(train_gen)

        print("\n1. Información General")
        print(f"   • Epochs entrenados: {epochs_trained}")
        print(f"   • Batches por epoch: {batches_per_epoch}")
        print(f"   • Duración promedio por epoch: ≈{avg_time_per_epoch:.0f} s (≈{avg_time_per_epoch/batches_per_epoch:.2f} s/step)")
        # Para mostrar el LR en notación científica como 1.00 x 10⁻⁴
        print(f"   • Learning rate inicial: {initial_lr:.2e}".replace("e-0", " × 10⁻").replace("e-", " × 10⁻"))


        print("\n2. Métricas por Epoch")
        current_best_val_loss = float('inf')
        for i in range(epochs_trained):
            print(f"\n   • Epoch {i+1}")
            train_acc = history.history['accuracy'][i] * 100
            train_loss = history.history['loss'][i]
            val_acc = history.history['val_accuracy'][i] * 100
            val_loss = history.history['val_loss'][i]

            print(f"     – Entrenamiento: accuracy = {train_acc:.2f} %, loss = {train_loss:.4f}")
            print(f"     – Validación:   accuracy = {val_acc:.2f} %, loss = {val_loss:.4f}")
            
            val_loss_msg = ""
            if val_loss < current_best_val_loss:
                val_loss_msg = f"val_loss mejoró de {current_best_val_loss if current_best_val_loss != float('inf') else 'inf'} → {val_loss:.5f} (modelo guardado)"
                current_best_val_loss = val_loss
            else:
                # Si no mejoró, no se imprime nada según el formato deseado, o se podría añadir un "no mejoró".
                # El formato del usuario solo muestra cuando mejora.
                pass 
            if val_loss_msg: # Solo imprimir si hay mensaje de mejora
                 print(f"     – {val_loss_msg}")


        print("\n3. Ruta del modelo guardado")
        final_best_model_path = best_model_filepath # Path que se usó en ModelCheckpoint
        
        # Chequear si el mejor modelo realmente existe en la ruta esperada
        # (puede que no si el entrenamiento fue muy corto y val_loss nunca mejoró)
        model_saved_path_to_display = "No se guardó un nuevo modelo (val_loss no mejoró o entrenamiento interrumpido)."
        if best_model_filepath.exists():
             # Si usamos restore_best_weights=True, el modelo en memoria es el mejor.
             # La ruta guardada por ModelCheckpoint es la relevante aquí.
            model_saved_path_to_display = str(best_model_filepath.resolve())


        print(f"   {model_saved_path_to_display}")

        print("\n=== Fin del Registro de Entrenamiento ===")
        # --- FIN DE LOGEO DE ENTRENAMIENTO ---

    # siempre inferir después
    # === INICIO DE INFERENCIA EN CONJUNTO DE TEST ===
    print_color(f"\nInferencia en test para '{fruit_type_arg}':", color="cyan")
    inferir_test(fruit_type_arg=fruit_type_arg) # Pasar fruit_type_arg

# Helper para imprimir con colores (sin cambios, pero asegurar que funcione en Git Bash)
def print_color(text, color="default"):

    colors = {
        "default": "\x1b[0m", # Reset
        "rojo": "\x1b[31m",   # Red
        "verde": "\x1b[32m", # Green
        "amarillo": "\x1b[33m", # Yellow
        "azul": "\x1b[34m",    # Blue
        "magenta": "\x1b[35m", # Magenta
        "cyan": "\x1b[36m",    # Cyan
    }

    end_color = colors["default"]
    start_color = colors.get(color.lower(), colors["default"]) # Usar default si el color no existe
    print(start_color + text + end_color)

# --- PUNTO DE ENTRADA DEL SCRIPT ---
if __name__ == '__main__':
    main()  