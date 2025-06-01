import os
import numpy as np
from PIL import Image
import datetime
import argparse
from sklearn.utils import class_weight
from pathlib import Path
import time

# Verbosidad de TensorFlow (0 = todos, 1 = filtrar INFO, 2 = filtrar INFO y WARNING, 3 = filtrar INFO, WARNING, y ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT_DEFAULT = (PROJECT_ROOT / ".." / "dataset").resolve()
LOGS_ROOT_DEFAULT = (PROJECT_ROOT / ".." / "logs").resolve()

DATASET_DIR: Path = None
LOGS_DIR: Path = None
train_gen = None
model = None

NIVEL_DE_MADUREZ = [
    'inmaduro',
    'maduro',
    'sobre-maduro',
    'descomposicion'
]


def crear_generadores(
    data_dir_param=None,
    img_size=(200, 200),
    batch_size=32,
    val_split=0.2
):
    actual_data_dir = data_dir_param if data_dir_param else DATASET_DIR / 'train'
    if not actual_data_dir.exists():
        print_color(f"Directorio de datos de entrenamiento no encontrado: {actual_data_dir}", color="rojo")
        print_color("Asegúrate de que la estructura es: dataset/<tipo_fruta>/train/<clase>/imagenes...", color="amarillo")
        return None, None

    # --- Sin data augmentation, solo rescale + valid split ---
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=val_split
    )

    try:
        train_generator = datagen.flow_from_directory(
            actual_data_dir,
            target_size=img_size,
            batch_size=batch_size,
            classes=NIVEL_DE_MADUREZ,
            class_mode='categorical',
            subset='training'
        )

        val_generator = datagen.flow_from_directory(
            actual_data_dir,
            target_size=img_size,
            batch_size=batch_size,
            classes=NIVEL_DE_MADUREZ,
            class_mode='categorical',
            subset='validation'
        )
        return train_generator, val_generator

    except Exception as e:
        print_color(f"Ocurrió un error inesperado al crear generadores: {e}", color="rojo")
        return None, None


def crear_modelo(input_shape=(200,200,3), n_classes=len(NIVEL_DE_MADUREZ)):
    inp = Input(shape=input_shape)
    x = Flatten()(inp)
    x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    out = Dense(n_classes, activation='softmax')(x)
    model_mlp = Model(inputs=inp, outputs=out)

    model_mlp.compile(
        optimizer=Adam(1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model_mlp


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

    unrecognized_dirs_warnings = {}
    errores_procesamiento_general = 0
    total_imagenes_clases_conocidas = 0
    correctas_global_clases_conocidas = 0

    y_true = []
    y_pred = []

    for clase_dir in actual_test_dir.iterdir():
        if not clase_dir.is_dir():
            continue

        etiqueta_real_clase_original = clase_dir.name
        clase_para_stats = 'Desconocida'

        if etiqueta_real_clase_original in NIVEL_DE_MADUREZ:
            clase_para_stats = etiqueta_real_clase_original
        else:
            if etiqueta_real_clase_original not in unrecognized_dirs_warnings:
                unrecognized_dirs_warnings[etiqueta_real_clase_original] = 0

        for img_file_path in clase_dir.iterdir():
            if not (img_file_path.is_file() and img_file_path.suffix.lower() in ('.png', '.jpg', '.jpeg')):
                continue

            stats_por_clase_real[clase_para_stats]['total'] += 1
            if clase_para_stats != 'Desconocida':
                total_imagenes_clases_conocidas += 1
            else:
                unrecognized_dirs_warnings[etiqueta_real_clase_original] += 1

            try:
                img = Image.open(img_file_path).convert('RGB')
                img_resized = img.resize(img_size)
                arr = np.expand_dims(np.array(img_resized) / 255.0, 0)
                pred_vector = model.predict(arr, verbose=0)[0]
                pred_idx = np.argmax(pred_vector)
                nivel_madurez_pred = label_map[pred_idx]
                y_true.append(NIVEL_DE_MADUREZ.index(etiqueta_real_clase_original))
                y_pred.append(NIVEL_DE_MADUREZ.index(nivel_madurez_pred))

                stats_por_clase_real[clase_para_stats]['predicciones'][nivel_madurez_pred] += 1

                if clase_para_stats != 'Desconocida' and nivel_madurez_pred == clase_para_stats:
                    stats_por_clase_real[clase_para_stats]['correctas'] += 1
                    correctas_global_clases_conocidas += 1

            except Exception:
                errores_procesamiento_general += 1

    if unrecognized_dirs_warnings:
        for dir_name, count in unrecognized_dirs_warnings.items():
            if count > 0:
                print_color(f"[WARNING] Directorio '{dir_name}' no reconocido → agrupado como 'Desconocida' ({count} imágenes).", "amarillo")
        print("")

    print(f"=== Resultados de Inferencia: {fruit_type_arg.upper()} ===\n")

    print("1. Resumen General")
    if total_imagenes_clases_conocidas > 0:
        print(f"   • Clases conocidas: {total_imagenes_clases_conocidas} imágenes")
        acc_global = (correctas_global_clases_conocidas / total_imagenes_clases_conocidas * 100)
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

    print("2. Desglose por Clase")
    for nivel_real_key in NIVEL_DE_MADUREZ:
        stats = stats_por_clase_real[nivel_real_key]
        if stats['total'] == 0:
            continue

        nombre_clase_display = nivel_real_key.replace("descomposicion", "descomposición")
        print(f"\n   • {nombre_clase_display}")
        num_predichas_clase = sum(stats['predicciones'].values())
        acc_clase = (stats['correctas'] / num_predichas_clase * 100) if num_predichas_clase > 0 else 0
        print(f"     – Total: {stats['total']} | Correctas: {stats['correctas']} ({acc_clase:.2f}%)")

        pred_str_parts = []
        for nivel_pred, count in sorted(stats['predicciones'].items()):
            if count > 0:
                nivel_pred_display = nivel_pred.replace("descomposicion", "descomposición")
                pred_str_parts.append(f"{nivel_pred_display}={count}")
        if pred_str_parts:
            print(f"     – Predicciones: {', '.join(pred_str_parts)}")
    print("")

    if stats_por_clase_real['Desconocida']['total'] > 0:
        print("3. Predicciones para 'Desconocida'")
        print(f"   • Total: {stats_por_clase_real['Desconocida']['total']}")
        pred_desconocida_detalle_parts = []
        for pred_label, count in sorted(stats_por_clase_real['Desconocida']['predicciones'].items()):
            if count > 0:
                pred_label_display = pred_label.replace("descomposicion", "descomposición")
                pred_desconocida_detalle_parts.append(f"{pred_label_display}={count}")
        if pred_desconocida_detalle_parts:
            print(f"   • Predicciones: {', '.join(pred_desconocida_detalle_parts)}")
        print("")

    if errores_procesamiento_general > 0:
        print_color(f"[INFO] Se produjeron {errores_procesamiento_general} errores durante el procesamiento de algunas imágenes.", "amarillo")
        print("")

    print("=== Fin de la inferencia ===")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(NIVEL_DE_MADUREZ))))
    df_cm = pd.DataFrame(cm, index=NIVEL_DE_MADUREZ, columns=NIVEL_DE_MADUREZ)

    print(f"\n=== Matriz de Confusión: {fruit_type_arg.upper()} ===\n")
    print("3. Matriz de Confusión\n")

    col_names_display = [c.replace('descomposicion', 'descomposición') for c in NIVEL_DE_MADUREZ]
    real_col_width = 14
    pred_col_width = 14
    total_col_width = 7
    aciertos_col_width = 18

    header_line = f"{'Real ↓':<{real_col_width}} {'Predichas →':<12}"
    for col_name in col_names_display:
        header_line += f" {col_name:>{pred_col_width}}"
    header_line += f" | {'Total':>{total_col_width}} | {'Aciertos (%)':>{aciertos_col_width}} | Errores (desglose)"
    print(header_line)
    print("─" * len(header_line))

    for i, real_class_key in enumerate(NIVEL_DE_MADUREZ):
        real_class_display = real_class_key.replace('descomposicion', 'descomposición')
        row_str = f"{real_class_display:<{real_col_width}} {'':<12}"
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

        row_str += f" | {total_real:>{total_col_width}} | {aciertos_str:>{aciertos_col_width}} | {errores_str}"
        print(row_str)

    print(f"\n=== Fin de la Matriz de Confusión ===\n")


def main():
    global DATASET_DIR, LOGS_DIR, train_gen, model

    parser = argparse.ArgumentParser(description='Entrenar y evaluar una CNN (ahora MLP) para clasificación de madurez de frutas.')
    parser.add_argument('--fruit_type', type=str, default='bananas', choices=['bananas', 'tomates'],
                        help='Tipo de fruta para entrenar/evaluar (ej: bananas, tomates). Default: bananas')
    parser.add_argument('--use_best_model', action='store_true',
                        help='Si se especifica, intenta cargar el mejor modelo guardado previamente para la fruta. Si no, reentrena por defecto.')
    args = parser.parse_args()

    fruit_type_arg = args.fruit_type
    print_color(f"Procesando para el tipo de fruta: {fruit_type_arg}", color="magenta")

    DATASET_DIR = DATASET_ROOT_DEFAULT / fruit_type_arg
    LOGS_DIR = LOGS_ROOT_DEFAULT / fruit_type_arg
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    train_gen, val_gen = crear_generadores()
    if train_gen is None or val_gen is None:
        print_color(f"No se pudieron crear los generadores de datos para {fruit_type_arg}. Abortando.", color="rojo")
        return

    model = crear_modelo()
    model.summary()

    BEST_MODEL_PATH = None
    load_model_attempted = False

    if args.use_best_model:
        print_color(f"Intentando cargar el mejor modelo pre-entrenado para '{fruit_type_arg}'...", color="amarillo")
        if LOGS_DIR.exists():
            all_runs = sorted([d for d in LOGS_DIR.iterdir() if d.is_dir()], reverse=True)
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
                BEST_MODEL_PATH = None
        else:
            print_color(f"No se encontró un modelo pre-entrenado para '{fruit_type_arg}'. Se procederá a entrenar.", color="amarillo")

    if not load_model_attempted or not BEST_MODEL_PATH:
        if not args.use_best_model:
            print_color(f"Entrenando nuevo modelo (MLP) para '{fruit_type_arg}'...", color="magenta")

        classes = np.array(train_gen.classes)
        class_weights_calculated = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(classes),
            y=classes
        )
        
        
        class_weights_dict = {}
        for nombre, peso_np in zip(NIVEL_DE_MADUREZ, class_weights_calculated):
            class_weights_dict[nombre] = float(peso_np)

        # Ahora imprimís el dict y vas a ver {'inmaduro': 0.77, 'maduro': 1.34, ...}
        print_color(f"Pesos de clase calculados para '{fruit_type_arg}': {class_weights_dict}", "cyan")

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_log_dir = LOGS_DIR / current_time
        run_log_dir.mkdir(parents=True, exist_ok=True)

        tb_cb = TensorBoard(log_dir=run_log_dir, histogram_freq=1)
        es_cb = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
        best_model_filepath = run_log_dir / 'best_model.keras'
        mc_cb = ModelCheckpoint(filepath=best_model_filepath, monitor='val_loss', save_best_only=True, verbose=1)
        lr_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5, verbose=1)
        callbacks_list = [tb_cb, es_cb, mc_cb, lr_cb]

        print_color(f"Iniciando entrenamiento del modelo para {fruit_type_arg}...", color="cyan")

        initial_lr = model.optimizer.learning_rate.numpy() if hasattr(model.optimizer.learning_rate, 'numpy') else model.optimizer.learning_rate

        start_time = time.time()
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=10,
            callbacks=callbacks_list,
            class_weight=class_weights_dict
        )
        end_time = time.time()
        print_color(f"Entrenamiento completado para '{fruit_type_arg}'.", color="verde")

        print(f"\n=== Registro de Entrenamiento: {fruit_type_arg.upper()} ===")
        epochs_trained = len(history.history['loss'])
        total_training_time = end_time - start_time
        avg_time_per_epoch = total_training_time / epochs_trained if epochs_trained > 0 else 0
        batches_per_epoch = len(train_gen)

        print("\n1. Información General")
        print(f"   • Epochs entrenados: {epochs_trained}")
        print(f"   • Batches por epoch: {batches_per_epoch}")
        print(f"   • Duración promedio por epoch: ≈{avg_time_per_epoch:.0f} s (≈{avg_time_per_epoch/batches_per_epoch:.2f} s/step)")
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
            if val_loss_msg:
                print(f"     – {val_loss_msg}")

        print("\n3. Ruta del modelo guardado")
        model_saved_path_to_display = "No se guardó un nuevo modelo (val_loss no mejoró o entrenamiento interrumpido)."
        if best_model_filepath.exists():
            model_saved_path_to_display = str(best_model_filepath.resolve())
        print(f"   {model_saved_path_to_display}")
        print("\n=== Fin del Registro de Entrenamiento ===")

    print_color(f"\nInferencia en test para '{fruit_type_arg}':", color="cyan")
    inferir_test(fruit_type_arg=fruit_type_arg)


def print_color(text, color="default"):
    colors = {
        "default": "\x1b[0m",
        "rojo": "\x1b[31m",
        "verde": "\x1b[32m",
        "amarillo": "\x1b[33m",
        "azul": "\x1b[34m",
        "magenta": "\x1b[35m",
        "cyan": "\x1b[36m",
    }
    end_color = colors["default"]
    start_color = colors.get(color.lower(), colors["default"])
    print(start_color + text + end_color)


if __name__ == '__main__':
    main()
