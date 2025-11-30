import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, save_img

# ================================
# CONFIGURACIÓN
# ================================
BASE_PATH = r"C:\Users\Lenovo\Downloads\dia\dataset"  # Carpeta con subcarpetas 0-4
CLASSES = ["0", "1", "2", "3", "4"]                   # Nombres de las carpetas
TARGET_SIZE = (224, 224)                              # Tamaño de imágenes
AUG_PER_IMAGE = 5                                     # Cuántas imágenes nuevas generar por imagen

# ================================
# 1. Crear generador de augmentación
# ================================
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode="nearest"
)

# ================================
# 2. Contar imágenes por clase
# ================================
counts = {}
for c in CLASSES:
    path = os.path.join(BASE_PATH, c)
    counts[c] = len(os.listdir(path))

print("📊 Distribución inicial:", counts)

# Definir la clase mayoritaria como referencia
max_count = max(counts.values())

# ================================
# 3. Aumentar solo clases minoritarias
# ================================
for c in CLASSES:
    path = os.path.join(BASE_PATH, c)
    imgs = os.listdir(path)

    if len(imgs) < max_count:  # solo aumentar clases pequeñas
        needed = max_count - len(imgs)  # cuántas faltan
        print(f"🚀 Generando {needed} imágenes extra para clase {c}...")

        i = 0
        while i < needed:
            for img_name in imgs:
                img_path = os.path.join(path, img_name)
                img = load_img(img_path, target_size=TARGET_SIZE)
                x = img_to_array(img)
                x = np.expand_dims(x, axis=0)

                # Generar imágenes aumentadas
                for batch in datagen.flow(x, batch_size=1):
                    new_name = f"aug_{i}_{img_name}"
                    save_img(os.path.join(path, new_name), batch[0])
                    i += 1
                    if i >= needed:
                        break
            if i >= needed:
                break

print("✅ Dataset balanceado con augmentación")
