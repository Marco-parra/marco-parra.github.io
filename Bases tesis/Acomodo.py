import pandas as pd
import os
import shutil

# ==============================
# 1. Rutas (ajústalas a tu PC)
# ==============================
BASE_PATH = r"C:\Users\Lenovo\Downloads\dataset2\aptos"
CSV_PATH = os.path.join(BASE_PATH, "train.csv")  # CSV con etiquetas
IMG_FOLDER = os.path.join(BASE_PATH, "train_images")       # Carpeta con imágenes
OUTPUT_FOLDER = os.path.join(BASE_PATH, "dataset2")     # Carpeta organizada

# ==============================
# 2. Leer CSV
# ==============================
labels = pd.read_csv(CSV_PATH)

# ==============================
# 3. Crear carpetas 0–4
# ==============================
for i in range(5):
    os.makedirs(os.path.join(OUTPUT_FOLDER, str(i)), exist_ok=True)

# ==============================
# 4. Mover/Copiar imágenes
# ==============================
not_found = []

for idx, row in labels.iterrows():
    # Quitar ceros a la izquierda del nombre del CSV
    img_name = row["id_code"].lstrip("0") + ".png"  
    label = str(row["diagnosis"])

    src = os.path.join(IMG_FOLDER, img_name)
    dst = os.path.join(OUTPUT_FOLDER, label, img_name)

    if os.path.exists(src):
        shutil.copy(src, dst)  # usa shutil.move si quieres mover en vez de copiar
    else:
        not_found.append(img_name)

print("✅ Dataset organizado en carpetas 0–4")
print(f"⚠️ Imágenes no encontradas: {len(not_found)}")
if not_found:
    print("Ejemplos de faltantes:", not_found[:10])
