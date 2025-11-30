import os
import pandas as pd
import shutil
from tqdm import tqdm

# === CONFIGURACIÓN ===
csv_path = r"C:\Users\Lenovo\Downloads\DeepDRiD-master\DeepDRiD-master\regular_fundus_images\regular-fundus-training\regular-fundus-training.csv"
images_root = r"C:\Users\Lenovo\Downloads\DeepDRiD-master\DeepDRiD-master\regular_fundus_images\regular-fundus-training\Images"
output_dir = r"C:\Users\Lenovo\Downloads\DeepDRiD-master\DeepDRiD-master"

# Crear carpeta destino si no existe
os.makedirs(output_dir, exist_ok=True)

# Leer CSV original
df = pd.read_csv(csv_path)

# Crear lista para nuevo CSV
new_data = []

print("📂 Organizando imágenes...")
for idx, row in tqdm(df.iterrows(), total=len(df)):
    patient_id = str(row["patient_id"])
    image_id = str(row["image_id"])

    # Buscar imagen (puede estar en subcarpeta del paciente)
    possible_ext = [".jpg", ".jpeg", ".png"]
    found_path = None

    for ext in possible_ext:
        candidate = os.path.join(images_root, f"{patient_id}", f"{image_id}{ext}")
        if os.path.exists(candidate):
            found_path = candidate
            break

    # Si no la encuentra, prueba directamente sin subcarpeta (por si acaso)
    if not found_path:
        for ext in possible_ext:
            candidate = os.path.join(images_root, f"{image_id}{ext}")
            if os.path.exists(candidate):
                found_path = candidate
                break

    if not found_path:
        print(f"⚠️ No se encontró imagen para {image_id}")
        continue

    # Crear nuevo nombre único
    ext = os.path.splitext(found_path)[1]
    new_name = f"{patient_id}_{image_id}_{idx}{ext}"
    new_path = os.path.join(output_dir, new_name)

    # Copiar imagen
    shutil.copy(found_path, new_path)

    # Etiqueta DR (elige izquierda o derecha según corresponda)
    if "_l" in image_id.lower():
        label = row["left_eye_DR"]
    elif "_r" in image_id.lower():
        label = row["right_eye_DR"]
    else:
        label = row["patient_DR_L"]

    # Guardar datos para nuevo CSV
    new_data.append({
        "new_filename": new_name,
        "label": label,
        "original_patient": patient_id,
        "original_image_id": image_id
    })

# Crear nuevo DataFrame
new_df = pd.DataFrame(new_data)

# Guardar nuevo CSV
new_csv_path = os.path.join(output_dir, "dataset_organizado.csv")
new_df.to_csv(new_csv_path, index=False, encoding="utf-8-sig")

print("\n✅ Proceso completado.")
print(f"Imágenes organizadas en: {output_dir}")
print(f"Nuevo CSV guardado en: {new_csv_path}")
