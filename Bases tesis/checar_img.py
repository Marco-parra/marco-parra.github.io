import os
from PIL import Image

# ======================
# Configuración
# ======================
INPUT_FOLDER = r"C:\Users\Lenovo\Downloads\dataset2\DDR_dataset\DDR-dataset\DR_grading\test"  # carpeta con tus imágenes originales
OUTPUT_FOLDER = r"C:\Users\Lenovo\Downloads\dataset2\DDR_dataset\DDR-dataset\DR_grading\test_png"  # carpeta para guardar en .png

# Crear carpeta de salida si no existe
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Contadores
total = 0
corruptas = 0
convertidas = 0

# ======================
# Procesar imágenes
# ======================
for fname in os.listdir(INPUT_FOLDER):
    if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
        total += 1
        path = os.path.join(INPUT_FOLDER, fname)
        try:
            with Image.open(path) as img:
                # Convertir a RGB para evitar problemas con modos raros (ej. CMYK, RGBA)
                img = img.convert("RGB")
                
                # Nombre nuevo en .png
                new_name = os.path.splitext(fname)[0] + ".png"
                save_path = os.path.join(OUTPUT_FOLDER, new_name)
                
                img.save(save_path, "PNG")
                convertidas += 1
        except Exception as e:
            print(f"❌ Imagen corrupta o ilegible: {fname} ({e})")
            corruptas += 1

print("\n===== RESULTADOS =====")
print(f"🔎 Total imágenes revisadas: {total}")
print(f"✅ Convertidas a .png: {convertidas}")
print(f"⚠️ Corruptas/ilegibles: {corruptas}")
print(f"📂 Imágenes guardadas en: {OUTPUT_FOLDER}")
