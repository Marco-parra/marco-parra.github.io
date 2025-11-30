import os
import matplotlib.pyplot as plt

# Ruta a tu dataset con carpetas 0,1,2,3,4
DATASET_PATH = r"C:\Users\Lenovo\Downloads\dia\dataset"

# Diccionario para guardar conteo
class_counts = {}

# Recorrer carpetas de clases
for class_name in sorted(os.listdir(DATASET_PATH)):
    class_folder = os.path.join(DATASET_PATH, class_name)
    if os.path.isdir(class_folder):
        count = len([f for f in os.listdir(class_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        class_counts[class_name] = count

# Mostrar conteo
print("📊 Distribución de imágenes por clase:")
for k,v in class_counts.items():
    print(f"Clase {k}: {v} imágenes")

# Graficar distribución
plt.figure(figsize=(8,5))
plt.bar(class_counts.keys(), class_counts.values(), color="skyblue")
plt.xlabel("Clases (0=No DR, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative)")
plt.ylabel("Número de imágenes")
plt.title("Distribución de imágenes por clase en el dataset")
plt.show()
