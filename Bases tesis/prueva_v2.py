import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import preprocess_input

# ==========================
# Configuración
# ==========================
MODEL_PATH = r"C:\Users\Lenovo\Downloads\dia\efficientnetv2b3_512_desktop.keras"
DATASET_PATH = r"C:\Users\Lenovo\Downloads\dia\dataset_rgb"
IMAGES_FOLDER = r"C:\Users\Lenovo\Downloads\dia\pruebav2"
CSV_OUTPUT_PRED = r"C:\Users\Lenovo\Downloads\dia\pruebav2\predicciones_eff.csv"
CSV_OUTPUT_REPORT = r"C:\Users\Lenovo\Downloads\dia\pruebav2\reporte_metricas_eff.csv"

# Nombres de clases
class_names = ["No DR (0)", "Mild (1)", "Moderate (2)", "Severe (3)", "Proliferative (4)"]

# ==========================
# 1. Cargar modelo
# ==========================
model = load_model(MODEL_PATH)
print("✅ Modelo cargado")

# ==========================
# 2. Evaluación en dataset (validación)
# ==========================
datagen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.2)

val_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

y_true = val_gen.classes
y_pred = model.predict(val_gen, verbose=1)
y_pred_classes = np.argmax(y_pred, axis=1)

# --- Matriz de confusión
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=val_gen.class_indices.keys(),
            yticklabels=val_gen.class_indices.keys())
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión")
plt.show()

# --- Métricas detalladas
report_dict = classification_report(
    y_true, y_pred_classes,
    target_names=val_gen.class_indices.keys(),
    output_dict=True
)

# Guardar reporte en CSV
df_report = pd.DataFrame(report_dict).transpose()
df_report.to_csv(CSV_OUTPUT_REPORT, index=True, encoding="utf-8-sig")
print(f"✅ Reporte de métricas guardado en {CSV_OUTPUT_REPORT}")

# ==========================
# 3. Predicción en imágenes nuevas
# ==========================
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    pred = model.predict(img_array, verbose=0)
    class_idx = np.argmax(pred)
    confidence = np.max(pred)

    return class_names[class_idx], confidence

results = []
for fname in os.listdir(IMAGES_FOLDER):
    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
        path = os.path.join(IMAGES_FOLDER, fname)
        label, conf = predict_image(path)
        results.append([fname, label, round(conf,4)])

# ==========================
# 4. Guardar predicciones en CSV
# ==========================
df_pred = pd.DataFrame(results, columns=["Imagen", "Prediccion", "Confianza"])
df_pred.to_csv(CSV_OUTPUT_PRED, index=False, encoding="utf-8-sig")
print(f"✅ Predicciones guardadas en {CSV_OUTPUT_PRED}")

