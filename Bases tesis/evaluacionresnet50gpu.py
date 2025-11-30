import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import preprocess_input
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

# ==========================
# CONFIGURACIÓN
# ==========================
MODEL_PATH = r"C:\Users\Lenovo\Downloads\dia\modelo_resnet50_gpu.keras"  # 🔹 Ajusta según tu ruta actual
DATASET_PATH = r"C:\Users\Lenovo\Downloads\dia\dataset_rgb"                   # 🔹 Ajusta según tu carpeta
OUTPUT_DIR = r"C:\Users\Lenovo\Downloads\dia\evaluacion_resnet50gpuRGB"
PDF_PATH = os.path.join(OUTPUT_DIR, "Reporte_Evaluacion_ResNet50gpuRGB.pdf")

os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = (529, 529)  # 🔹 CAMBIO CLAVE
BATCH_SIZE = 8         # 🔹 Tamaño de batch reducido (por RAM, ya que las imágenes son grandes)
CLASS_NAMES = ["No DR (0)", "Mild (1)", "Moderate (2)", "Severe (3)", "Proliferative (4)"]

# ==========================
# 1️⃣ Cargar modelo
# ==========================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"No se encontró el modelo en: {MODEL_PATH}")

model = load_model(MODEL_PATH)
print("✅ Modelo ResNet50 (529x529) cargado correctamente.")

# ==========================
# 2️⃣ Generar conjunto de validación
# ==========================
datagen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.2)

val_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,  # 🔹 Ajustado a 529x529
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

# ==========================
# 3️⃣ Predicciones y métricas
# ==========================
y_true = val_gen.classes
y_pred = model.predict(val_gen, verbose=1)
y_pred_classes = np.argmax(y_pred, axis=1)

report = classification_report(y_true, y_pred_classes, target_names=CLASS_NAMES, output_dict=True)
df_report = pd.DataFrame(report).transpose()
df_report.to_csv(os.path.join(OUTPUT_DIR, "reporte_metricas_resnet529.csv"), encoding="utf-8-sig")

# ==========================
# 4️⃣ Matriz de confusión
# ==========================
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión - ResNet50 (529x529)")
plt.tight_layout()
conf_path = os.path.join(OUTPUT_DIR, "matriz_confusion_resnet529.png")
plt.savefig(conf_path, dpi=300)
plt.close()

# ==========================
# 5️⃣ Curvas ROC por clase
# ==========================
y_true_bin = np.zeros_like(y_pred)
for i, c in enumerate(y_true):
    y_true_bin[i, c] = 1

plt.figure(figsize=(8,6))
for i in range(len(CLASS_NAMES)):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"{CLASS_NAMES[i]} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel("Tasa Falsos Positivos (FPR)")
plt.ylabel("Tasa Verdaderos Positivos (TPR)")
plt.title("Curvas ROC por Clase - ResNet50 (529x529)")
plt.legend(loc="lower right")
plt.tight_layout()
roc_path = os.path.join(OUTPUT_DIR, "curva_ROC_resnet529.png")
plt.savefig(roc_path, dpi=300)
plt.close()

# ==========================
# 6️⃣ Crear reporte PDF
# ==========================
doc = SimpleDocTemplate(PDF_PATH, pagesize=letter)
styles = getSampleStyleSheet()
elements = []

elements.append(Paragraph("Evaluación del Modelo ResNet50 (529x529)", styles["Title"]))
elements.append(Spacer(1, 12))

elements.append(Paragraph(
    "Este informe presenta los resultados de la evaluación del modelo ResNet50 aplicado al diagnóstico automático de retinopatía diabética. "
    "Se incluyen las métricas de desempeño, la matriz de confusión y las curvas ROC por clase. "
    "Los datos fueron procesados utilizando imágenes de tamaño 529x529 píxeles con un conjunto de validación del 20%.",
    styles["Normal"]
))
elements.append(Spacer(1, 12))

elements.append(Paragraph("Matriz de Confusión", styles["Heading2"]))
elements.append(RLImage(conf_path, width=400, height=300))
elements.append(Spacer(1, 12))

elements.append(Paragraph("Curvas ROC por Clase", styles["Heading2"]))
elements.append(RLImage(roc_path, width=400, height=300))
elements.append(Spacer(1, 12))

elements.append(Paragraph("Tabla de Métricas de Evaluación", styles["Heading2"]))
data = [df_report.columns.tolist()] + df_report.reset_index().values.tolist()
table = Table(data)
table.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#003366")),
    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
    ('BOTTOMPADDING', (0,0), (-1,0), 6),
    ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
]))
elements.append(table)
elements.append(Spacer(1, 24))
elements.append(Paragraph(f"Exactitud global: {report['accuracy']:.4f}", styles["Normal"]))

doc.build(elements)
print(f"✅ Reporte PDF generado correctamente en: {PDF_PATH}")
