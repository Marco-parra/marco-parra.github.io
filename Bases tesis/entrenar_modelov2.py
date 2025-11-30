# ================================================
# 1. Rutas locales (ajusta a tu carpeta)
# ================================================
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import class_weight
import matplotlib.pyplot as plt

BASE_PATH = r"C:\Users\Lenovo\Downloads\dia"
DATASET_PATH = os.path.join(BASE_PATH, "dataset")   # dataset\0,1,2,3,4

# ================================================
# 2. Crear modelo desde cero (sin pesos)
# ================================================
# Arquitectura ResNet50 sin pesos preentrenados
base_model = ResNet50(include_top=False, weights=None, input_shape=(224,224,3))

# Añadir nueva cabeza para 5 clases
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)  # regularización para evitar sobreajuste
output = layers.Dense(5, activation="softmax")(x)

model = models.Model(inputs=base_model.input, outputs=output)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# ================================================
# 3. Generadores de datos con aumentación
# ================================================
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

val_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

# ================================================
# 4. Entrenamiento completo (todas las capas)
# ================================================
es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

# Calcular class weights por desbalance
labels_array = train_gen.classes
class_weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels_array),
    y=labels_array
)
class_weights = dict(enumerate(class_weights))
print("⚖️ Pesos por clase:", class_weights)

print("🚀 Entrenando modelo desde cero")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=50,   # puedes aumentar si quieres más exactitud
    class_weight=class_weights,
    callbacks=[es]
)

# ================================================
# 5. Evaluación final
# ================================================
val_loss, val_acc = model.evaluate(val_gen)
print(f"📊 Pérdida validación: {val_loss:.4f} | Precisión validación: {val_acc:.4f}")

# ================================================
# 6. Guardar modelo final
# ================================================
FINAL_MODEL_PATH = os.path.join(BASE_PATH, "modelo_resnet50_desde_cero.keras")
model.save(FINAL_MODEL_PATH)
print("✅ Modelo final guardado en:", FINAL_MODEL_PATH)

# ================================================
# 7. Graficar curvas de entrenamiento
# ================================================
def plot_history(history, title):
    plt.plot(history.history['accuracy'], label='Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Validación')
    plt.title(title)
    plt.xlabel("Épocas")
    plt.ylabel("Precisión")
    plt.legend()
    plt.show()

plot_history(history, "Entrenamiento desde cero")
