# ================================================
# Reentrenamiento desde cero con dataset balanceado
# ================================================
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils import class_weight
import matplotlib.pyplot as plt

# ============================
# 1. Configuración de rutas
# ============================
BASE_PATH = r"C:\Users\Lenovo\Downloads\dia"
DATASET_PATH = os.path.join(BASE_PATH, "dataset")   # dataset balanceado
FINAL_MODEL_PATH = os.path.join(BASE_PATH, "modelo_resnet50_mejorado.keras")

# ============================
# 2. Definir modelo desde cero
# ============================
base_model = ResNet50(include_top=False, weights="imagenet", input_shape=(224,224,3))
base_model.trainable = True   # entrenar TODA la red (mejor exactitud)

# Nueva cabeza
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)  # dropout ayuda contra overfitting
output = layers.Dense(5, activation="softmax")(x)

model = models.Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss="categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

# ============================
# 3. Generadores de datos
# ============================
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
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

# ============================
# 4. Calcular class weights
# ============================
labels_array = train_gen.classes
class_weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels_array),
    y=labels_array
)
class_weights = dict(enumerate(class_weights))
print("⚖️ Pesos por clase:", class_weights)

# ============================
# 5. Callbacks útiles
# ============================
es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
checkpoint = ModelCheckpoint(FINAL_MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1)
lr_reducer = ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, verbose=1)

# ============================
# 6. Entrenamiento completo
# ============================
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=50,
    class_weight=class_weights,
    callbacks=[es, checkpoint, lr_reducer]
)

# ============================
# 7. Evaluación final
# ============================
val_loss, val_acc = model.evaluate(val_gen)
print(f"📊 Pérdida validación: {val_loss:.4f} | Precisión validación: {val_acc:.4f}")

# ============================
# 8. Graficar curvas
# ============================
plt.figure(figsize=(8,4))
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title("Precisión durante el entrenamiento")
plt.xlabel("Épocas")
plt.ylabel("Precisión")
plt.legend()
plt.show()
