# ================================================
# 1. Librerías necesarias
# ================================================
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils import class_weight
import matplotlib.pyplot as plt

# ================================================
# 2. Rutas locales
# ================================================
BASE_PATH = r"C:\Users\Lenovo\Downloads\dia"
DATASET_PATH = os.path.join(BASE_PATH, "dataset")   # dataset\0,1,2,3,4
FINAL_MODEL_PATH = os.path.join(BASE_PATH, "modelo_resnet50_mejorado.keras")

# ================================================
# 3. Focal Loss
# ================================================
def focal_loss(gamma=2.0, alpha=0.25):
    def _loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1-1e-7)
        ce = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)
        fl = weight * ce
        return tf.reduce_sum(fl, axis=1)
    return _loss

# ================================================
# 4. Crear modelo base (desde cero)
# ================================================
IMG_SIZE = (299,299)

base_model = ResNet50(include_top=False, weights=None, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(5, activation="softmax")(x)

model = models.Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), 
              loss=focal_loss(gamma=2.0, alpha=0.5),
              metrics=["accuracy"])
model.summary()

# ================================================
# 5. Generadores de datos
# ================================================
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=25,
    width_shift_range=0.12,
    height_shift_range=0.12,
    shear_range=0.12,
    zoom_range=0.15,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

val_gen = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

# ================================================
# 6. Callbacks
# ================================================
es = EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True, verbose=1)
rlr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)
ckp = ModelCheckpoint(FINAL_MODEL_PATH, monitor="val_loss", save_best_only=True, verbose=1)

# ================================================
# 7. Entrenamiento (fase 1: solo cabeza)
# ================================================
for layer in base_model.layers:
    layer.trainable = False

print("🚀 Entrenando fase 1 (solo cabeza)")
history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=[es, rlr, ckp]
)

# ================================================
# 8. Fine-tuning (fase 2: últimas 100 capas)
# ================================================
for layer in base_model.layers[-100:]:
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), 
              loss=focal_loss(gamma=2.0, alpha=0.5),
              metrics=["accuracy"])

labels_array = train_gen.classes
class_weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels_array),
    y=labels_array
)
class_weights = dict(enumerate(class_weights))
print("⚖️ Pesos por clase:", class_weights)

print("🚀 Entrenando fase 2 (fine-tuning)")
history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=30,
    class_weight=class_weights,
    callbacks=[es, rlr, ckp]
)

# ================================================
# 9. Evaluación final
# ================================================
val_loss, val_acc = model.evaluate(val_gen)
print(f"📊 Pérdida validación: {val_loss:.4f} | Precisión validación: {val_acc:.4f}")

# ================================================
# 10. Graficar curvas de entrenamiento
# ================================================
def plot_history(history, title):
    plt.plot(history.history['accuracy'], label='Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Validación')
    plt.title(title)
    plt.xlabel("Épocas")
    plt.ylabel("Precisión")
    plt.legend()
    plt.show()

plot_history(history1, "Fase 1 - Solo cabeza")
plot_history(history2, "Fase 2 - Fine-tuning")
