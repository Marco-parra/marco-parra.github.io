import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# ===============================
# 1. Configuración de rutas
# ===============================
BASE_PATH = r"C:\Users\Lenovo\Downloads\dia"
DATASET_PATH = os.path.join(BASE_PATH, "dataset")  # dataset\0,1,2,3,4
MODEL_OUTPUT = os.path.join(BASE_PATH, "modelo_resnet50_cpu_529.keras")

# ===============================
# 2. Definir modelo base
# ===============================
base_model = ResNet50(include_top=False, weights="imagenet", input_shape=(529,529,3))

# Congelar capas (fase 1: entrenar solo la cabeza)
for layer in base_model.layers:
    layer.trainable = False

# Nueva cabeza
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(128, activation="relu")(x)
output = layers.Dense(5, activation="softmax")(x)

model = models.Model(inputs=base_model.input, outputs=output)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
print("✅ Modelo compilado con entrada 529x529")

# ===============================
# 3. Generadores de datos
# ===============================
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(529,529),  # 🔹 ahora 529x529
    batch_size=4,           # ⚠️ reducimos batch por CPU y tamaño
    class_mode="categorical",
    subset="training"
)

val_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(529,529),
    batch_size=4,
    class_mode="categorical",
    subset="validation"
)

# ===============================
# 4. Entrenamiento con CPU
# ===============================
es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,  # máximo 20, se detendrá antes si no mejora
    callbacks=[es],
    verbose=1
)

# ===============================
# 5. Guardar modelo
# ===============================
model.save(MODEL_OUTPUT)
print(f"✅ Modelo guardado en {MODEL_OUTPUT}")

# ===============================
# 6. Gráfica de entrenamiento
# ===============================
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.xlabel("Épocas")
plt.ylabel("Precisión")
plt.legend()
plt.title("Curva de entrenamiento (529x529)")
plt.show()
