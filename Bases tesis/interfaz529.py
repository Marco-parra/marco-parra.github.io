import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input
from tkinter import Tk, Label, Button, filedialog, Canvas
from PIL import Image, ImageTk

# ===============================
# CONFIGURACIÓN
# ===============================
MODEL_PATH = r"C:\Users\Lenovo\Downloads\dia\modelo_resnet50_gpu.keras"
IMG_SIZE = (529, 529) 
CLASS_NAMES = ["No DR (0)", "Mild (1)", "Moderate (2)", "Severe (3)", "Proliferative (4)"]

# Cargar modelo
model = load_model(MODEL_PATH)
print("✅ Modelo ResNet50 cargado correctamente.")

# ===============================
# FUNCIONES
# ===============================
def cargar_imagen():
    """Permite seleccionar una imagen y mostrarla."""
    global img_path, tk_img
    img_path = filedialog.askopenfilename(
        title="Selecciona una imagen",
        filetypes=[("Imagenes", "*.png;*.jpg;*.jpeg")]
    )
    if not img_path:
        return
    
    # Mostrar imagen en el canvas
    img = Image.open(img_path).resize((224, 224))  # 🔹 Canvas más pequeño solo para mostrar
    tk_img = ImageTk.PhotoImage(img)
    canvas.create_image(112, 112, image=tk_img)
    lbl_resultado.config(text="Imagen cargada, lista para predecir ✅")
    lbl_confianza.config(text="")
    root.update()

def predecir():
    """Procesa la imagen seleccionada y realiza la predicción."""
    if not img_path:
        lbl_resultado.config(text="⚠️ No se ha cargado ninguna imagen.")
        return

    # Cargar y preprocesar la imagen al tamaño correcto del modelo
    img = image.load_img(img_path, target_size=IMG_SIZE)  # 🔹 Cambiado a 529x529
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predicción
    pred = model.predict(img_array)
    class_idx = np.argmax(pred)
    confidence = np.max(pred)

    lbl_resultado.config(
        text=f"Diagnóstico: {CLASS_NAMES[class_idx]}",
        fg="#007acc",
        font=("Arial", 13, "bold")
    )
    lbl_confianza.config(
        text=f"Confianza: {confidence*100:.2f}%",
        fg="#444",
        font=("Arial", 11)
    )

# ===============================
# INTERFAZ GRÁFICA
# ===============================
root = Tk()
root.title("Diagnóstico de Retinopatía Diabética - ResNet50")
root.geometry("420x480")
root.resizable(False, False)
root.configure(bg="#f7f9fc")

Label(root, text="Diagnóstico Automático (ResNet50)", bg="#f7f9fc",
      fg="#003366", font=("Arial", 14, "bold")).pack(pady=10)

canvas = Canvas(root, width=224, height=224, bg="#e6e6e6", highlightthickness=1)
canvas.pack(pady=10)

btn_cargar = Button(root, text="📂 Cargar imagen", command=cargar_imagen,
                    bg="#007acc", fg="white", font=("Arial", 11, "bold"), width=18)
btn_cargar.pack(pady=5)

btn_predecir = Button(root, text="🔍 Predecir diagnóstico", command=predecir,
                      bg="#009933", fg="white", font=("Arial", 11, "bold"), width=18)
btn_predecir.pack(pady=5)

lbl_resultado = Label(root, text="", bg="#f7f9fc", fg="#333", font=("Arial", 11))
lbl_resultado.pack(pady=10)

lbl_confianza = Label(root, text="", bg="#f7f9fc", fg="#333", font=("Arial", 10))
lbl_confianza.pack()

Label(root, text="Desarrollado por Marco", bg="#f7f9fc",
      fg="#666", font=("Arial", 8, "italic")).pack(side="bottom", pady=5)

root.mainloop()
