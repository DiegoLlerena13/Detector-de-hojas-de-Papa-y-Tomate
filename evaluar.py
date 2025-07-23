import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Cargar el modelo entrenado
modelo = tf.keras.models.load_model("modelo_planta3.h5")

# Nombres de las clases (en el mismo orden que el entrenamiento)
clases = [
    'Tizón tardío de papa',      # Potato___Late_blight
    'Papa saludable',            # Potato___healthy
    'Mancha foliar por Septoria en tomate',  # Tomato_Septoria_leaf_spot
    'Tomate saludable'           # Tomato_healthy
]

# Función de predicción
def predecir_enfermedad(img_pil):
    img = img_pil.convert("RGB")  # Asegura que no tenga canal alfa
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediccion = modelo.predict(img_array)
    clase_predicha = np.argmax(prediccion)
    return clases[clase_predicha]

# Interfaz de usuario
st.title("🧪 Detección de Enfermedades en Hojas de Papa y Tomate")
st.write("Sube una imagen de hoja para obtener un diagnóstico.")

uploaded_file = st.file_uploader("Selecciona una imagen", type=["jpg", "png", "jpeg"])

if uploaded_file:
    imagen = Image.open(uploaded_file)
    st.image(imagen, caption="Imagen cargada", use_container_width=True)  # ← corregido aquí

    with st.spinner("Analizando imagen..."):
        resultado = predecir_enfermedad(imagen)

    st.success(f"🩺 Enfermedad detectada: **{resultado}**")
