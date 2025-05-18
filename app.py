import gradio as gr
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('final_model.h5')

# Prediction function
def predict_crack(img):
    img = img.resize((150, 150))  # Resize image to match model input
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize
    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        return "🟥 Crack Detected"
    else:
        return "🟩 No Crack Detected"

# Gradio interface
interface = gr.Interface(
    fn=predict_crack,
    inputs=gr.Image(type="pil"),
    outputs=gr.Text(label="Prediction"),
    title="Concrete Crack Detection",
    description="Upload an image of a concrete surface to detect cracks."
)

# Launch the app
interface.launch()
