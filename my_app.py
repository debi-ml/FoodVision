import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load your trained models
model1 = tf.keras.models.load_model('model/FoodVisionFineTuneAug/')
model2 = tf.keras.models.load_model('model/FoodVisionFineTune/')

with open('classes.txt', 'r') as f:
    classes = [line.strip() for line in f]

# Add information about the models
model1_info = """
### Model 1 Information

This model is based on the EfficientNetB0 architecture and was trained on the Food101 dataset.
"""

model2_info = """
### Model 2 Information

This model is based on the EfficientNetB0 architecture and was trained on augmented data, providing improved generalization.
"""

def preprocess(image: Image.Image):
    # Convert numpy array to PIL Image
    image = Image.fromarray((image * 255).astype(np.uint8))
    image = image.resize((224, 224))  # replace with the input size of your models
    image = np.array(image)
    # image = image / 255.0  # normalize if you've done so while training
    image = np.expand_dims(image, axis=0)
    return image

def predict(model_selection, image: Image.Image):
    # Choose the model based on the dropdown selection
    model = model1 if model_selection == "EfficentNetB0 Fine Tune" else model2

    image = preprocess(image)
    prediction = model.predict(image)
    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction)
    return predicted_class, confidence

iface = gr.Interface(
    fn=predict,
    inputs=[gr.Dropdown(["EfficentNetB0 Fine Tune", "EfficentNetB0 Fine Tune Augmented"]), gr.Image()],
    outputs=[gr.Textbox(label="Predicted Class"), gr.Textbox(label="Confidence")],
    title="Transfer Learning Mini Project",
    description=f"{model1_info}\n\n{model2_info}",
)

iface.launch()
