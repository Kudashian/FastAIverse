import gradio as gr
from fastai.vision.all import *

# Load your exported learner
learn_inf = load_learner('waste_classifier.pkl')

# Prediction function
def classify_image(img):
    img = PILImage.create(img)
    pred, idx, probs = learn_inf.predict(img)
    return {learn_inf.dls.vocab[i]: float(probs[i]) for i in range(len(probs))}

# Interface
title = "♻️ Waste Material Classifier"
description = "Upload a photo of a waste item (e.g. bottle, paper, food scrap) and this model will predict whether it's **biodegradable** or **non-biodegradable**."

# Launch
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# {title}")
    gr.Markdown(description)

    with gr.Row():
        with gr.Column():
            image = gr.Image(type="pil", label="Upload Image", shape=(224, 224))
            button = gr.Button("Classify")
        with gr.Column():
            label = gr.Label(label="Prediction")
    
    button.click(fn=classify_image, inputs=image, outputs=label)

demo.launch()