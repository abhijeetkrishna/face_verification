"""import gradio as gr

with gr.Blocks() as demo:
    gr.Markdown("# Hello World")
    with gr.Row():
        inp = gr.Textbox(label="Input")
        out = gr.Textbox(label="Output")
    btn = gr.Button("Submit")
    btn.click(lambda x: x, inp, out)

demo.launch(share=True)
"""

import gradio as gr
import base64
import requests
import io
from PIL import Image
import numpy as np

VERIFY_URL = "http://localhost:9696/verify"

def image_to_base64(img_np):
    # Defensive checks
    if img_np is None:
        raise ValueError("Received None image")

    # Ensure uint8 [0,255]
    if img_np.dtype != np.uint8:
        img_np = (img_np * 255).clip(0, 255).astype("uint8")

    img = Image.fromarray(img_np)

    buf = io.BytesIO()
    img.save(buf, format="PNG")  # IMPORTANT
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def verify_faces(img1, img2):
    if img1 is None or img2 is None:
        return "Please upload two images"

    payload = {
        "image1": image_to_base64(img1),
        "image2": image_to_base64(img2),
    }

    r = requests.post(VERIFY_URL, json=payload)

    # TEMPORARY: print response for debugging
    print("Status:", r.status_code)
    print("Response:", r.text)

    r.raise_for_status()
    res = r.json()

    return (
        f"Same person: {res['same_person']}\n"
        f"Score: {res['score']:.3f}\n"
        f"Threshold: {res['threshold']}"
    )


with gr.Blocks() as demo:
    gr.Markdown("# Face Verification Demo")

    with gr.Row():
        img1 = gr.Image(type="numpy", label="Image 1")
        img2 = gr.Image(type="numpy", label="Image 2")

    btn = gr.Button("Verify")
    output = gr.Textbox(label="Result")

    btn.click(
        fn=verify_faces,
        inputs=[img1, img2],
        outputs=output
    )

demo.launch(share=True)