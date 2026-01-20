import gradio as gr
import base64
import requests
import io
from PIL import Image
import numpy as np

VERIFY_URL = "http://localhost:9696/verify"


def image_to_base64(img_np):
    if img_np is None:
        raise ValueError("Received None image")

    if img_np.dtype != np.uint8:
        img_np = (img_np * 255).clip(0, 255).astype("uint8")

    img = Image.fromarray(img_np)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def verify_faces(img1, img2):
    if img1 is None or img2 is None:
        return "### ❌ Please upload two images"

    payload = {
        "image1": image_to_base64(img1),
        "image2": image_to_base64(img2),
    }

    r = requests.post(VERIFY_URL, json=payload)
    r.raise_for_status()
    res = r.json()

    if res["same_person"]:
        return (
            "<div style='text-align:center; "
            "font-size:48px; font-weight:bold; color:green;'>"
            "SAME PERSON"
            "</div>"
        )
    else:
        return (
            "<div style='text-align:center; "
            "font-size:48px; font-weight:bold; color:red;'>"
            "DIFFERENT PERSONS"
            "</div>"
        )


with gr.Blocks() as demo:
    gr.Markdown("# Face Verification Demo")

    with gr.Row():
        img1 = gr.Image(type="numpy", label="Image 1")
        img2 = gr.Image(type="numpy", label="Image 2")

    btn = gr.Button("Verify")

    # IMPORTANT: Markdown output, not Textbox
    output = gr.Markdown()

    btn.click(
        fn=verify_faces,
        inputs=[img1, img2],
        outputs=output
    )

demo.launch(share=True)
