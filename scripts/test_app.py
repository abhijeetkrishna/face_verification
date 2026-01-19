import base64
import requests
from PIL import Image
import io

# -----------------------
# Paths to images
# -----------------------
img_path1 = "data/raw/lfw_home/lfw_funneled/Abdoulaye_Wade/Abdoulaye_Wade_0001.jpg"
img_path2 = "data/raw/lfw_home/lfw_funneled/Abdoulaye_Wade/Abdoulaye_Wade_0002.jpg"

VERIFY_URL = "http://localhost:9696/verify"

# -----------------------
# Helper: image file -> base64
# -----------------------
def image_path_to_base64(path):
    img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")  # encode as real image
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# -----------------------
# Read & convert images
# -----------------------
img1_b64 = image_path_to_base64(img_path1)
img2_b64 = image_path_to_base64(img_path2)

# -----------------------
# Call verification API
# -----------------------
payload = {
    "image1": img1_b64,
    "image2": img2_b64
}

response = requests.post(VERIFY_URL, json=payload)

# -----------------------
# Print result
# -----------------------
print("Status code:", response.status_code)
print("Response JSON:", response.json())
