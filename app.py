import base64
import io
from pathlib import Path

import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps, ImageStat
from torchvision import transforms

from vae_model import ConvVAE

MODEL_PATH = Path(__file__).resolve().parent / "vae_mnist.pt"

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing {MODEL_PATH.name}. Train first with: python vae_mnist.py"
        )
    model = ConvVAE().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model


model = load_model()
to_tensor = transforms.ToTensor()


def to_mnist(image: Image.Image) -> Image.Image:
    """Convert any image size to the 28x28 white-on-black MNIST layout."""
    gray = image.convert("L")
    if ImageStat.Stat(gray).mean[0] > 127:
        gray = ImageOps.invert(gray)

    if gray.size != (28, 28):
        bbox = gray.point(lambda p: 255 if p > 20 else 0).getbbox()
        if bbox:
            gray = gray.crop(bbox)
        gray.thumbnail((20, 20), Image.Resampling.LANCZOS)
        canvas = Image.new("L", (28, 28), 0)
        canvas.paste(gray, ((28 - gray.width) // 2, (28 - gray.height) // 2))
        return canvas
    return gray


def png_data_url(image: Image.Image, size=(224, 224)) -> str:
    large = image.resize(size, Image.Resampling.NEAREST)
    buf = io.BytesIO()
    large.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/reconstruct", methods=["POST"])
def reconstruct():
    uploaded = request.files.get("image")
    if uploaded is None or uploaded.filename == "":
        return jsonify({"error": "Please choose an image to upload."}), 400

    try:
        original = Image.open(uploaded.stream)
        original_size = original.size
    except Exception:
        return jsonify({"error": "That file could not be read as an image."}), 400

    mnist_image = to_mnist(original)
    x = to_tensor(mnist_image).unsqueeze(0).to(device)

    with torch.no_grad():
        recon, mu = model.reconstruct(x)

    recon_image = Image.fromarray(
        (recon.squeeze().cpu().numpy() * 255).clip(0, 255).astype("uint8")
    )

    return jsonify(
        {
            "original": png_data_url(mnist_image),
            "reconstructed": png_data_url(recon_image),
            "latent": [round(float(v), 3) for v in mu.squeeze().tolist()],
            "input_size": list(original_size),
            "model_size": [28, 28],
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
