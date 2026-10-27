# 🧠 Image Reconstruction with Convolutional VAE

A **Variational Autoencoder (VAE)** that learns to reconstruct handwritten digits from the MNIST dataset, with a Flask web app for interactive use.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch)
![Flask](https://img.shields.io/badge/Flask-3.0%2B-000?logo=flask)

---

## ✨ Features

- **Convolutional VAE** — Encoder-decoder architecture with a 32-dimensional latent space
- **Auto-resize** — Upload any image size; it gets normalized to 28×28 MNIST format
- **Web UI** — Drag-and-drop Flask interface to upload digits and see reconstructions live
- **Latent space visualization** — PCA projection of the latent codes, colored by digit class

---

## 🏗️ Architecture

```
Input (1×28×28)
  │
  ├─ Encoder ─────────────────────────┐
  │   Conv2d(1→32, 4×4, s=2)  → ReLU │
  │   Conv2d(32→64, 4×4, s=2) → ReLU │
  │   Conv2d(64→128, 3×3)     → ReLU │
  │   Flatten → fc_mu, fc_logvar      │
  │                                    │
  │  Latent z ~ N(μ, σ²)  [dim=32]   │
  │                                    │
  ├─ Decoder ─────────────────────────┘
  │   Linear → Reshape (128×7×7)
  │   ConvT2d(128→64, 4×4, s=2) → ReLU
  │   ConvT2d(64→32, 4×4, s=2)  → ReLU
  │   Conv2d(32→1, 3×3)         → Sigmoid
  │
Output (1×28×28)
```

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/Shamil-Xero/Image_Reconstruction_VAE.git
cd Image_Reconstruction_VAE
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate          # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

```bash
python vae_mnist.py
```

This will:
- Download the MNIST dataset (first run only)
- Train the ConvVAE for 15 epochs
- Save model weights to `vae_mnist.pt`
- Generate `reconstructions.png` and `latent_space.png`

### 5. Launch the web app

```bash
python app.py
```

Open **http://127.0.0.1:5001** in your browser.

---

## 🖥️ Usage

1. Open the web app in your browser
2. Upload (or drag-and-drop) an image of a handwritten digit
3. Click **Reconstruct**
4. View the original (normalized to 28×28) alongside the VAE reconstruction
5. Inspect the first 8 values of the 32-dimensional latent code

---

## 📂 Project Structure

```
.
├── vae_model.py         # ConvVAE model definition & loss function
├── vae_mnist.py         # Training script (MNIST download, train loop, plots)
├── app.py               # Flask web server with /reconstruct endpoint
├── templates/
│   └── index.html       # Upload UI with drag-and-drop
├── static/
│   └── style.css        # Stylesheet
├── samples/             # Sample digit images for testing
├── requirements.txt     # Python dependencies
└── .gitignore
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | PyTorch |
| Dataset | MNIST (torchvision) |
| Web Framework | Flask |
| Image Processing | Pillow |
| Visualization | Matplotlib |

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).
