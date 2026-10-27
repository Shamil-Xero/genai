import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from vae_model import ConvVAE, vae_loss


def maybe_show():
    if plt.get_backend().lower() != "agg":
        plt.show()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

EPOCHS = 15
LEARNING_RATE = 1e-3


def train(model, loader, optimizer, epoch):
    model.train()
    train_loss = 0.0
    for data, _ in loader:
        data = data.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(data)
        loss, _, _ = vae_loss(recon, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    avg_loss = train_loss / len(loader.dataset)
    print(f"Epoch [{epoch}/{EPOCHS}], Loss: {avg_loss:.4f}")
    return avg_loss


@torch.no_grad()
def reconstruction_error(model, loader):
    model.eval()
    total = 0.0
    count = 0
    for data, _ in loader:
        data = data.to(device)
        recon, _ = model.reconstruct(data)
        total += torch.mean((recon - data) ** 2, dim=(1, 2, 3)).sum().item()
        count += data.size(0)
    return total / count


def show_reconstructions(model, loader, num_images=10):
    model.eval()
    data, _ = next(iter(loader))
    data = data[:num_images].to(device)
    with torch.no_grad():
        recon, _ = model.reconstruct(data)
    recon = recon.cpu()
    data = data.cpu()

    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 1.2, 2.5))
    for i in range(num_images):
        axes[0, i].imshow(data[i].squeeze(), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(recon[i].squeeze(), cmap="gray")
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Original")
    axes[1, 0].set_ylabel("Reconstructed")
    plt.suptitle("Original vs Reconstructed MNIST Digits")
    plt.tight_layout()
    plt.savefig("reconstructions.png", dpi=150)
    maybe_show()


def plot_latent_space(model, loader, max_points=5000):
    model.eval()
    zs = []
    labels = []
    count = 0
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            mu, _ = model.encode(data)
            zs.append(mu.cpu())
            labels.append(target)
            count += data.size(0)
            if count >= max_points:
                break
    z = torch.cat(zs, dim=0)[:max_points]
    y = torch.cat(labels, dim=0)[:max_points].numpy()
    z2, _, _ = torch.pca_lowrank(z, q=2)
    z2 = z2.numpy()

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(z2[:, 0], z2[:, 1], c=y, cmap="tab10", alpha=0.6, s=8)
    plt.colorbar(scatter, ticks=range(10), label="Digit")
    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.title("Latent space (PCA of encoder mean μ)")
    plt.tight_layout()
    plt.savefig("latent_space.png", dpi=150)
    maybe_show()


def main():
    print(f"Using device: {device}")
    model = ConvVAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        train(model, train_loader, optimizer, epoch)

    mse = reconstruction_error(model, test_loader)
    print(f"Test reconstruction MSE: {mse:.6f}")
    show_reconstructions(model, test_loader)
    plot_latent_space(model, test_loader)
    torch.save(model.state_dict(), "vae_mnist.pt")
    print("Saved model to vae_mnist.pt")


if __name__ == "__main__":
    main()
