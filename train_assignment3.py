import os
import shutil

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from helper_lib.model import get_model
from helper_lib.trainer import train_gan


def main():
    """
    Train the Assignment 3 GAN model on MNIST.
    The trained generator will be saved for use in the FastAPI endpoint.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
    )

    model = get_model("GAN")

    trained_model = train_gan(
        model=model,
        data_loader=train_loader,
        device=device,
        epochs=1,
        latent_dim=100,
        lr=0.0002,
        checkpoint_dir="checkpoints",
    )

    os.makedirs("app", exist_ok=True)

    source_path = "checkpoints/gan_generator.pth"
    target_path = "app/gan_generator.pth"

    if os.path.exists(source_path):
        shutil.copy(source_path, target_path)
        print(f"Saved trained generator to {target_path}")
    else:
        torch.save(trained_model.generator.state_dict(), target_path)
        print(f"Saved trained generator to {target_path}")


if __name__ == "__main__":
    main()