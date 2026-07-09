import torch
from torchvision import datasets, transforms


def get_data_loader(
    dataset_name: str = "CIFAR10",
    data_dir: str = "./data",
    batch_size: int = 64,
    train: bool = True,
):
    """
    Create and return a data loader.

    For Assignment 2, we use CIFAR-10 and resize images to 64x64
    because the required CNN architecture expects RGB images of size 64x64x3.
    """

    if dataset_name != "CIFAR10":
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])

    dataset = datasets.CIFAR10(
        root=data_dir,
        train=train,
        download=True,
        transform=transform,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
    )

    return data_loader