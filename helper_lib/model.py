import torch
import torch.nn as nn


class Assignment2CNN(nn.Module):
    """
    CNN architecture required by Assignment 2.

    Input: RGB image of size 64x64x3
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class Generator(nn.Module):
    """
    GAN Generator for MNIST.

    Input:
        random noise vector of shape (batch_size, latent_dim)

    Output:
        generated image of shape (batch_size, 1, 28, 28)
    """

    def __init__(self, latent_dim=100):
        super().__init__()

        self.latent_dim = latent_dim

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256 * 7 * 7),
            nn.BatchNorm1d(256 * 7 * 7),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Unflatten(1, (256, 7, 7)),

            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                in_channels=64,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    """
    GAN Discriminator for MNIST.

    Input:
        image of shape (batch_size, 1, 28, 28)

    Output:
        probability of image being real, shape (batch_size, 1)
    """

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        return self.model(img)


class GAN(nn.Module):
    """
    Wrapper class containing both the generator and discriminator.
    """

    def __init__(self, latent_dim=100):
        super().__init__()

        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim=latent_dim)
        self.discriminator = Discriminator()


def get_model(model_name: str):
    """
    Return a model based on the model name.
    This follows the helper library pattern from the class guide.
    """

    if model_name == "CNN":
        return Assignment2CNN()

    if model_name == "GAN":
        return GAN(latent_dim=100)

    raise ValueError(f"Unknown model name: {model_name}")