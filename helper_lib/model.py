import torch.nn as nn


class Assignment2CNN(nn.Module):
    """
    CNN architecture from Assignment 2.
    Kept here so the previous /predict endpoint can still work.
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
    Assignment 3 GAN Generator.

    Required architecture:
    - Input: noise vector of shape (batch_size, 100)
    - Fully connected layer to 7 x 7 x 128, then reshape
    - ConvTranspose2D: 128 -> 64, kernel size 4, stride 2, padding 1
      Output size: 14 x 14
      Followed by BatchNorm2D and ReLU
    - ConvTranspose2D: 64 -> 1, kernel size 4, stride 2, padding 1
      Output size: 28 x 28
      Followed by Tanh activation
    """

    def __init__(self, latent_dim=100):
        super().__init__()

        self.latent_dim = latent_dim

        self.fc = nn.Linear(latent_dim, 7 * 7 * 128)

        self.generator = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=1,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(z.size(0), 128, 7, 7)
        x = self.generator(x)
        return x


class Discriminator(nn.Module):
    """
    Assignment 3 GAN Discriminator.

    Required architecture:
    - Input: image of shape (1, 28, 28)
    - Conv2D: 1 -> 64, kernel size 4, stride 2, padding 1
      Output size: 14 x 14
      Followed by LeakyReLU(0.2)
    - Conv2D: 64 -> 128, kernel size 4, stride 2, padding 1
      Output size: 7 x 7
      Followed by BatchNorm2D and LeakyReLU(0.2)
    - Flatten
    - Linear layer to get a single real/fake probability
    """

    def __init__(self):
        super().__init__()

        self.discriminator = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(0.2),

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        return self.discriminator(img)


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
    """

    if model_name == "CNN":
        return Assignment2CNN()

    if model_name == "GAN":
        return GAN(latent_dim=100)

    raise ValueError(f"Unknown model name: {model_name}")