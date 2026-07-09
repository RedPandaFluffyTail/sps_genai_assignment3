import os
import torch
from tqdm import tqdm

from .evaluator import evaluate_model
from .checkpoints import save_checkpoint


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device="cpu",
    epochs=10,
    checkpoint_dir="checkpoints",
):
    """
    Enhanced training loop with checkpoint functionality.

    This function:
    1. Trains the model for the specified number of epochs
    2. Tracks training loss and validation metrics
    3. Saves checkpoints after each epoch
    4. Saves the best performing model
    5. Returns the trained model
    """

    model.to(device)

    best_accuracy = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            leave=True,
        )

        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            progress_bar.set_postfix({"train_loss": loss.item()})

        train_loss = running_loss / len(train_loader)

        val_loss, val_accuracy = evaluate_model(
            model,
            val_loader,
            criterion,
            device,
        )

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            loss=val_loss,
            accuracy=val_accuracy,
            checkpoint_dir=checkpoint_dir,
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=val_loss,
                accuracy=val_accuracy,
                checkpoint_dir=checkpoint_dir,
                filename="best_model.pth",
            )

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Val Accuracy: {val_accuracy:.2f}%"
        )

    return model

def train_gan(
    model,
    data_loader,
    criterion=None,
    optimizer=None,
    device="cpu",
    epochs=10,
    latent_dim=100,
    lr=0.0002,
    checkpoint_dir="checkpoints",
):
    """
    Train a GAN model on image data.

    The model should be the GAN wrapper from helper_lib.model.
    It contains:
    - model.generator
    - model.discriminator

    This function trains the discriminator and generator separately.
    """

    os.makedirs(checkpoint_dir, exist_ok=True)

    model.to(device)
    model.generator.train()
    model.discriminator.train()

    if criterion is None:
        criterion = torch.nn.BCELoss()

    optimizer_G = torch.optim.Adam(
        model.generator.parameters(),
        lr=lr,
        betas=(0.5, 0.999),
    )

    optimizer_D = torch.optim.Adam(
        model.discriminator.parameters(),
        lr=lr,
        betas=(0.5, 0.999),
    )

    for epoch in range(epochs):
        generator_loss_total = 0.0
        discriminator_loss_total = 0.0

        progress_bar = tqdm(
            data_loader,
            desc=f"GAN Epoch {epoch + 1}/{epochs}",
            leave=True,
        )

        for batch in progress_bar:
            real_images = batch[0].to(device)
            batch_size = real_images.size(0)

            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # -------------------------
            # Train Discriminator
            # -------------------------
            optimizer_D.zero_grad()

            real_outputs = model.discriminator(real_images)
            real_loss = criterion(real_outputs, real_labels)

            noise = torch.randn(batch_size, latent_dim, device=device)
            fake_images = model.generator(noise)

            fake_outputs = model.discriminator(fake_images.detach())
            fake_loss = criterion(fake_outputs, fake_labels)

            discriminator_loss = (real_loss + fake_loss) / 2
            discriminator_loss.backward()
            optimizer_D.step()

            # -------------------------
            # Train Generator
            # -------------------------
            optimizer_G.zero_grad()

            noise = torch.randn(batch_size, latent_dim, device=device)
            generated_images = model.generator(noise)
            generated_outputs = model.discriminator(generated_images)

            generator_loss = criterion(generated_outputs, real_labels)
            generator_loss.backward()
            optimizer_G.step()

            generator_loss_total += generator_loss.item()
            discriminator_loss_total += discriminator_loss.item()

            progress_bar.set_postfix(
                {
                    "D_loss": discriminator_loss.item(),
                    "G_loss": generator_loss.item(),
                }
            )

        avg_generator_loss = generator_loss_total / len(data_loader)
        avg_discriminator_loss = discriminator_loss_total / len(data_loader)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"D Loss: {avg_discriminator_loss:.4f} "
            f"G Loss: {avg_generator_loss:.4f}"
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "generator_state_dict": model.generator.state_dict(),
                "discriminator_state_dict": model.discriminator.state_dict(),
                "optimizer_G_state_dict": optimizer_G.state_dict(),
                "optimizer_D_state_dict": optimizer_D.state_dict(),
                "generator_loss": avg_generator_loss,
                "discriminator_loss": avg_discriminator_loss,
                "latent_dim": latent_dim,
            },
            os.path.join(checkpoint_dir, "gan_checkpoint.pth"),
        )

        torch.save(
            model.generator.state_dict(),
            os.path.join(checkpoint_dir, "gan_generator.pth"),
        )

    return model