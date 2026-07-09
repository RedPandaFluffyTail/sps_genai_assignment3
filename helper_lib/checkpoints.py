import os
import torch


def save_checkpoint(
    model,
    optimizer,
    epoch,
    loss,
    accuracy,
    checkpoint_dir="checkpoints",
    filename=None,
):
    """
    Save a model checkpoint.

    The checkpoint includes:
    - model state
    - optimizer state
    - epoch
    - loss
    - accuracy
    """

    os.makedirs(checkpoint_dir, exist_ok=True)

    if filename is None:
        filename = f"checkpoint_epoch_{epoch + 1}.pth"

    checkpoint_path = os.path.join(checkpoint_dir, filename)

    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "accuracy": accuracy,
        },
        checkpoint_path,
    )

    return checkpoint_path


def load_checkpoint(model, optimizer, checkpoint_path, device):
    """
    Load a model checkpoint.
    """

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    accuracy = checkpoint["accuracy"]

    return model, optimizer, epoch, loss, accuracy