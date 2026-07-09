import io
import torch
from PIL import Image
from torchvision.utils import make_grid


def _get_generator(model):
    """
    Accept either a GAN wrapper model or a generator-only model.
    """
    if hasattr(model, "generator"):
        return model.generator
    return model


def tensor_to_pil_image(image_tensor):
    """
    Convert one generated MNIST tensor to a PIL image.

    Generator output range is [-1, 1] because of Tanh.
    This function converts it back to [0, 255].
    """

    image_tensor = image_tensor.detach().cpu()
    image_tensor = (image_tensor + 1) / 2
    image_tensor = image_tensor.clamp(0, 1)

    if image_tensor.dim() == 3:
        image_tensor = image_tensor.squeeze(0)

    image_array = (image_tensor.numpy() * 255).astype("uint8")

    return Image.fromarray(image_array, mode="L")


def generate_samples(model, device="cpu", num_samples=10, latent_dim=100):
    """
    Generate a grid of MNIST-like images from random latent vectors.

    Returns:
        PIL.Image
    """

    generator = _get_generator(model)
    generator.to(device)
    generator.eval()

    with torch.no_grad():
        noise = torch.randn(num_samples, latent_dim, device=device)
        generated_images = generator(noise)

        generated_images = (generated_images + 1) / 2
        generated_images = generated_images.clamp(0, 1)

        grid = make_grid(
            generated_images,
            nrow=min(num_samples, 5),
            padding=2,
            normalize=False,
        )

        grid = grid.detach().cpu()

        if grid.shape[0] == 1:
            image_array = (grid.squeeze(0).numpy() * 255).astype("uint8")
            image = Image.fromarray(image_array, mode="L")
        else:
            image_array = (
                grid.permute(1, 2, 0).numpy() * 255
            ).astype("uint8")
            image = Image.fromarray(image_array)

    return image


def generate_single_image(model, device="cpu", latent_dim=100):
    """
    Generate one MNIST-like image.

    Returns:
        PIL.Image
    """

    generator = _get_generator(model)
    generator.to(device)
    generator.eval()

    with torch.no_grad():
        noise = torch.randn(1, latent_dim, device=device)
        generated_image = generator(noise)[0]

    return tensor_to_pil_image(generated_image)


def pil_image_to_bytes(image, image_format="PNG"):
    """
    Convert a PIL image to bytes so FastAPI can return it.
    """

    image_buffer = io.BytesIO()
    image.save(image_buffer, format=image_format)
    image_buffer.seek(0)
    return image_buffer