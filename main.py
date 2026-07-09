from pathlib import Path
import io

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from PIL import Image
from torchvision import transforms
import spacy

from app.bigram_model import BigramModel
from helper_lib.model import get_model
from helper_lib.generator import generate_single_image, generate_samples, pil_image_to_bytes


app = FastAPI(
    title="Assignment 3 API",
    description=(
        "FastAPI app with text generation, CIFAR-10 classification, "
        "and Assignment 3 GAN image generation."
    ),
    version="3.0.0",
)


# -----------------------------
# Assignment 1: Bigram API
# -----------------------------

corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas",
    "It tells the story of Edmond Dantes who is falsely imprisoned and later seeks revenge",
    "This is another example sentence",
    "We are generating text based on bigram probabilities",
    "Bigram models are simple but effective",
]

bigram_model = BigramModel(corpus)


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


class EmbeddingRequest(BaseModel):
    query_word: str


nlp = None


def get_spacy_model():
    """
    Load spaCy model only when the embeddings endpoint is called.
    This prevents spaCy from blocking the image endpoints if the model is missing.
    """
    global nlp

    if nlp is None:
        try:
            nlp = spacy.load("en_core_web_lg")
        except OSError as exc:
            raise HTTPException(
                status_code=500,
                detail="spaCy model en_core_web_lg is not installed.",
            ) from exc

    return nlp


# -----------------------------
# Shared device
# -----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Assignment 2: CIFAR-10 CNN API
# -----------------------------

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

cnn_model = None
cnn_model_path = Path(__file__).resolve().parent / "app" / "cifar10_cnn.pth"

if cnn_model_path.exists():
    cnn_model = get_model("CNN")
    cnn_model.load_state_dict(torch.load(cnn_model_path, map_location=device))
    cnn_model.to(device)
    cnn_model.eval()

image_transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ]
)


# -----------------------------
# Assignment 3: GAN Image Generation API
# -----------------------------

gan_model_path = Path(__file__).resolve().parent / "app" / "gan_generator.pth"

gan_model = get_model("GAN")
if gan_model_path.exists():
    gan_model.generator.load_state_dict(
        torch.load(gan_model_path, map_location=device)
    )
else:
    print(
        "Warning: app/gan_generator.pth was not found. "
        "The GAN endpoint will use an untrained generator."
    )

gan_model.to(device)
gan_model.generator.eval()


# -----------------------------
# API Endpoints
# -----------------------------

@app.get("/")
def read_root():
    return {
        "message": "Assignment 3 API is running",
        "endpoints": [
            "/generate",
            "/embeddings",
            "/predict",
            "/generate_gan_image",
            "/generate_gan_grid",
        ],
    }


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(
        request.start_word,
        request.length,
    )

    return {
        "generated_text": generated_text,
    }


@app.post("/embeddings")
def get_embeddings(request: EmbeddingRequest):
    word = request.query_word.strip()

    if not word:
        raise HTTPException(
            status_code=400,
            detail="query_word cannot be empty",
        )

    spacy_model = get_spacy_model()
    doc = spacy_model(word)

    if len(doc) != 1:
        raise HTTPException(
            status_code=400,
            detail="Please provide one word only",
        )

    token = doc[0]

    if not token.has_vector:
        raise HTTPException(
            status_code=404,
            detail=f"No embedding found for word: {word}",
        )

    return {
        "word": token.text,
        "dimension": len(token.vector),
        "embedding": token.vector.tolist(),
    }


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Predict the CIFAR-10 class of an uploaded image.
    """

    if cnn_model is None:
        raise HTTPException(
            status_code=500,
            detail="CIFAR-10 CNN model file app/cifar10_cnn.pth was not found.",
        )

    image_bytes = await file.read()

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file.",
        ) from exc

    input_tensor = image_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = cnn_model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    class_index = predicted_class.item()
    class_name = CLASS_NAMES[class_index]

    return {
        "predicted_class": class_name,
        "class_index": class_index,
        "confidence": round(confidence.item(), 4),
    }


@app.get("/generate_gan_image")
def generate_gan_image():
    """
    Generate one MNIST-like handwritten digit image using the trained GAN generator.
    Returns a PNG image.
    """

    image = generate_single_image(
        model=gan_model,
        device=device,
        latent_dim=100,
    )

    image_bytes = pil_image_to_bytes(image)

    return StreamingResponse(
        image_bytes,
        media_type="image/png",
    )


@app.get("/generate_gan_grid")
def generate_gan_grid(num_samples: int = 10):
    """
    Generate a grid of MNIST-like handwritten digit images using the trained GAN generator.
    Returns a PNG image.
    """

    if num_samples < 1 or num_samples > 25:
        raise HTTPException(
            status_code=400,
            detail="num_samples must be between 1 and 25.",
        )

    image = generate_samples(
        model=gan_model,
        device=device,
        num_samples=num_samples,
        latent_dim=100,
    )

    image_bytes = pil_image_to_bytes(image)

    return StreamingResponse(
        image_bytes,
        media_type="image/png",
    )