# Assignment 3: GAN Image Generation

This project implements a Generative Adversarial Network (GAN) using PyTorch and adds it to a FastAPI application with Docker deployment.

The GAN is trained on the MNIST dataset to generate handwritten digit images.

## Project Structure

```text
SPS_GENAI_ASSIGNMENT3/
├── app/
│   ├── bigram_model.py
│   ├── cifar10_cnn.pth
│   └── gan_generator.pth
├── helper_lib/
│   ├── model.py
│   ├── trainer.py
│   ├── generator.py
│   ├── data_loader.py
│   ├── evaluator.py
│   ├── checkpoints.py
│   └── utils.py
├── main.py
├── train_assignment3.py
├── Dockerfile
├── pyproject.toml
├── uv.lock
└── README.md
```

## Overview

Assignment 3 focuses on image generation with a GAN.

The project includes:

1. A PyTorch GAN model.
2. A training script using the MNIST dataset.
3. Helper library updates for the GAN model, training loop, and image generation.
4. A FastAPI endpoint that returns generated images.
5. Docker deployment so the API can run on another machine.

## GAN Components

### Generator

The generator takes a random latent vector as input and creates a MNIST-like handwritten digit image.

### Discriminator

The discriminator takes an image as input and predicts whether the image is real or generated.

### Training Loop

The discriminator and generator are trained separately.

The discriminator learns to distinguish real MNIST images from generated images.

The generator learns to create images that can fool the discriminator.

### Image Generator Utility

The helper library includes image generation functions that can:

1. Generate one image.
2. Generate a grid of images.
3. Convert generated images into PNG format for FastAPI responses.

## Training

To train the GAN locally, run:

```bash
python train_assignment3.py
```

This downloads the MNIST dataset, trains the GAN, and saves the trained generator to:

```text
app/gan_generator.pth
```

## Run Locally

Start the FastAPI server locally:

```bash
uvicorn main:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

## Docker Deployment

Build the Docker image:

```bash
docker build -t assignment3-gan .
```

Run the Docker container:

```bash
docker run -p 8000:8000 assignment3-gan
```

Then open:

```text
http://127.0.0.1:8000/docs
```

## API Endpoints

### Root

```text
GET /
```

Returns a list of available endpoints.

### Text Generation

```text
POST /generate
```

Generates text using the bigram text generation model from the earlier assignment.

### Word Embeddings

```text
POST /embeddings
```

Returns a word embedding using spaCy.

### CIFAR-10 Prediction

```text
POST /predict
```

Classifies an uploaded image using the CIFAR-10 CNN model from Assignment 2.

### GAN Single Image Generation

```text
GET /generate_gan_image
```

Returns one generated MNIST-like handwritten digit image as a PNG image.

### GAN Image Grid Generation

```text
GET /generate_gan_grid?num_samples=10
```

Returns a grid of generated MNIST-like handwritten digit images as a PNG image.

The `num_samples` parameter must be between 1 and 25.

## Docker Test Result

The Dockerized FastAPI app was tested successfully.

Both GAN endpoints returned status code 200 and PNG image responses:

```text
GET /generate_gan_image
GET /generate_gan_grid?num_samples=10
```

## Files Updated for Assignment 3

The main Assignment 3 updates are:

```text
helper_lib/model.py
helper_lib/trainer.py
helper_lib/generator.py
train_assignment3.py
main.py
Dockerfile
README.md
```

## Notes

The trained GAN generator is included at:

```text
app/gan_generator.pth
```

This allows the Dockerized API to generate images without requiring the instructor to retrain the model first.