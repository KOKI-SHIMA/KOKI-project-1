# AnimeTwin

AnimeTwin is an AI-powered visual similarity project built for the NVIDIA Jetson Orin Nano.

The program captures a portrait using a USB camera, detects and crops the face, compares its visual features with a local anime character dataset, and displays the Top 3 results in the terminal.

## Features

- Captures an image using a USB camera
- Detects faces with OpenCV YuNet
- Automatically crops the detected face
- Extracts visual features using OpenAI CLIP
- Supports multiple reference images for each character
- Calculates an average feature vector for each character
- Displays the Top 3 results in the terminal
- Saves the complete ranking as a CSV file
- Runs the complete pipeline with one command

## How It Works

1. The USB camera captures an image.
2. YuNet detects the largest face in the image.
3. The detected face is cropped.
4. CLIP converts the face image into a feature vector.
5. CLIP also converts the character images into feature vectors.
6. The program calculates an average vector for each character.
7. Cosine similarity is used to rank the characters.
8. The Top 3 results are displayed in the terminal.

## Technologies

- NVIDIA Jetson Orin Nano
- Python
- PyTorch
- CUDA
- OpenCV
- YuNet
- OpenAI CLIP
- Docker
- jetson-containers

## Project Structure

```text
anime-twin/
├── src/
│   ├── capture.py
│   ├── detect_face.py
│   ├── compare_clip.py
│   └── main.py
├── data/
│   ├── characters/
│   └── test/
├── models/
├── output/
├── requirements.txt
├── README.md
└── .gitignore
```

## Character Dataset

Create one folder for each character inside `data/characters`.

```text
data/characters/
├── character_a/
│   ├── image_01.jpg
│   ├── image_02.jpg
│   └── image_03.jpg
└── character_b/
    ├── image_01.jpg
    ├── image_02.jpg
    └── image_03.jpg
```

The folder name is used as the character name in the results.

Supported image formats:

- JPG
- JPEG
- PNG
- WebP

## Run the Program

Run the following command inside the Docker container:

```bash
cd /workspace/anime-twin
python3 src/main.py
```

The program will automatically:

1. Capture an image
2. Detect and crop the face
3. Compare the image with the character dataset
4. Display the Top 3 results

To reuse the existing input image without taking a new photo:

```bash
python3 src/main.py --no-capture
```

## Output

The ranking is displayed directly in the terminal.

Example:

```text
Top 3 results
1. character_a (similarity: 0.5850, images: 3)
2. character_b (similarity: 0.5700, images: 3)
3. character_c (similarity: 0.5500, images: 3)
```

The complete results are also saved to:

```text
output/clip_results.csv
```

## Important Notes

- Cosine similarity is a similarity score, not a probability.
- CLIP is a general-purpose image model and is not specifically trained for facial recognition.
- Character images, captured photos, downloaded models, and generated results are not included in this repository.
- Only images that can be used legally and ethically should be added to the local dataset.
- Permission should be obtained before using a person's photograph.

## Limitations

- Results depend on lighting, camera angle, image quality, and the reference dataset.
- The program selects the closest match from the available characters.
- The result does not prove that two faces are objectively identical.

## Project Status

Working prototype completed on NVIDIA Jetson Orin Nano.