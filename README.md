# AnimeTwin

AnimeTwin is an AI-powered visual similarity project built for the NVIDIA Jetson Orin Nano.

The program captures a portrait using a USB camera, detects and crops the face, compares its visual features with a local anime character dataset, and displays the Top 3 results in the terminal.

## Demo Video

This video demonstrates the complete AnimeTwin pipeline running on the NVIDIA Jetson Orin Nano.

[Watch the AnimeTwin demonstration video](https://drive.google.com/file/d/1kyNhTPseplqxdmkhCdjlsBTNwJhcE8sM/view?usp=sharing)

## Why This Project Matters

Anime and manga have a large global audience. In March 2026, the official ONE PIECE website announced that the manga had surpassed 600 million copies in worldwide circulation, including more than 150 million copies outside Japan. This demonstrates the enormous international interest in anime characters and related experiences.

AnimeTwin turns that interest into an interactive AI experience. Instead of only viewing a character, users can take a photo and discover which available character has the most similar visual features.

The international cosplay community is also expanding through events and competitions in many countries. AnimeTwin may help cosplayers explore character ideas and discover characters with visually similar features or styles. However, its results should be treated as creative suggestions, not judgments about a person's face or body.

Sources:

- [Official ONE PIECE announcement: over 600 million copies worldwide](https://one-piece.com/news/78258/index.html)
- [World Cosplay Summit](https://worldcosplaysummit.jp/2025/en/)

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

## Tested Environment

This project was tested with:

- NVIDIA Jetson Orin Nano
- JetPack 6.0
- L4T 36.3.0
- CUDA 12.2
- Python 3
- PyTorch 2.2.0
- OpenCV 4.8.1
- USB V4L2 camera

Other environments may require different package versions.

## Installation

### 1. Install jetson-containers

Follow the official installation instructions:

```bash
git clone https://github.com/dusty-nv/jetson-containers
bash jetson-containers/install.sh
```

### 2. Clone AnimeTwin

```bash
cd /home/nvidia
git clone https://github.com/KOKI-SHIMA/KOKI-project-1.git anime-twin
cd anime-twin
```

### 3. Create the Local Directories

Images, downloaded models, and generated results are excluded from GitHub. Create their directories manually:

```bash
mkdir -p data/characters
mkdir -p data/test
mkdir -p models/clip
mkdir -p output
```

### 4. Start the PyTorch Container

```bash
jetson-containers run \
  --name anime-twin \
  --volume /home/nvidia/anime-twin:/workspace/anime-twin \
  $(autotag l4t-pytorch)
```

The commands below should be executed inside the container.

### 5. Install the Additional Python Packages

```bash
cd /workspace/anime-twin
python3 -m pip install -r requirements.txt
```

Install OpenAI CLIP from its official repository:

```bash
python3 -m pip install --no-deps \
  git+https://github.com/openai/CLIP.git
```

If the NVIDIA package index cannot be reached, use the public PyPI index:

```bash
env -u PIP_INDEX_URL \
  -u PIP_EXTRA_INDEX_URL \
  PIP_CONFIG_FILE=/dev/null \
  python3 -m pip install \
  --index-url https://pypi.org/simple \
  -r requirements.txt
```

### 6. Download the YuNet Model

```bash
wget \
  -O models/face_detection_yunet_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

Confirm that the model was downloaded:

```bash
ls -lh models/face_detection_yunet_2023mar.onnx
```

### 7. Add Reference Images

Before running AnimeTwin, add at least three reference folders to:

```text
data/characters/
```

Each folder must contain at least one supported image. Using three to five images per category is recommended.

## Preparing the Reference Dataset

Reference images are not included in this repository. Before running the program, users must add their own images locally.

Inside `data/characters`, create one folder for each character:

```text
data/characters/
├── character_a/
│   ├── image_01.jpg
│   ├── image_02.jpg
│   └── image_03.jpg
├── character_b/
│   ├── image_01.jpg
│   ├── image_02.jpg
│   └── image_03.jpg
└── character_c/
    ├── image_01.jpg
    ├── image_02.jpg
    └── image_03.jpg
```

The folder name becomes the name displayed in the results.

For example:

```text
data/characters/
└── example_character/
    ├── front.jpg
    ├── side.jpg
    └── smile.png
```

### How to Add a New Character

1. Open the `data/characters` folder.
2. Create a new folder for the character.
3. Name the folder using the name that should appear in the results.
4. Add one or more reference images to the new folder.
5. Run `python3 src/main.py` again.

No Python code needs to be changed when a new folder is added.

For more stable results, use approximately three to five reference images for each character. Images with different facial angles and expressions can reduce the influence of a single reference image.

Recommended image preparation:

- Place the face near the center.
- Use a similar crop for every reference image.
- Prefer face or head-and-shoulders images.
- Avoid large backgrounds and unrelated objects.
- Keep the original aspect ratio.
- Use only images that can be used legally and ethically.

Supported formats:

- `.jpg`
- `.jpeg`
- `.png`
- `.webp`

Character images are excluded from GitHub by `.gitignore`.

## Use Cases Beyond Anime

Although AnimeTwin was designed for anime characters, its comparison system is not limited to anime.

The program treats each subfolder in `data/characters` as one searchable category. By replacing these folders and reference images, the same system can rank visually similar examples from other datasets, such as:

- Original characters
- Video game characters
- Mascots
- Art styles
- Clothing or costume styles
- Objects or products

For example:

```text
data/characters/
├── red_sneakers/
├── blue_sneakers/
└── black_sneakers/
```

In this example, the program would compare the input image with three sneaker categories instead of anime characters.

The current face-detection step is designed for portrait input. Comparing general objects would require changing or skipping `detect_face.py`, but the CLIP similarity system itself can compare many kinds of images.

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
