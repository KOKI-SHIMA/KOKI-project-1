from pathlib import Path
import csv

import clip
import torch
from PIL import Image


QUERY_PATH = Path(
    "/workspace/anime-twin/output/face_crop.jpg"
)

CHARACTER_DIRECTORY = Path(
    "/workspace/anime-twin/data/characters"
)

MODEL_DIRECTORY = Path(
    "/workspace/anime-twin/models/clip"
)

RESULT_PATH = Path(
    "/workspace/anime-twin/output/clip_results.csv"
)

MODEL_NAME = "ViT-B/32"

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


def prepare_image(image_path, preprocess):
    with Image.open(image_path) as image:
        if (
            image.mode == "P"
            and "transparency" in image.info
        ):
            image = image.convert("RGBA")

        rgb_image = image.convert("RGB")
        return preprocess(rgb_image)


if not QUERY_PATH.exists():
    raise RuntimeError(
        f"Query image not found: {QUERY_PATH}"
    )

character_paths = sorted(
    path
    for path in CHARACTER_DIRECTORY.iterdir()
    if path.is_file()
    and path.suffix.lower() in SUPPORTED_EXTENSIONS
)

if len(character_paths) < 3:
    raise RuntimeError(
        "Add at least 3 character images."
    )


device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

MODEL_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

print(f"Device: {device}")
print(f"Loading CLIP model: {MODEL_NAME}")


model, preprocess = clip.load(
    MODEL_NAME,
    device=device,
    download_root=str(MODEL_DIRECTORY),
)

model.eval()


query_tensor = prepare_image(
    QUERY_PATH,
    preprocess,
).unsqueeze(0).to(device)

character_tensors = torch.stack(
    [
        prepare_image(path, preprocess)
        for path in character_paths
    ]
).to(device)


with torch.no_grad():
    query_features = model.encode_image(
        query_tensor
    )

    character_features = model.encode_image(
        character_tensors
    )


query_features = (
    query_features
    / query_features.norm(
        dim=-1,
        keepdim=True,
    )
)

character_features = (
    character_features
    / character_features.norm(
        dim=-1,
        keepdim=True,
    )
)


similarities = (
    query_features
    @ character_features.T
)[0]


top_count = min(
    3,
    len(character_paths),
)

top_values, top_indices = similarities.topk(
    top_count
)


RESULT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


results = []

for rank, (score, index) in enumerate(
    zip(
        top_values.cpu().tolist(),
        top_indices.cpu().tolist(),
    ),
    start=1,
):
    character_path = character_paths[index]

    result = {
        "rank": rank,
        "character": character_path.stem,
        "cosine_similarity": score,
    }

    results.append(result)


with RESULT_PATH.open(
    "w",
    newline="",
    encoding="utf-8",
) as result_file:
    writer = csv.DictWriter(
        result_file,
        fieldnames=[
            "rank",
            "character",
            "cosine_similarity",
        ],
    )

    writer.writeheader()
    writer.writerows(results)


print("\nTop 3 results")

for result in results:
    print(
        f"{result['rank']}. "
        f"{result['character']} "
        f"(similarity: "
        f"{result['cosine_similarity']:.4f})"
    )

print(f"\nResults saved: {RESULT_PATH}")