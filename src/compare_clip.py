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

if not CHARACTER_DIRECTORY.exists():
    raise RuntimeError(
        f"Character directory not found: "
        f"{CHARACTER_DIRECTORY}"
    )


# characterごとのfolderを探す
character_directories = sorted(
    path
    for path in CHARACTER_DIRECTORY.iterdir()
    if path.is_dir()
)

if len(character_directories) < 3:
    raise RuntimeError(
        "Add at least 3 character folders."
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


# 撮影した顔画像をCLIP用のTensorに変換
query_tensor = prepare_image(
    QUERY_PATH,
    preprocess,
).unsqueeze(0).to(device)


with torch.no_grad():
    query_features = model.encode_image(
        query_tensor
    )

query_features = (
    query_features
    / query_features.norm(
        dim=-1,
        keepdim=True,
    )
)


character_names = []
character_features_list = []
character_image_counts = []


# 各characterのfolderを順番に処理
for character_directory in character_directories:
    image_paths = sorted(
        path
        for path in character_directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    valid_tensors = []

    for image_path in image_paths:
        try:
            image_tensor = prepare_image(
                image_path,
                preprocess,
            )
            valid_tensors.append(image_tensor)

        except Exception as error:
            print(
                f"Warning: Could not read "
                f"{image_path.name}: {error}"
            )

    if not valid_tensors:
        print(
            f"Warning: No valid images in "
            f"{character_directory.name}"
        )
        continue

    image_batch = torch.stack(
        valid_tensors
    ).to(device)

    with torch.no_grad():
        image_features = model.encode_image(
            image_batch
        )

    # 各画像の特徴をnormalize
    image_features = (
        image_features
        / image_features.norm(
            dim=-1,
            keepdim=True,
        )
    )

    # 同じcharacterの複数画像から平均特徴を作る
    average_features = image_features.mean(
        dim=0,
        keepdim=True,
    )

    # 平均後にもう一度normalize
    average_features = (
        average_features
        / average_features.norm(
            dim=-1,
            keepdim=True,
        )
    )

    character_names.append(
        character_directory.name
    )

    character_features_list.append(
        average_features
    )

    character_image_counts.append(
        len(valid_tensors)
    )

    print(
        f"Loaded {len(valid_tensors)} image(s): "
        f"{character_directory.name}"
    )


if len(character_features_list) < 3:
    raise RuntimeError(
        "At least 3 character folders must "
        "contain valid images."
    )


character_features = torch.cat(
    character_features_list,
    dim=0,
)


# Queryと各characterの平均特徴を比較
similarities = (
    query_features
    @ character_features.T
)[0]


top_count = min(
    3,
    len(character_names),
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
    result = {
        "rank": rank,
        "character": character_names[index],
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
    character_index = character_names.index(
        result["character"]
    )

    image_count = character_image_counts[
        character_index
    ]

    print(
        f"{result['rank']}. "
        f"{result['character']} "
        f"(similarity: "
        f"{result['cosine_similarity']:.4f}, "
        f"images: {image_count})"
    )


print(f"\nResults saved: {RESULT_PATH}")