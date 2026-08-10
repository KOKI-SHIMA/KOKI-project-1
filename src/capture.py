from pathlib import Path

import cv2


CAMERA_INDEX = 0
OUTPUT_PATH = Path("/workspace/anime-twin/data/test/input.jpg")


camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)

if not camera.isOpened():
    raise RuntimeError("Could not open the camera.")

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

success = False
frame = None

# 最初の数frameは明るさが安定しないため、15frame読み込む
for _ in range(15):
    success, frame = camera.read()

camera.release()

if not success or frame is None:
    raise RuntimeError("Could not capture an image.")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

saved = cv2.imwrite(str(OUTPUT_PATH), frame)

if not saved:
    raise RuntimeError("Could not save the image.")

print(f"Image saved: {OUTPUT_PATH}")