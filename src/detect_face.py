from pathlib import Path

import cv2


INPUT_PATH = Path("/workspace/anime-twin/data/test/input.jpg")

MODEL_PATH = Path(
    "/workspace/anime-twin/models/"
    "face_detection_yunet_2023mar.onnx"
)

FACE_OUTPUT_PATH = Path(
    "/workspace/anime-twin/output/face_crop.jpg"
)

PREVIEW_OUTPUT_PATH = Path(
    "/workspace/anime-twin/output/face_detected.jpg"
)


image = cv2.imread(str(INPUT_PATH))

if image is None:
    raise RuntimeError(f"Could not read image: {INPUT_PATH}")

if not MODEL_PATH.exists():
    raise RuntimeError(f"Face detection model not found: {MODEL_PATH}")


image_height, image_width = image.shape[:2]

face_detector = cv2.FaceDetectorYN.create(
    str(MODEL_PATH),
    "",
    (image_width, image_height),
    0.7,
    0.3,
    5000,
)


_, faces = face_detector.detect(image)

if faces is None or len(faces) == 0:
    raise RuntimeError(
        "No face was detected. "
        "Use a clear, upright, front-facing portrait."
    )


# 複数検出した場合、最も面積が大きいfaceを選ぶ
largest_face = max(
    faces,
    key=lambda face: face[2] * face[3],
)

x = int(largest_face[0])
y = int(largest_face[1])
width = int(largest_face[2])
height = int(largest_face[3])
confidence = float(largest_face[-1])


# 髪や輪郭も含めるため、検出範囲を25%広げる
padding = int(max(width, height) * 0.25)

x1 = max(0, x - padding)
y1 = max(0, y - padding)
x2 = min(image_width, x + width + padding)
y2 = min(image_height, y + height + padding)


face_crop = image[y1:y2, x1:x2]

if face_crop.size == 0:
    raise RuntimeError("The detected face crop was empty.")


FACE_OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


face_saved = cv2.imwrite(
    str(FACE_OUTPUT_PATH),
    face_crop,
)


preview = image.copy()

cv2.rectangle(
    preview,
    (x1, y1),
    (x2, y2),
    (0, 255, 0),
    3,
)

cv2.putText(
    preview,
    f"Face: {confidence:.2f}",
    (x1, max(30, y1 - 10)),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 255, 0),
    2,
)

preview_saved = cv2.imwrite(
    str(PREVIEW_OUTPUT_PATH),
    preview,
)


if not face_saved or not preview_saved:
    raise RuntimeError("Could not save detection results.")


print(f"Detected faces: {len(faces)}")
print(f"Confidence: {confidence:.3f}")
print(f"Face crop saved: {FACE_OUTPUT_PATH}")
print(f"Preview saved: {PREVIEW_OUTPUT_PATH}")