from pathlib import Path
import argparse
import subprocess
import sys


PROJECT_DIRECTORY = Path(
    "/workspace/anime-twin"
)

INPUT_PATH = (
    PROJECT_DIRECTORY
    / "data"
    / "test"
    / "input.jpg"
)

CAPTURE_SCRIPT = (
    PROJECT_DIRECTORY
    / "src"
    / "capture.py"
)

PROCESS_STEPS = [
    (
        "Face detection",
        PROJECT_DIRECTORY
        / "src"
        / "detect_face.py",
    ),
    (
        "CLIP comparison",
        PROJECT_DIRECTORY
        / "src"
        / "compare_clip.py",
    ),
]


def run_script(step_name, script_path):
    print(f"\n--- {step_name} ---")

    if not script_path.exists():
        raise RuntimeError(
            f"Script not found: {script_path}"
        )

    try:
        subprocess.run(
            [
                sys.executable,
                str(script_path),
            ],
            cwd=str(PROJECT_DIRECTORY),
            check=True,
        )

    except subprocess.CalledProcessError as error:
        print(
            f"\nFailed: {step_name}"
            f" (exit code: {error.returncode})"
        )

        sys.exit(error.returncode)


parser = argparse.ArgumentParser(
    description=(
        "Capture a face and find visually "
        "similar anime characters."
    )
)

parser.add_argument(
    "--no-capture",
    action="store_true",
    help=(
        "Use the existing input.jpg "
        "without taking a new photo."
    ),
)

args = parser.parse_args()


print("AnimeTwin")


if args.no_capture:
    print("Camera capture: skipped")

else:
    run_script(
        "Camera capture",
        CAPTURE_SCRIPT,
    )


if not INPUT_PATH.exists():
    raise RuntimeError(
        f"Input image not found: {INPUT_PATH}"
    )


print(f"Input: {INPUT_PATH}")


for step_name, script_path in PROCESS_STEPS:
    run_script(
        step_name,
        script_path,
    )


print("\nAnimeTwin completed successfully.")

print(
    "CSV result: "
    "/workspace/anime-twin/output/"
    "clip_results.csv"
)