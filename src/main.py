from pathlib import Path
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

PROCESS_STEPS = [
    (
        "Face detection",
        PROJECT_DIRECTORY / "src" / "detect_face.py",
    ),
    (
        "CLIP comparison",
        PROJECT_DIRECTORY / "src" / "compare_clip.py",
    ),
    (
        "Result page",
        PROJECT_DIRECTORY
        / "src"
        / "create_result_page.py",
    ),
]


if not INPUT_PATH.exists():
    raise RuntimeError(
        f"Input image not found: {INPUT_PATH}"
    )


print("AnimeTwin")
print(f"Input: {INPUT_PATH}")


for step_name, script_path in PROCESS_STEPS:
    print(f"\n--- {step_name} ---")

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


print("\nAnimeTwin completed successfully.")

print(
    "CSV result: "
    "/workspace/anime-twin/output/"
    "clip_results.csv"
)

print(
    "HTML result: "
    "/workspace/anime-twin/output/"
    "result.html"
)