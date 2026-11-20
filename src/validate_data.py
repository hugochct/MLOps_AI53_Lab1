from __future__ import annotations

import json
from pathlib import Path
from PIL import Image

EXPECTED_SPLITS = {"train": 1034, "validation": 133, "test": 128}
EXPECTED_CLASSES = {"angular_leaf_spot", "bean_rust", "healthy"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def validate_dataset(raw_root: Path) -> dict:
    """Validate the raw dataset and return a deterministic quality report.

    Required Core checks:
      1. required splits exist;
      2. exactly the expected classes exist in every split;
      3. expected image counts are present;
      4. every image can be fully decoded by Pillow (`Image.load()` or equivalent);
      5. no split or class is empty.

    TODO: implement the checks and return a JSON-serializable report.
    The report must not contain timestamps because it is a DVC output.
    """
    split_counts: dict[str, int] = {}

    for split, expected_count in EXPECTED_SPLITS.items():
        split_dir = raw_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing required split: {split}")

        class_dirs = {
            path.name: path
            for path in split_dir.iterdir()
            if path.is_dir()
        }
        found_classes = set(class_dirs)

        if found_classes != EXPECTED_CLASSES:
            raise RuntimeError(
                f"Unexpected classes in split {split}: "
                f"found={sorted(found_classes)}, expected={sorted(EXPECTED_CLASSES)}"
            )

        split_image_paths: list[Path] = []
        for class_name in sorted(EXPECTED_CLASSES):
            class_dir = class_dirs[class_name]
            image_paths = sorted(
                (
                    path
                    for path in class_dir.rglob("*")
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                ),
                key=lambda path: str(path),
            )
            if not image_paths:
                raise RuntimeError(
                    f"Class {class_name} is empty in split {split}"
                )

            for image_path in image_paths:
                with Image.open(image_path) as image:
                    image.load()

            split_image_paths.extend(image_paths)

        split_count = len(split_image_paths)
        if split_count != expected_count:
            raise RuntimeError(
                f"Unexpected image count in split {split}: "
                f"found={split_count}, expected={expected_count}"
            )
        split_counts[split] = split_count

    return {
        "status": "PASS",
        "splits": split_counts,
        "total_images": sum(split_counts.values()),
    }


def main() -> None:
    report = validate_dataset(Path("data/raw"))
    Path("reports").mkdir(exist_ok=True)
    Path("reports/data_quality.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

# Milestone 6 check: validation logic unchanged