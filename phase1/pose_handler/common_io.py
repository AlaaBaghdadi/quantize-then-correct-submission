from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List, Sequence


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: str | Path) -> Any:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def list_images(input_dir: str | Path) -> List[Path]:
    d = Path(input_dir)
    if not d.exists():
        raise FileNotFoundError(f"Input directory not found: {d}")
    if not d.is_dir():
        raise NotADirectoryError(f"Input is not a directory: {d}")
    images = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    images.sort(key=lambda p: p.name.lower())
    return images


def as_posix_str(p: str | Path) -> str:
    return Path(p).as_posix()


def require_file(path: str | Path) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    if not p.is_file():
        raise FileNotFoundError(f"Expected a file, got: {p}")
    return p


def first_existing(paths: Sequence[str | Path]) -> Path | None:
    for p in paths:
        pp = Path(p)
        if pp.exists():
            return pp
    return None

# python -m pose_handler.calc_angles --in_json pose_handler_out/averaged_keypoints.json --out_path pose_handler_out/averaged_angles.json