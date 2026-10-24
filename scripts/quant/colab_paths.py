"""Shared paths and Drive layout helpers for Colab PTQ notebooks."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

DRIVE_ROOT = Path("/content/drive/MyDrive/diffusion_quantization")
DRIVE_SCRIPTS = DRIVE_ROOT / "scripts" / "quant"
REPO = Path("/content/DynamiCtrl")
MODEL_PATH = REPO / "checkpoints" / "DynamiCtrl"
DRIVE_CKPT = DRIVE_ROOT / "checkpoints" / "DynamiCtrl"
DRIVE_DWPOSE = DRIVE_ROOT / "checkpoints" / "DWPose"
LOCAL_DWPOSE = Path("/content/checkpoints/DWPose")

# Bump when changing Drive helpers; notebooks call require_colab_paths_version().
COLAB_PATHS_VERSION = 9

DYNAMICTRL_REPO_ID = "gulucaptain/DynamiCtrl"

# Minimum files required before inference / PTQ (matches gulucaptain/DynamiCtrl on HF).
REQUIRED_CKPT_FILES = (
    "model_index.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
    "text_encoder/config.json",
    "text_encoder/model.safetensors.index.json",
    "text_encoder/model-00001-of-00004.safetensors",
    "text_encoder/model-00002-of-00004.safetensors",
    "text_encoder/model-00003-of-00004.safetensors",
    "text_encoder/model-00004-of-00004.safetensors",
)

# Small JSON sidecars snapshot_download sometimes skips on partial Drive copies.
DYNAMICTRL_CONFIG_FILES = (
    "model_index.json",
    "transformer/config.json",
    "vae/config.json",
    "text_encoder/config.json",
    "tokenizer/tokenizer_config.json",
    "scheduler/scheduler_config.json",
)

_CONSISTENCY_ERR = re.compile(
    r"file should be of size (\d+) but has size (\d+) \(([^)]+)\)",
    re.IGNORECASE,
)

DRIVE_DIRS = [
    DRIVE_ROOT / "checkpoints",
    DRIVE_CKPT.parent,
    DRIVE_DWPOSE,
    DRIVE_SCRIPTS,
    DRIVE_ROOT / "configs" / "quant",
    DRIVE_ROOT / "calib_data",
    DRIVE_ROOT / "outputs" / "fp16",
    DRIVE_ROOT / "outputs" / "w8a8",
    DRIVE_ROOT / "outputs" / "vidit_w8a8",
    DRIVE_ROOT / "outputs" / "hybrid_w8a8_grid",
    DRIVE_ROOT / "outputs" / "hybrid_vidit_w8a8_grid",
    DRIVE_ROOT / "logs",
    DRIVE_ROOT / "logs" / "kinematic_gate_hybrid_grid",
    DRIVE_ROOT / "repo" / "scripts" / "quant",
]


def _tqdm(*args, **kwargs):
    from tqdm.auto import tqdm

    return tqdm(*args, **kwargs)


def _twrite(message: str) -> None:
    from tqdm.auto import tqdm

    tqdm.write(message)


def _iter_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*") if p.is_file())


def _copy_tree_progress(
    src: Path,
    dest: Path,
    desc: str,
    *,
    skip_if_current: bool = False,
) -> int:
    """Copy files under src into dest, showing a byte progress bar. Returns files copied."""
    files = _iter_files(src)
    if not files:
        return 0

    total_bytes = sum(f.stat().st_size for f in files)
    bar = _tqdm(
        total=total_bytes,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=desc,
    )
    copied = 0
    for src_file in files:
        rel = src_file.relative_to(src)
        dest_file = dest / rel
        if skip_if_current and dest_file.exists():
            if dest_file.stat().st_mtime >= src_file.stat().st_mtime:
                bar.update(src_file.stat().st_size)
                continue
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dest_file)
        copied += 1
        bar.update(src_file.stat().st_size)
    bar.close()
    return copied


def require_colab_paths_version(min_version: int = 2) -> None:
    """Fail fast if an old colab_paths.py is still on Drive or cached in the kernel."""
    mod_path = Path(__file__).resolve()
    source = mod_path.read_text(encoding="utf-8")
    if COLAB_PATHS_VERSION < min_version:
        raise RuntimeError(
            f"colab_paths.py version {COLAB_PATHS_VERSION} is too old; need >= {min_version}."
        )
    if "snapshot_download" not in source:
        raise RuntimeError(
            f"Stale colab_paths.py at {mod_path}. Re-upload scripts/quant/ from your repo to Drive."
        )
    _twrite(f"colab_paths v{COLAB_PATHS_VERSION} from {mod_path}")


def ensure_drive_layout() -> None:
    """Create diffusion_quantization folder tree on Google Drive."""
    for path in DRIVE_DIRS:
        path.mkdir(parents=True, exist_ok=True)


def _replace_path(link: Path, target: Path) -> None:
    if link.is_symlink() or link.is_file():
        link.unlink()
    elif link.is_dir() and not link.is_symlink():
        shutil.rmtree(link)
    link.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        os.symlink(target, link)


def link_repo_paths(resume: bool = True) -> None:
    """Symlink Drive-backed dirs into the DynamiCtrl clone."""
    del resume  # symlinks are idempotent
    ensure_drive_layout()
    REPO.mkdir(parents=True, exist_ok=True)
    (REPO / "checkpoints").mkdir(parents=True, exist_ok=True)
    (REPO / "configs").mkdir(parents=True, exist_ok=True)

    links: list[tuple[str, Path, Path]] = []
    if DRIVE_CKPT.exists():
        links.append(("checkpoints/DynamiCtrl", REPO / "checkpoints" / "DynamiCtrl", DRIVE_CKPT))
    links.extend(
        [
            ("calib_data", REPO / "calib_data", DRIVE_ROOT / "calib_data"),
            ("configs/quant", REPO / "configs" / "quant", DRIVE_ROOT / "configs" / "quant"),
        ]
    )

    for label, link, target in _tqdm(links, desc="Linking Drive paths", unit="link"):
        target.mkdir(parents=True, exist_ok=True)
        _replace_path(link, target)
        _twrite(f"  linked {label} -> {target}")


def sync_dwpose_to_local() -> None:
    """Copy DWPose weights to path expected by inference scripts."""
    if not DRIVE_DWPOSE.exists():
        _twrite("DWPose not on Drive yet — skipping local sync.")
        return
    LOCAL_DWPOSE.mkdir(parents=True, exist_ok=True)
    copied = _copy_tree_progress(
        DRIVE_DWPOSE,
        LOCAL_DWPOSE,
        "Sync DWPose to /content",
        skip_if_current=True,
    )
    _twrite(f"DWPose local sync complete ({copied} files updated).")


def _remote_lfs_sizes(repo_id: str) -> dict[str, int]:
    """Map repo-relative paths to expected byte sizes from Hugging Face metadata."""
    from huggingface_hub import HfApi

    info = HfApi().repo_info(repo_id, files_metadata=True)
    sizes: dict[str, int] = {}
    for entry in info.siblings or []:
        if entry.size is not None:
            sizes[entry.rfilename] = int(entry.size)
    return sizes


def _remove_incomplete_downloads(local_dir: Path) -> list[str]:
    """Delete huggingface_hub partial download artifacts under local_dir."""
    removed: list[str] = []
    if not local_dir.exists():
        return removed
    for path in local_dir.rglob("*"):
        if path.is_file() and (path.name.endswith(".incomplete") or path.name.endswith(".lock")):
            path.unlink(missing_ok=True)
            removed.append(str(path.relative_to(local_dir)))
    return removed


def scrub_truncated_files(local_dir: Path, repo_id: str) -> list[str]:
    """Remove local weight files whose size does not match the HF repo (partial downloads)."""
    remote = _remote_lfs_sizes(repo_id)
    removed: list[str] = []
    for rel, expected in remote.items():
        if not rel.endswith(".safetensors"):
            continue
        local = local_dir / rel
        if not local.is_file():
            continue
        actual = local.stat().st_size
        if actual != expected:
            local.unlink()
            removed.append(f"{rel} ({actual:,} B != {expected:,} B)")
    removed.extend(_remove_incomplete_downloads(local_dir))
    return removed


def _missing_config_files(ckpt_dir: Path) -> list[str]:
    """Return config sidecars absent from a checkpoint tree."""
    missing: list[str] = []
    for rel in DYNAMICTRL_CONFIG_FILES:
        path = ckpt_dir / rel
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(rel)
    return missing


def ensure_dynamictrl_config_files(ckpt_dir: Path | None = None) -> list[str]:
    """Fetch missing small HF config files (e.g. transformer/config.json)."""
    root = ckpt_dir or DRIVE_CKPT
    missing = _missing_config_files(root)
    if not missing:
        return []

    from huggingface_hub import hf_hub_download

    root.mkdir(parents=True, exist_ok=True)
    fetched: list[str] = []
    for rel in _tqdm(missing, desc="Fetch DynamiCtrl configs", unit="file"):
        hf_hub_download(
            repo_id=DYNAMICTRL_REPO_ID,
            filename=rel,
            local_dir=str(root),
        )
        fetched.append(rel)
        _twrite(f"  fetched {rel}")
    return fetched


def _missing_ckpt_files(ckpt_dir: Path, remote_sizes: dict[str, int] | None = None) -> list[str]:
    """Return relative paths of required weight files that are absent, empty, or truncated."""
    if remote_sizes is None:
        remote_sizes = _remote_lfs_sizes(DYNAMICTRL_REPO_ID)
    missing: list[str] = []
    for rel in REQUIRED_CKPT_FILES:
        path = ckpt_dir / rel
        expected = remote_sizes.get(rel)
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(rel)
            continue
        if expected is not None and path.stat().st_size != expected:
            missing.append(f"{rel} (truncated)")
    return missing


def is_dynamictrl_checkpoint_complete(ckpt_dir: Path | None = None) -> bool:
    """True when DynamiCtrl HF weights are present enough for pipeline load."""
    root = ckpt_dir or DRIVE_CKPT
    return not _missing_ckpt_files(root)


def validate_dynamictrl_checkpoint(ckpt_dir: Path | None = None) -> None:
    """Raise with actionable guidance when checkpoint files are missing."""
    root = ckpt_dir or DRIVE_CKPT
    missing = _missing_ckpt_files(root)
    if missing:
        preview = "\n  - ".join(missing[:8])
        extra = f"\n  ... and {len(missing) - 8} more" if len(missing) > 8 else ""
        raise FileNotFoundError(
            f"Incomplete DynamiCtrl checkpoint at {root}.\n"
            f"Missing or empty:\n  - {preview}{extra}\n"
            "Re-run setup cell D with RESUME=True (HF will fetch only missing files), "
            "or set RESUME=False to force a full re-download."
        )


def _delete_local_weight_files(local_dir: Path) -> int:
    """Remove all safetensors under local_dir (used when RESUME=False)."""
    count = 0
    if not local_dir.exists():
        return count
    for path in local_dir.rglob("*.safetensors"):
        path.unlink()
        count += 1
    _remove_incomplete_downloads(local_dir)
    return count


def _snapshot_download(
    repo_id: str,
    local_dir: Path,
    desc: str,
    *,
    max_attempts: int = 5,
) -> None:
    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    _twrite(
        f"Downloading {repo_id} -> {local_dir}\n"
        "Progress bars appear below. Safe to interrupt (Ctrl+C); re-run with RESUME=True to continue.\n"
        "Truncated partial files are removed automatically before each attempt."
    )

    # Google Drive FUSE is slow; fewer parallel writers reduces corrupt partial files.
    max_workers = 1 if "drive/MyDrive" in str(local_dir) else 4

    for attempt in range(1, max_attempts + 1):
        removed = scrub_truncated_files(local_dir, repo_id)
        if removed:
            preview = "\n  - ".join(removed[:6])
            extra = f"\n  ... and {len(removed) - 6} more" if len(removed) > 6 else ""
            _twrite(f"Removed truncated/incomplete files before attempt {attempt}:\n  - {preview}{extra}")

        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_dir),
                max_workers=max_workers,
            )
            _twrite(f"Finished {desc}.")
            return
        except OSError as exc:
            message = str(exc)
            if "Consistency check failed" not in message:
                raise

            match = _CONSISTENCY_ERR.search(message)
            if match:
                expected, actual, rel = match.group(1), match.group(2), match.group(3)
                bad_path = local_dir / rel
                if bad_path.is_file():
                    bad_path.unlink()
                _twrite(
                    f"HF consistency check failed for {rel} "
                    f"({actual} B vs expected {expected} B)."
                )
            else:
                scrub_truncated_files(local_dir, repo_id)

            if attempt >= max_attempts:
                raise RuntimeError(
                    f"{desc} failed after {max_attempts} attempts due to truncated downloads.\n"
                    f"Last error: {exc}\n"
                    "Re-run cell D with RESUME=True, or set RESUME=False to wipe safetensors "
                    "and download again. Keep the Colab session alive until all shards reach 100%."
                ) from exc

            _twrite(f"Retrying download ({attempt + 1}/{max_attempts})...")


def download_dynamictrl_weights(resume: bool = True) -> None:
    """Download HF weights to Drive if missing or incomplete."""
    ensure_drive_layout()
    scrub_truncated_files(DRIVE_CKPT, DYNAMICTRL_REPO_ID)

    if resume and is_dynamictrl_checkpoint_complete():
        ensure_dynamictrl_config_files(DRIVE_CKPT)
        if is_dynamictrl_checkpoint_complete():
            _twrite(f"Reusing complete DynamiCtrl weights at {DRIVE_CKPT}")
            return

    if not resume:
        wiped = _delete_local_weight_files(DRIVE_CKPT)
        _twrite(f"RESUME=False: removed {wiped} safetensors file(s) before re-download.")

    missing = _missing_ckpt_files(DRIVE_CKPT)
    if missing:
        _twrite(
            f"DynamiCtrl checkpoint incomplete at {DRIVE_CKPT} "
            f"({len(missing)} required file(s) missing/truncated). Resuming HF download..."
        )
    _snapshot_download(DYNAMICTRL_REPO_ID, DRIVE_CKPT, "DynamiCtrl weights")
    ensure_dynamictrl_config_files(DRIVE_CKPT)
    validate_dynamictrl_checkpoint()


# Backward-compatible alias (typo in earlier version)
download_dynamictl_weights = download_dynamictrl_weights


def download_dwpose_weights(resume: bool = True) -> None:
    """Download DWPose weights to Drive if missing."""
    marker = DRIVE_DWPOSE / "dw-ll_ucoco_384.onnx"
    if resume and marker.exists():
        _twrite(f"Reusing DWPose at {DRIVE_DWPOSE}")
        return
    ensure_drive_layout()
    _snapshot_download("yzd-v/DWPose", DRIVE_DWPOSE, "DWPose weights")


def run_setup_downloads(resume: bool = True) -> None:
    """Run weight download, symlink, and local sync with an overall progress bar."""
    require_colab_paths_version()
    steps = [
        ("DynamiCtrl weights (HF)", lambda: download_dynamictrl_weights(resume=resume)),
        ("DWPose weights (HF)", lambda: download_dwpose_weights(resume=resume)),
        ("Link Drive paths", lambda: link_repo_paths(resume=resume)),
        ("Sync DWPose to /content", sync_dwpose_to_local),
    ]
    for label, fn in _tqdm(steps, desc="Colab setup", unit="step"):
        _twrite(f"\n=== {label} ===")
        fn()
    ensure_dynamictrl_config_files(DRIVE_CKPT)
    validate_dynamictrl_checkpoint(DRIVE_CKPT)
    _twrite("\nCheckpoints linked and DWPose synced.")


def _safe_copy2(src: Path, dst: Path) -> bool:
    """Copy src to dst unless they resolve to the same file. Returns True if copied."""
    try:
        if src.resolve() == dst.resolve():
            return False
    except OSError:
        pass
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _unique_existing_dirs(dirs: list[Path]) -> list[Path]:
    """Drop missing dirs and dedupe by resolved path (symlinks / duplicate entries)."""
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in dirs:
        if not path.exists():
            continue
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _newest_file(candidates: list[Path]) -> Path | None:
    best: Path | None = None
    best_mtime = -1.0
    for path in candidates:
        if not path.is_file():
            continue
        mtime = path.stat().st_mtime
        if mtime > best_mtime:
            best_mtime = mtime
            best = path
    return best


def patch_validate_offline_for_colab(path: Path) -> bool:
    """Upgrade stale validate_offline.py copied from Drive (Colab has no phase2 notebooks)."""
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    if "OFFLINE_VALIDATE_VERSION = 2" in text:
        return False

    original = text
    old_gaps = (
        '    notebook = (PROJECT_ROOT / "notebooks" / "DynamiCtrl-PTQ-ViDiT.ipynb").read_text(\n'
        '        encoding="utf-8"\n'
        '    )'
    )
    new_gaps = (
        '    notebook_path = PROJECT_ROOT / "notebooks" / "DynamiCtrl-PTQ-ViDiT.ipynb"\n'
        '    notebook = notebook_path.read_text(encoding="utf-8") if notebook_path.is_file() else ""'
    )
    text = text.replace(old_gaps, new_gaps)

    start = text.find("def check_notebooks")
    end = text.find("\ndef main()", start)
    if start != -1 and end != -1:
        text = text[:start] + _CHECK_NOTEBOOKS_COLAB_BODY + text[end:]

    if '"--colab"' not in text and "def main()" in text:
        text = text.replace(
            '    parser.add_argument("--skip-help", action="store_true")\n    args = parser.parse_args()',
            '    parser.add_argument("--skip-help", action="store_true")\n'
            '    parser.add_argument(\n'
            '        "--colab",\n'
            '        action="store_true",\n'
            '        help="Skip checks that require phase2 notebooks on disk (Colab DynamiCtrl clone)",\n'
            '    )\n'
            '    args = parser.parse_args()',
        )
        text = text.replace(
            "    check_notebooks()\n",
            "    check_notebooks(colab=args.colab)\n",
        )

    if "OFFLINE_VALIDATE_VERSION" not in text:
        marker = "from pathlib import Path\n\n"
        if marker in text:
            text = text.replace(
                marker,
                marker + "OFFLINE_VALIDATE_VERSION = 2\n\n",
                1,
            )

    if text == original:
        return False
    path.write_text(text, encoding="utf-8")
    _twrite(f"Patched {path.name} for Colab (offline validate v2)")
    return True


_CHECK_NOTEBOOKS_COLAB_BODY = '''def check_notebooks(*, colab: bool = False) -> None:
    if colab:
        print("NOTE skip notebook checks (--colab; no phase2 notebooks in DynamiCtrl clone)")
        return
    notebook_dir = PROJECT_ROOT / "notebooks"
    integration_path = notebook_dir / "DynamiCtrl-PTQ-Integration.ipynb"
    vidit_path = notebook_dir / "DynamiCtrl-PTQ-ViDiT.ipynb"
    hybrid_path = notebook_dir / "run_hybrid_map_experiment.ipynb"
    notebook_paths = (integration_path, vidit_path, hybrid_path)
    present = [p for p in notebook_paths if p.is_file()]
    missing = [p.name for p in notebook_paths if not p.is_file()]
    if missing:
        print(
            "NOTE skip notebook content checks (not on disk in Colab clone): "
            + ", ".join(missing)
        )
    if not present:
        print("NOTE no local notebooks to validate — skipping orchestration checks")
        return

    for nb_path in present:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        source = "\\n".join(
            cell.get("source", "")
            if isinstance(cell.get("source"), str)
            else "".join(cell.get("source", []))
            for cell in nb.get("cells", [])
        )
        if "validate_offline.py" not in source:
            raise RuntimeError(f"{nb_path.name} does not run validate_offline.py")
        if "run_setup_downloads" not in source:
            raise RuntimeError(f"{nb_path.name} does not run run_setup_downloads")
        if ' / "phase2" / ' in source or "/phase2/" in source:
            raise RuntimeError(f"{nb_path.name} still references deprecated Drive path phase2/")


'''


def _copy_python_package(src_root: Path, package: str, dest_root: Path) -> int:
    """Copy all *.py files under src_root/package into dest_root/package."""
    src_pkg = src_root / package
    if not src_pkg.is_dir():
        return 0
    copied = 0
    for py_file in sorted(src_pkg.rglob("*.py")):
        rel = py_file.relative_to(src_pkg)
        dst = dest_root / package / rel
        if _safe_copy2(py_file, dst):
            copied += 1
    return copied


def deploy_quant_scripts(source_dirs: list[Path] | None = None) -> Path:
    """Copy quant scripts into DynamiCtrl repo and mirror to Drive."""
    if source_dirs is None:
        source_dirs = [
            DRIVE_SCRIPTS,
            DRIVE_ROOT / "repo" / "scripts" / "quant",
        ]
    dest = REPO / "scripts" / "quant"
    dest.mkdir(parents=True, exist_ok=True)
    drive_dest = DRIVE_ROOT / "repo" / "scripts" / "quant"
    drive_dest.mkdir(parents=True, exist_ok=True)

    sources = _unique_existing_dirs(source_dirs)
    py_by_name: dict[str, Path] = {}
    for src in sources:
        for py_file in sorted(src.glob("*.py")):
            prev = py_by_name.get(py_file.name)
            if prev is None or py_file.stat().st_mtime >= prev.stat().st_mtime:
                py_by_name[py_file.name] = py_file

    if not py_by_name:
        raise FileNotFoundError(
            "No quant scripts found. Upload scripts/quant to "
            f"{DRIVE_SCRIPTS} before running the notebooks."
        )

    for name, py_file in _tqdm(py_by_name.items(), desc="Deploy quant scripts", unit="file"):
        _safe_copy2(py_file, dest / name)
        _safe_copy2(py_file, drive_dest / name)

    pkg_copied = 0
    for package in ("kinematic_gate",):
        best_src = None
        for src in reversed(sources):
            if (src / package).is_dir():
                best_src = src
                break
        if best_src is not None:
            pkg_copied += _copy_python_package(best_src, package, dest)
            _copy_python_package(best_src, package, drive_dest)

    _twrite(f"Deployed quant scripts to {dest}")
    if pkg_copied:
        _twrite(f"  copied {pkg_copied} kinematic_gate module file(s)")
    patch_validate_offline_for_colab(dest / "validate_offline.py")
    return dest


def write_run_meta(step: str, extra: dict | None = None) -> None:
    """Append run metadata to Drive logs."""
    ensure_drive_layout()
    meta_path = DRIVE_ROOT / "logs" / "run_meta.json"
    entry = {
        "step": step,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "repo_commit": _git_head(REPO),
    }
    if extra:
        entry.update(extra)
    history = []
    if meta_path.exists():
        history = json.loads(meta_path.read_text())
    history.append(entry)
    meta_path.write_text(json.dumps(history, indent=2))


def _git_head(repo: Path) -> str:
    if not (repo / ".git").exists():
        return "unknown"
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo,
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        return "unknown"
