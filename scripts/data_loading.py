"""Phase 2 — Data acquisition for the Olist e-commerce dataset.

Two acquisition modes are supported:
  A) Automated download via kagglehub (requires internet + Kaggle credentials)
  B) Manual load from ./data/raw (dataset pre-downloaded by the user)

Every load is logged to outputs/tables/acquisition_manifest.csv so the
report can show provenance.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.config import RAW_DIR, RAW_FILES, TAB_DIR, KAGGLE_SLUG


def download_from_kaggle() -> Path:
    """Download the dataset via kagglehub. Returns the local dataset path."""
    import kagglehub
    print(f"[INFO] Downloading '{KAGGLE_SLUG}' via kagglehub ...")
    path = Path(kagglehub.dataset_download(KAGGLE_SLUG))
    print(f"[INFO] Kagglehub stored dataset at: {path}")
    return path


def locate_raw_files() -> dict[str, Path]:
    """Return {logical_name: full_path} for every expected CSV.

    Searches ./data/raw first, then falls back to the kagglehub cache.
    Raises FileNotFoundError listing missing files if any are absent.
    """
    resolved: dict[str, Path] = {}
    missing: list[str] = []

    for key, fname in RAW_FILES.items():
        local = RAW_DIR / fname
        if local.exists():
            resolved[key] = local
        else:
            missing.append(fname)

    if missing:
        # Try kagglehub cache as a fallback
        try:
            hub_path = download_from_kaggle()
            for key, fname in RAW_FILES.items():
                if key in resolved:
                    continue
                candidate = hub_path / fname
                if candidate.exists():
                    resolved[key] = candidate
                else:
                    # kagglehub sometimes nests; search one level deep
                    hits = list(hub_path.rglob(fname))
                    if hits:
                        resolved[key] = hits[0]

            still_missing = [k for k in RAW_FILES if k not in resolved]
            if still_missing:
                raise FileNotFoundError(
                    "Missing files after download attempt: "
                    + ", ".join(RAW_FILES[k] for k in still_missing)
                )
        except Exception as e:
            raise FileNotFoundError(
                f"Could not locate raw files in {RAW_DIR} and kagglehub "
                f"download failed: {e}\nMissing: {missing}"
            )

    return resolved


def load_all() -> dict[str, pd.DataFrame]:
    """Load every CSV into a DataFrame; record provenance manifest."""
    paths = locate_raw_files()
    frames: dict[str, pd.DataFrame] = {}
    manifest_rows = []

    for key, path in paths.items():
        df = pd.read_csv(path, encoding="utf-8", low_memory=False)
        frames[key] = df
        manifest_rows.append({
            "logical_name": key,
            "file_name":    path.name,
            "source_path":  str(path),
            "rows":         df.shape[0],
            "cols":         df.shape[1],
            "size_kb":      round(path.stat().st_size / 1024, 1),
        })
        print(f"[OK] {key:<12} rows={df.shape[0]:>8,}  cols={df.shape[1]:>3}  "
              f"file={path.name}")

    pd.DataFrame(manifest_rows).to_csv(
        TAB_DIR / "acquisition_manifest.csv", index=False
    )
    print(f"\n[INFO] Manifest written to {TAB_DIR / 'acquisition_manifest.csv'}")
    return frames


if __name__ == "__main__":
    frames = load_all()
    print("\n=== PER-TABLE HEAD (first 3 rows) ===")
    for name, df in frames.items():
        print(f"\n--- {name} ---")
        print(df.head(3).to_string())