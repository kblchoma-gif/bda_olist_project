"""Phase 2 — Exploratory profiling of every raw table.

Writes three artefacts used in the report:
  outputs/tables/schema_overview.csv      (name, dtype, missing%, unique, sample)
  outputs/tables/missing_summary.csv      (per-table missing-value counts)
  outputs/tables/duplicate_summary.csv    (per-table duplicate rows)
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.config import TAB_DIR
from scripts.data_loading import load_all


def schema_overview(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for tname, df in frames.items():
        n = len(df)
        for col in df.columns:
            rows.append({
                "table":       tname,
                "column":      col,
                "dtype":       str(df[col].dtype),
                "n_rows":      n,
                "n_missing":   int(df[col].isna().sum()),
                "pct_missing": round(100 * df[col].isna().mean(), 2),
                "n_unique":    int(df[col].nunique(dropna=True)),
                "sample":      str(df[col].dropna().iloc[0])[:40]
                                if df[col].notna().any() else "—",
            })
    return pd.DataFrame(rows)


def missing_summary(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for tname, df in frames.items():
        total_cells = df.shape[0] * df.shape[1]
        miss = int(df.isna().sum().sum())
        rows.append({
            "table":         tname,
            "rows":          df.shape[0],
            "cols":          df.shape[1],
            "missing_cells": miss,
            "pct_missing":   round(100 * miss / total_cells, 3) if total_cells else 0.0,
        })
    return pd.DataFrame(rows)


def duplicate_summary(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for tname, df in frames.items():
        dup_all = int(df.duplicated().sum())
        rows.append({
            "table":               tname,
            "rows":                df.shape[0],
            "full_row_duplicates": dup_all,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    frames = load_all()

    schema   = schema_overview(frames)
    missing  = missing_summary(frames)
    dup      = duplicate_summary(frames)

    schema.to_csv(TAB_DIR / "schema_overview.csv", index=False)
    missing.to_csv(TAB_DIR / "missing_summary.csv", index=False)
    dup.to_csv(TAB_DIR / "duplicate_summary.csv", index=False)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 200)

    print("\n=== MISSING-VALUE SUMMARY ===")
    print(missing.to_string(index=False))

    print("\n=== DUPLICATE SUMMARY ===")
    print(dup.to_string(index=False))

    print("\n=== SCHEMA OVERVIEW (first 60 rows) ===")
    print(schema.head(60).to_string(index=False))

    print(f"\n[INFO] Tables written to {TAB_DIR}")