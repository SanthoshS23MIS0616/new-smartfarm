"""
merge_datasets.py
=================
Merges the existing 13-crop recommendation dataset with the Kaggle 22-crop
dataset (https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)
to produce an expanded 22-crop training dataset.

Usage
-----
1. Download Crop_recommendation.csv from Kaggle and place at:
       data/kaggle_crop_recommendation.csv

2. Run:
       python scripts/merge_datasets.py

Output
------
   data/realistic_v2/crop_recommendation_realistic.csv  (updated, 22 crops)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
KAGGLE_CSV = ROOT / "data" / "kaggle_crop_recommendation.csv"
EXISTING_CSV = ROOT / "data" / "realistic_v2" / "crop_recommendation_realistic.csv"
OUTPUT_CSV = ROOT / "data" / "realistic_v2" / "crop_recommendation_realistic.csv"
BACKUP_CSV = ROOT / "data" / "realistic_v2" / "crop_recommendation_realistic_backup.csv"

# ── Column mapping: Kaggle → our schema ──────────────────────────────────────
KAGGLE_COL_MAP = {
    "N": "nitrogen",
    "P": "phosphorous",
    "K": "potassium",
    "temperature": "temperature_c",
    "humidity": "humidity",
    "ph": "ph",
    "rainfall": "rainfall_mm",
    "label": "crop",
}

# Kaggle crop name → our canonical name
KAGGLE_CROP_NAME_MAP = {
    "rice": "Rice",
    "maize": "Maize",
    "jute": "Jute",
    "cotton": "Cotton",
    "coconut": "Coconut",
    "papaya": "Papaya",
    "orange": "Orange",
    "apple": "Apple",
    "muskmelon": "Muskmelon",
    "watermelon": "Watermelon",
    "grapes": "Grapes",
    "mango": "Mango",
    "banana": "Banana",
    "pomegranate": "Pomegranate",
    "lentil": "Lentil",
    "blackgram": "Blackgram",
    "mungbean": "Mungbean",
    "mothbeans": "Mothbeans",
    "pigeonpeas": "Pigeonpeas",
    "kidneybeans": "Kidneybeans",
    "chickpea": "Chickpea",
    "coffee": "Coffee",
}

# Season assignment for each crop (new crops need this added)
CROP_SEASON_MAP = {
    "Rice": "Kharif",
    "Maize": "Kharif",
    "Jute": "Kharif",
    "Cotton": "Kharif",
    "Blackgram": "Kharif",
    "Pigeonpeas": "Kharif",
    "Kidneybeans": "Kharif",
    "Mothbeans": "Kharif",
    "Mungbean": "Kharif",
    "Lentil": "Rabi",
    "Chickpea": "Rabi",
    "Watermelon": "Summer",
    "Muskmelon": "Summer",
    "Coconut": "Whole Year",
    "Papaya": "Whole Year",
    "Banana": "Whole Year",
    "Mango": "Whole Year",
    "Orange": "Whole Year",
    "Grapes": "Whole Year",
    "Apple": "Whole Year",
    "Coffee": "Kharif",
    "Pomegranate": "Whole Year",
}

# Representative states per crop
CROP_STATE_MAP = {
    "Rice": ["West Bengal", "Uttar Pradesh", "Andhra Pradesh", "Tamil Nadu", "Punjab"],
    "Maize": ["Karnataka", "Andhra Pradesh", "Rajasthan", "Maharashtra", "Bihar"],
    "Jute": ["West Bengal", "Bihar", "Assam", "Odisha"],
    "Cotton": ["Gujarat", "Maharashtra", "Telangana", "Andhra Pradesh"],
    "Blackgram": ["Andhra Pradesh", "Maharashtra", "Tamil Nadu", "Madhya Pradesh"],
    "Lentil": ["Madhya Pradesh", "Uttar Pradesh", "Bihar", "West Bengal"],
    "Chickpea": ["Madhya Pradesh", "Rajasthan", "Maharashtra", "Uttar Pradesh"],
    "Kidneybeans": ["Himachal Pradesh", "Uttarakhand", "Jammu and Kashmir"],
    "Mothbeans": ["Rajasthan", "Gujarat", "Haryana"],
    "Mungbean": ["Rajasthan", "Maharashtra", "Andhra Pradesh", "Tamil Nadu"],
    "Pigeonpeas": ["Maharashtra", "Uttar Pradesh", "Karnataka", "Andhra Pradesh"],
    "Coconut": ["Kerala", "Tamil Nadu", "Karnataka", "Andhra Pradesh"],
    "Papaya": ["Andhra Pradesh", "Gujarat", "Maharashtra", "Tamil Nadu"],
    "Banana": ["Tamil Nadu", "Andhra Pradesh", "Maharashtra", "Gujarat"],
    "Mango": ["Uttar Pradesh", "Andhra Pradesh", "Karnataka", "Maharashtra"],
    "Orange": ["Maharashtra", "Madhya Pradesh", "Nagaland"],
    "Grapes": ["Maharashtra", "Karnataka", "Tamil Nadu"],
    "Apple": ["Himachal Pradesh", "Jammu and Kashmir", "Uttarakhand"],
    "Coffee": ["Karnataka", "Kerala", "Tamil Nadu"],
    "Watermelon": ["Uttar Pradesh", "Karnataka", "Andhra Pradesh"],
    "Muskmelon": ["Uttar Pradesh", "Punjab", "Haryana"],
    "Pomegranate": ["Maharashtra", "Karnataka", "Gujarat", "Andhra Pradesh"],
}

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)


def _add_moisture(df: pd.DataFrame) -> pd.DataFrame:
    """Synthesise realistic moisture from humidity + rainfall."""
    if "moisture" not in df.columns:
        base = df["humidity"] * 0.28 + df["rainfall_mm"] * 0.04
        noise = rng.normal(0, 2.5, size=len(df))
        df["moisture"] = np.clip(base + noise, 10.0, 50.0).round(2)
    return df


def _add_state_season(df: pd.DataFrame) -> pd.DataFrame:
    """Add state_name and season columns if missing."""
    if "state_name" not in df.columns:
        states = df["crop"].map(CROP_STATE_MAP)
        df["state_name"] = [
            rng.choice(s_list) if isinstance(s_list, list) else "Maharashtra"
            for s_list in states
        ]
    if "season" not in df.columns:
        df["season"] = df["crop"].map(CROP_SEASON_MAP).fillna("Kharif")
    return df


def load_kaggle(path: Path) -> pd.DataFrame:
    print(f"Loading Kaggle dataset from: {path}")
    df = pd.read_csv(path)
    df = df.rename(columns=KAGGLE_COL_MAP)
    df["crop"] = df["crop"].str.strip().str.lower().map(KAGGLE_CROP_NAME_MAP)
    df = df.dropna(subset=["crop"])
    df = _add_moisture(df)
    df = _add_state_season(df)
    print(f"  Kaggle: {len(df)} rows, crops: {sorted(df['crop'].unique())}")
    return df


def load_existing(path: Path) -> pd.DataFrame:
    print(f"Loading existing dataset from: {path}")
    df = pd.read_csv(path)
    print(f"  Existing: {len(df)} rows, crops: {sorted(df['crop'].unique())}")
    return df


def merge_and_balance(existing: pd.DataFrame, kaggle: pd.DataFrame, target_per_crop: int = 1000) -> pd.DataFrame:
    """
    Merge datasets, deduplicate, and balance classes.
    Existing data is prioritised; Kaggle fills gaps for new/under-represented crops.
    """
    SHARED_COLS = ["crop", "state_name", "season", "nitrogen", "phosphorous",
                   "potassium", "temperature_c", "humidity", "ph", "rainfall_mm", "moisture"]

    # Align columns
    for col in SHARED_COLS:
        if col not in existing.columns:
            existing[col] = np.nan
        if col not in kaggle.columns:
            kaggle[col] = np.nan

    combined = pd.concat([existing[SHARED_COLS], kaggle[SHARED_COLS]], ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates()
    print(f"  Combined: {before} rows → {len(combined)} after dedup")

    # Balance per crop
    parts = []
    for crop, group in combined.groupby("crop"):
        if len(group) >= target_per_crop:
            parts.append(group.sample(n=target_per_crop, random_state=RANDOM_STATE))
        else:
            # Oversample with slight noise to pad small classes
            needed = target_per_crop - len(group)
            sample = group.sample(n=needed, replace=True, random_state=RANDOM_STATE)
            num_cols = ["nitrogen", "phosphorous", "potassium", "temperature_c",
                        "humidity", "ph", "rainfall_mm", "moisture"]
            for col in num_cols:
                noise = rng.normal(0, sample[col].std() * 0.05, size=len(sample))
                sample[col] = (sample[col] + noise).clip(lower=0).round(2)
            parts.append(pd.concat([group, sample], ignore_index=True))
        print(f"    {crop}: {len(parts[-1])} rows")

    result = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"\nFinal merged dataset: {len(result)} rows, {result['crop'].nunique()} crops")
    return result


def main() -> None:
    if not KAGGLE_CSV.exists():
        print(f"\nERROR: Kaggle CSV not found at: {KAGGLE_CSV}")
        print("Please download it from:")
        print("  https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset")
        print(f"and save it as: {KAGGLE_CSV}")
        sys.exit(1)

    # Backup existing dataset
    if EXISTING_CSV.exists():
        import shutil
        shutil.copy(EXISTING_CSV, BACKUP_CSV)
        print(f"Backed up existing dataset to: {BACKUP_CSV}")

    existing = load_existing(EXISTING_CSV)
    kaggle = load_kaggle(KAGGLE_CSV)

    merged = merge_and_balance(existing, kaggle, target_per_crop=1000)

    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved merged dataset to: {OUTPUT_CSV}")
    print("\nCrop counts in merged dataset:")
    print(merged["crop"].value_counts().to_string())
    print("\nDone! You can now retrain the models with: python scripts/run_training.py")


if __name__ == "__main__":
    main()
