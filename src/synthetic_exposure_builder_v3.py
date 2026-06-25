"""
synthetic_exposure_builder_v3.py

Programmatically constructs 750 synthetic anomaly examples (5 archetypes x 150).
All examples are 68-dim, aligned to engineered_features_v3.csv column order.
Writes: data/processed/synthetic_exposure_set_v3.pt  shape=(750, 68)

Construction method: sample real rows, perturb target fields.
Never uses a tabular GAN -- see AGENTS.md Appendix B.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config_v3 import N_FEATURES, RANDOM_SEED

RNG = np.random.default_rng(RANDOM_SEED)

FINAL_CSV   = Path("data/processed/engineered_features_v3.csv")
SCHEMA_JSON = Path("data/processed/v3_feature_schema.json")
OUT_PT      = Path("data/processed/synthetic_exposure_set_v3.pt")

N_PER_ARCHETYPE = 150


def _get_col_idx(features: list[str], name: str) -> int | None:
    try:
        return features.index(name)
    except ValueError:
        return None


def _archetype_ip_concentration(feat: np.ndarray, features: list[str]) -> np.ndarray:
    """Sample 15 real rows, set ip_application_count to 97th-percentile value."""
    idx_ip = _get_col_idx(features, "ip_application_count")
    idx_ipm = _get_col_idx(features, "ip_to_mobile_ratio")
    idx_deg_ip = _get_col_idx(features, "degree_shares_ip")

    rows = feat[RNG.choice(len(feat), N_PER_ARCHETYPE, replace=True)].copy()
    high_ip = np.percentile(feat[:, idx_ip], 97) if idx_ip is not None else 1.0

    if idx_ip is not None:
        rows[:, idx_ip] = np.clip(
            high_ip + RNG.uniform(0, 0.05, N_PER_ARCHETYPE), 0, 1
        )
    if idx_ipm is not None:
        rows[:, idx_ipm] = np.clip(
            rows[:, idx_ipm] * RNG.uniform(1.5, 3.0, N_PER_ARCHETYPE), 0, 1
        )
    if idx_deg_ip is not None:
        rows[:, idx_deg_ip] = np.clip(
            np.percentile(feat[:, idx_deg_ip], 95) + RNG.uniform(0, 0.05, N_PER_ARCHETYPE),
            0, 1,
        )
    return rows


def _archetype_mother_name_collision(feat: np.ndarray, features: list[str]) -> np.ndarray:
    """Set is_father_name_eq_mother=1, name_similarity_score high."""
    idx_fm   = _get_col_idx(features, "is_father_name_eq_mother")
    idx_sim  = _get_col_idx(features, "name_similarity_score")
    idx_degm = _get_col_idx(features, "degree_shares_mother_name")
    idx_degf = _get_col_idx(features, "degree_shares_father_name")

    rows = feat[RNG.choice(len(feat), N_PER_ARCHETYPE, replace=True)].copy()

    if idx_fm is not None:
        rows[:, idx_fm] = 1.0
    if idx_sim is not None:
        rows[:, idx_sim] = np.clip(
            np.percentile(feat[:, idx_sim], 90) + RNG.uniform(0, 0.1, N_PER_ARCHETYPE),
            0, 1,
        )
    if idx_degm is not None:
        rows[:, idx_degm] = np.clip(
            np.percentile(feat[:, idx_degm], 90) + RNG.uniform(0, 0.05, N_PER_ARCHETYPE),
            0, 1,
        )
    if idx_degf is not None:
        rows[:, idx_degf] = np.clip(
            np.percentile(feat[:, idx_degf], 90) + RNG.uniform(0, 0.05, N_PER_ARCHETYPE),
            0, 1,
        )
    return rows


def _archetype_fee_inflation(feat: np.ndarray, features: list[str]) -> np.ndarray:
    """Set fee_income_ratio above 97th percentile while keeping income moderate."""
    idx_fir = _get_col_idx(features, "fee_income_ratio")
    idx_inc = _get_col_idx(features, "annual_family_income")

    rows = feat[RNG.choice(len(feat), N_PER_ARCHETYPE, replace=True)].copy()

    high_fir = np.percentile(feat[:, idx_fir], 97) if idx_fir is not None else 1.0

    if idx_fir is not None:
        rows[:, idx_fir] = np.clip(
            high_fir + RNG.uniform(0, 0.05, N_PER_ARCHETYPE), 0, 1
        )
    if idx_inc is not None:
        median_inc = np.percentile(feat[:, idx_inc], 50)
        rows[:, idx_inc] = np.clip(
            median_inc + RNG.uniform(-0.05, 0.05, N_PER_ARCHETYPE), 0.1, 1
        )
    return rows


def _archetype_age_violation(feat: np.ndarray, features: list[str]) -> np.ndarray:
    """Set age_at_registration above 97th percentile."""
    idx_age  = _get_col_idx(features, "age_at_registration")
    idx_ppm  = _get_col_idx(features, "pre_post_matric")

    rows = feat[RNG.choice(len(feat), N_PER_ARCHETYPE, replace=True)].copy()

    high_age = np.percentile(feat[:, idx_age], 97) if idx_age is not None else 1.0

    if idx_age is not None:
        rows[:, idx_age] = np.clip(
            high_age + RNG.uniform(0, 0.05, N_PER_ARCHETYPE), 0, 1
        )
    if idx_ppm is not None:
        rows[:, idx_ppm] = np.percentile(feat[:, idx_ppm], 10)
    return rows


def _archetype_income_violation(feat: np.ndarray, features: list[str]) -> np.ndarray:
    """Set annual_family_income to near-zero (bottom 3rd percentile)."""
    idx_inc = _get_col_idx(features, "annual_family_income")
    idx_fir = _get_col_idx(features, "fee_income_ratio")

    rows = feat[RNG.choice(len(feat), N_PER_ARCHETYPE, replace=True)].copy()

    low_inc = np.percentile(feat[:, idx_inc], 3) if idx_inc is not None else 0.0

    if idx_inc is not None:
        rows[:, idx_inc] = np.clip(
            low_inc + RNG.uniform(0, 0.01, N_PER_ARCHETYPE), 0, 0.02
        )
    if idx_fir is not None:
        rows[:, idx_fir] = np.clip(
            np.percentile(feat[:, idx_fir], 95) + RNG.uniform(0, 0.05, N_PER_ARCHETYPE),
            0, 1,
        )
    return rows


ARCHETYPES = [
    ("IP_CONCENTRATION",    _archetype_ip_concentration),
    ("MOTHER_NAME_COLLISION", _archetype_mother_name_collision),
    ("FEE_INFLATION",       _archetype_fee_inflation),
    ("AGE_VIOLATION",       _archetype_age_violation),
    ("INCOME_VIOLATION",    _archetype_income_violation),
]


def build_exposure_set() -> None:
    print("[synthetic_exposure] build_exposure_set() starting ...")

    schema   = json.loads(SCHEMA_JSON.read_text())
    features = schema["features"]

    df = pd.read_csv(FINAL_CSV)
    feat_cols = [c for c in df.columns if c != "application_id"]
    feat = df[feat_cols].values.astype(np.float32)

    if feat.shape[1] != N_FEATURES:
        raise ValueError(f"Expected {N_FEATURES} features, got {feat.shape[1]}")

    if feat_cols != features:
        raise ValueError("Feature CSV column order does not match schema. Regenerate feature engine outputs.")

    chunks = []
    for name, fn in ARCHETYPES:
        synth = fn(feat, feat_cols)
        chunks.append(synth)
        print(f"[synthetic_exposure]   {name}: {synth.shape[0]} examples")

    exposure = np.vstack(chunks).astype(np.float32)
    assert exposure.shape == (750, N_FEATURES), f"Shape mismatch: {exposure.shape}"

    OUT_PT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.tensor(exposure), OUT_PT)
    print(f"[synthetic_exposure] Saved exposure set {exposure.shape} -> {OUT_PT}")


if __name__ == "__main__":
    build_exposure_set()
