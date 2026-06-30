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
# Split: 50 clean-archetype + 100 perturbed variants per archetype.
# This covers a wider band of fraud severity rather than a single spike at the
# 97th-percentile value, reducing the risk that the model only detects obvious fraud.
N_CLEAN   = 50
N_PERTURB = 100


def _get_col_idx(features: list[str], name: str) -> int | None:
    try:
        return features.index(name)
    except ValueError:
        return None


def _add_context_noise(rows: np.ndarray, target_cols: set[int], noise_frac: float = 0.2) -> np.ndarray:
    """
    Add ±noise_frac random noise to a random 25% of non-target columns per row.
    Keeps fraud signal columns intact while widening the context geometry so the
    model sees fraud embedded in varied (not identical) background features.
    """
    n_rows, n_cols = rows.shape
    all_cols = set(range(n_cols))
    context_cols = list(all_cols - target_cols)
    n_to_perturb = max(1, int(len(context_cols) * 0.25))

    rows = rows.copy()
    for i in range(n_rows):
        cols = RNG.choice(context_cols, n_to_perturb, replace=False)
        noise = RNG.uniform(-noise_frac, noise_frac, size=n_to_perturb)
        rows[i, cols] = np.clip(rows[i, cols] + noise, 0.0, 1.0)
    return rows


def _archetype_ip_concentration(feat: np.ndarray, features: list[str]) -> np.ndarray:
    """ip_application_count and degree_shares_ip pushed to extreme values."""
    idx_ip     = _get_col_idx(features, "ip_application_count")
    idx_ipm    = _get_col_idx(features, "ip_to_mobile_ratio")
    idx_deg_ip = _get_col_idx(features, "degree_shares_ip")
    target_cols = {i for i in [idx_ip, idx_ipm, idx_deg_ip] if i is not None}

    rows  = feat[RNG.choice(len(feat), N_PER_ARCHETYPE, replace=True)].copy()
    p85   = np.percentile(feat[:, idx_ip], 85) if idx_ip is not None else 1.0
    p97   = np.percentile(feat[:, idx_ip], 97) if idx_ip is not None else 1.0
    # Graduated signal: N_CLEAN at peak, N_PERTURB spanning 85th–97th percentile
    strengths = np.concatenate([
        np.full(N_CLEAN, p97),
        np.linspace(p85, p97, N_PERTURB),
    ])
    RNG.shuffle(strengths)

    if idx_ip is not None:
        rows[:, idx_ip] = np.clip(strengths + RNG.uniform(0, 0.02, N_PER_ARCHETYPE), 0, 1)
    if idx_ipm is not None:
        rows[:, idx_ipm] = np.clip(rows[:, idx_ipm] * RNG.uniform(1.2, 3.0, N_PER_ARCHETYPE), 0, 1)
    if idx_deg_ip is not None:
        rows[:, idx_deg_ip] = np.clip(
            np.percentile(feat[:, idx_deg_ip], 90) + RNG.uniform(0, 0.08, N_PER_ARCHETYPE), 0, 1)

    return _add_context_noise(rows, target_cols)


def _archetype_mother_name_collision(feat: np.ndarray, features: list[str]) -> np.ndarray:
    """is_father_name_eq_mother=1 with varying name_similarity_score strength."""
    idx_fm   = _get_col_idx(features, "is_father_name_eq_mother")
    idx_sim  = _get_col_idx(features, "name_similarity_score")
    idx_degm = _get_col_idx(features, "degree_shares_mother_name")
    idx_degf = _get_col_idx(features, "degree_shares_father_name")
    target_cols = {i for i in [idx_fm, idx_sim, idx_degm, idx_degf] if i is not None}

    rows = feat[RNG.choice(len(feat), N_PER_ARCHETYPE, replace=True)].copy()

    if idx_fm is not None:
        rows[:, idx_fm] = 1.0
    if idx_sim is not None:
        p80 = np.percentile(feat[:, idx_sim], 80)
        p97 = np.percentile(feat[:, idx_sim], 97)
        strengths = np.concatenate([np.full(N_CLEAN, p97), np.linspace(p80, p97, N_PERTURB)])
        RNG.shuffle(strengths)
        rows[:, idx_sim] = np.clip(strengths + RNG.uniform(0, 0.05, N_PER_ARCHETYPE), 0, 1)
    if idx_degm is not None:
        rows[:, idx_degm] = np.clip(
            np.percentile(feat[:, idx_degm], 88) + RNG.uniform(0, 0.08, N_PER_ARCHETYPE), 0, 1)
    if idx_degf is not None:
        rows[:, idx_degf] = np.clip(
            np.percentile(feat[:, idx_degf], 88) + RNG.uniform(0, 0.08, N_PER_ARCHETYPE), 0, 1)

    return _add_context_noise(rows, target_cols)


def _archetype_fee_inflation(feat: np.ndarray, features: list[str]) -> np.ndarray:
    """fee_income_ratio from 85th to 97th percentile, graduated."""
    idx_fir = _get_col_idx(features, "fee_income_ratio")
    idx_inc = _get_col_idx(features, "annual_family_income")
    target_cols = {i for i in [idx_fir, idx_inc] if i is not None}

    rows  = feat[RNG.choice(len(feat), N_PER_ARCHETYPE, replace=True)].copy()
    p85   = np.percentile(feat[:, idx_fir], 85) if idx_fir is not None else 0.8
    p97   = np.percentile(feat[:, idx_fir], 97) if idx_fir is not None else 1.0
    strengths = np.concatenate([np.full(N_CLEAN, p97), np.linspace(p85, p97, N_PERTURB)])
    RNG.shuffle(strengths)

    if idx_fir is not None:
        rows[:, idx_fir] = np.clip(strengths + RNG.uniform(0, 0.02, N_PER_ARCHETYPE), 0, 1)
    if idx_inc is not None:
        median_inc = np.percentile(feat[:, idx_inc], 50)
        rows[:, idx_inc] = np.clip(median_inc + RNG.uniform(-0.08, 0.08, N_PER_ARCHETYPE), 0.05, 1)

    return _add_context_noise(rows, target_cols)


def _archetype_age_violation(feat: np.ndarray, features: list[str]) -> np.ndarray:
    """age_at_registration from 88th to 97th percentile, graduated."""
    idx_age = _get_col_idx(features, "age_at_registration")
    idx_ppm = _get_col_idx(features, "pre_post_matric")
    target_cols = {i for i in [idx_age, idx_ppm] if i is not None}

    rows  = feat[RNG.choice(len(feat), N_PER_ARCHETYPE, replace=True)].copy()
    p88   = np.percentile(feat[:, idx_age], 88) if idx_age is not None else 0.8
    p97   = np.percentile(feat[:, idx_age], 97) if idx_age is not None else 1.0
    strengths = np.concatenate([np.full(N_CLEAN, p97), np.linspace(p88, p97, N_PERTURB)])
    RNG.shuffle(strengths)

    if idx_age is not None:
        rows[:, idx_age] = np.clip(strengths + RNG.uniform(0, 0.02, N_PER_ARCHETYPE), 0, 1)
    if idx_ppm is not None:
        rows[:, idx_ppm] = np.percentile(feat[:, idx_ppm], 10)

    return _add_context_noise(rows, target_cols)


def _archetype_income_violation(feat: np.ndarray, features: list[str]) -> np.ndarray:
    """annual_family_income from 3rd to 12th percentile (bottom band), graduated."""
    idx_inc = _get_col_idx(features, "annual_family_income")
    idx_fir = _get_col_idx(features, "fee_income_ratio")
    target_cols = {i for i in [idx_inc, idx_fir] if i is not None}

    rows  = feat[RNG.choice(len(feat), N_PER_ARCHETYPE, replace=True)].copy()
    p3    = np.percentile(feat[:, idx_inc], 3)  if idx_inc is not None else 0.0
    p12   = np.percentile(feat[:, idx_inc], 12) if idx_inc is not None else 0.05
    # Graduated from mildly low (p12) down to extreme low (p3)
    strengths = np.concatenate([np.full(N_CLEAN, p3), np.linspace(p12, p3, N_PERTURB)])
    RNG.shuffle(strengths)

    if idx_inc is not None:
        rows[:, idx_inc] = np.clip(strengths + RNG.uniform(0, 0.005, N_PER_ARCHETYPE), 0, 1)
    if idx_fir is not None:
        rows[:, idx_fir] = np.clip(
            np.percentile(feat[:, idx_fir], 90) + RNG.uniform(0, 0.08, N_PER_ARCHETYPE), 0, 1)

    return _add_context_noise(rows, target_cols)


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
