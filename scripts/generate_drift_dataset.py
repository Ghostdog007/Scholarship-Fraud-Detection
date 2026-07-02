"""
Generate a programmatically-constructed "unseen next cycle" dataset for
drift-simulation testing (see docs/API_TESTING_GUIDE.md §9).

Per AGENTS.md Appendix B, synthetic data must be built by sampling and
perturbing real rows — no GAN/CTGAN/TVAE. This script builds an
institute-cluster + income-rounding pattern that is NOT one of the 5
existing synthetic exposure archetypes (IP_CONCENTRATION, MOTHER_NAME_
COLLISION, FEE_INFLATION, AGE_VIOLATION, INCOME_VIOLATION), so it is a
genuine test of whether the model generalizes to a new fraud shape rather
than re-detecting a pattern it was already exposed to.

Pattern: ~600 applications funnelled through 3 institutes (instead of the
normal spread across thousands of institutes) with suspiciously round
declared incomes (multiples of 50,000) — mimics a coordinated cluster of
applications processed through a small number of colluding institutes.

Usage:
    .venv/Scripts/python.exe scripts/generate_drift_dataset.py
Output:
    data/raw/new_cohort_2026.csv  (600 rows, same 136-column raw schema,
    application_id values guaranteed not to collide with the existing 15,000)
"""
import numpy as np
import pandas as pd

RAW_CSV = "data/raw/data_for_ml_model.csv"
OUT_CSV = "data/raw/new_cohort_2026.csv"

N_ROWS       = 600
CLUSTER_INSTITUTES = [900001, 900002, 900003]  # synthetic institute IDs, not in real data
ROUND_INCOMES       = [50000, 100000, 150000, 200000, 250000]
NEW_ID_PREFIX        = "ZZ"   # not a real state code — marks these as simulated
NEW_ID_CYCLE          = "202627"
SEED = 777


def main() -> None:
    rng = np.random.default_rng(SEED)
    base = pd.read_csv(RAW_CSV, low_memory=False)

    sample = base.sample(n=N_ROWS, random_state=SEED).reset_index(drop=True).copy()

    # Institute-cluster perturbation: funnel through 3 institutes, skewed 70/20/10
    sample["c_institution_id"] = rng.choice(
        CLUSTER_INSTITUTES, size=N_ROWS, p=[0.7, 0.2, 0.1]
    )

    # Income-rounding perturbation: suspiciously round declared incomes
    sample["annual_family_income"] = rng.choice(ROUND_INCOMES, size=N_ROWS)

    # New, non-colliding application IDs
    new_ids = [f"{NEW_ID_PREFIX}{NEW_ID_CYCLE}{900001 + i:09d}" for i in range(N_ROWS)]
    sample["application_id"]   = new_ids
    sample["application_id-2"] = new_ids

    assert not set(new_ids) & set(base["application_id"]), "application_id collision with existing data"
    assert list(sample.columns) == list(base.columns), "column schema mismatch"

    sample.to_csv(OUT_CSV, index=False)
    print(f"[generate_drift_dataset] Wrote {len(sample)} rows -> {OUT_CSV}")
    print(f"[generate_drift_dataset] Institutes: {sample['c_institution_id'].value_counts().to_dict()}")
    print(f"[generate_drift_dataset] Incomes: {sample['annual_family_income'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
