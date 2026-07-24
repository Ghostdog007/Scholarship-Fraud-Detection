"""
Generate a second, independent 50,000-application stress-test dataset —
"stress_testing_2" — for cross-validating findings from stress_testing_1
(e.g. the LAMBDA_EDGE ablation in scripts/ablation_lambda_edge.py) against
a different random draw, not just a re-read of the same rows.

Identical methodology to generate_stress_test_dataset.py (same archetype
proportions, same sampling-with-replacement + perturbation approach per
hard stop 7 -- never GAN/CTGAN/TVAE) -- only SEED and output paths differ,
so this is a genuine second population, not a duplicate.

Writes:
  data/uploads/stress_testing_2.csv               50,000 rows, 136-col raw schema
  data/uploads/stress_testing_2_ground_truth.csv   application_id, is_fraud,
                                                    fraud_type, ring_id

Run:
  .venv/Scripts/python.exe scripts/generate_stress_test_dataset_2.py
"""
import numpy as np
import pandas as pd

RAW_CSV = "data/raw/data_for_ml_model.csv"
OUT_CSV = "data/uploads/stress_testing_2.csv"
GT_CSV  = "data/uploads/stress_testing_2_ground_truth.csv"

SEED     = 20260723   # different from stress_testing_1's 20260721
N_TOTAL  = 50_000

NEW_PREFIX = "ST2"       # distinct prefix -- no id collision with stress_testing_1 or real data
NEW_CYCLE  = "202729"

# Same archetype proportions as stress_testing_1, for a like-for-like comparison
N_FEE_INFLATION    = 1_000
N_INCOME_VIOLATION = 1_000
N_AGE_VIOLATION    = 1_000
N_NAME_COLLISION   = 1_000

N_IP_POOL      = 1_300
N_MOBILE_POOL  = 1_300
N_PINCODE_POOL = 900

RING_SIZE_RANGE = (6, 40)  # same as config_v3.TOPO_CLUSTER_SIZE_RANGE


def _make_rings(idx_pool: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    pool = idx_pool.copy()
    rng.shuffle(pool)
    rings, i, n = [], 0, len(pool)
    while i < n:
        remaining = n - i
        size = min(int(rng.integers(RING_SIZE_RANGE[0], RING_SIZE_RANGE[1] + 1)), remaining)
        if size < RING_SIZE_RANGE[0] and rings:
            rings[-1] = np.concatenate([rings[-1], pool[i:i + remaining]])
            break
        rings.append(pool[i:i + size])
        i += size
    return rings


def main() -> None:
    rng  = np.random.default_rng(SEED)
    base = pd.read_csv(RAW_CSV, low_memory=False)
    print(f"[stress_test_gen_2] Base population: {len(base)} real applications")

    sample_pos = rng.integers(0, len(base), size=N_TOTAL)
    df = base.iloc[sample_pos].reset_index(drop=True).copy()

    new_ids = [f"{NEW_PREFIX}{NEW_CYCLE}{i:09d}" for i in range(N_TOTAL)]
    df["application_id"]   = new_ids
    df["application_id-2"] = new_ids
    assert not set(new_ids) & set(base["application_id"]), "application_id collision with primary population"
    assert list(df.columns) == list(base.columns), "column schema mismatch"

    gt = pd.DataFrame({
        "application_id": new_ids,
        "is_fraud":       False,
        "fraud_type":     "NONE",
        "ring_id":        "",
    })

    n_fraud_needed = (N_FEE_INFLATION + N_INCOME_VIOLATION + N_AGE_VIOLATION +
                      N_NAME_COLLISION + N_IP_POOL + N_MOBILE_POOL + N_PINCODE_POOL)
    assert n_fraud_needed < N_TOTAL, "fraud allocation exceeds total rows"
    perm = rng.permutation(N_TOTAL)
    cursor = 0

    def take(n: int) -> np.ndarray:
        nonlocal cursor
        sl = perm[cursor:cursor + n]
        cursor += n
        return sl

    idx_fee     = take(N_FEE_INFLATION)
    idx_income  = take(N_INCOME_VIOLATION)
    idx_age     = take(N_AGE_VIOLATION)
    idx_name    = take(N_NAME_COLLISION)
    idx_ip_pool = take(N_IP_POOL)
    idx_mob_pool = take(N_MOBILE_POOL)
    idx_pin_pool = take(N_PINCODE_POOL)

    # ── FEE_INFLATION ────────────────────────────────────────────────────────
    n = len(idx_fee)
    df.loc[idx_fee, "tution_fee"]    = (df.loc[idx_fee, "tution_fee"].clip(lower=5_000)  * rng.uniform(6, 15, n)).astype(int)
    df.loc[idx_fee, "admission_fee"] = (df.loc[idx_fee, "admission_fee"].clip(lower=500) * rng.uniform(3, 8, n)).astype(int)
    df.loc[idx_fee, "misc_fee"]      = (df.loc[idx_fee, "misc_fee"].clip(lower=200)      * rng.uniform(3, 8, n)).astype(int)
    gt.loc[idx_fee, ["is_fraud", "fraud_type"]] = [True, "FEE_INFLATION"]

    # ── INCOME_VIOLATION ─────────────────────────────────────────────────────
    n = len(idx_income)
    df.loc[idx_income, "annual_family_income"] = rng.integers(3_000, 15_000, size=n)
    df.loc[idx_income, "tution_fee"] = df.loc[idx_income, "tution_fee"].clip(lower=3_000).astype(int)
    gt.loc[idx_income, ["is_fraud", "fraud_type"]] = [True, "INCOME_VIOLATION"]

    # ── AGE_VIOLATION ────────────────────────────────────────────────────────
    n = len(idx_age)
    reg = pd.to_datetime(df.loc[idx_age, "registered_date"], errors="coerce")
    reg = reg.fillna(pd.Timestamp("2025-06-01"))
    new_dob = reg - pd.to_timedelta(rng.integers(45 * 365, 65 * 365, size=n), unit="D")
    df.loc[idx_age, "date_of_birth"] = new_dob.dt.strftime("%Y-%m-%d")
    gt.loc[idx_age, ["is_fraud", "fraud_type"]] = [True, "AGE_VIOLATION"]

    # ── MOTHER_NAME_COLLISION ────────────────────────────────────────────────
    df.loc[idx_name, "mother_name"] = df.loc[idx_name, "father_name"]
    gt.loc[idx_name, ["is_fraud", "fraud_type"]] = [True, "MOTHER_NAME_COLLISION"]

    # ── Relational rings ─────────────────────────────────────────────────────
    def apply_rings(idx_pool: np.ndarray, col: str, value_fn, fraud_type: str, ring_prefix: str) -> int:
        rings = _make_rings(idx_pool, rng)
        for r_i, ring in enumerate(rings):
            df.loc[ring, col] = value_fn(r_i)
            ring_id = f"{ring_prefix}_{r_i:04d}"
            gt.loc[ring, "is_fraud"]   = True
            gt.loc[ring, "fraud_type"] = fraud_type
            gt.loc[ring, "ring_id"]    = ring_id
        return len(rings)

    n_ip_rings  = apply_rings(idx_ip_pool,  "ip_address",
                               lambda i: f"10.{210 + i % 40}.{i % 256}.{(i * 11) % 256}",
                               "IP_CLUSTER", "ip_ring")
    n_mob_rings = apply_rings(idx_mob_pool, "mobile_no",
                               lambda i: 8_000_000_000 + i,
                               "MOBILE_CLUSTER", "mobile_ring")
    n_pin_rings = apply_rings(idx_pin_pool, "permanent_pincode",
                               lambda i: 800_000 + i,
                               "PINCODE_CLUSTER", "pincode_ring")

    df.to_csv(OUT_CSV, index=False)
    gt.to_csv(GT_CSV, index=False)

    n_fraud = int(gt["is_fraud"].sum())
    print(f"[stress_test_gen_2] Wrote {len(df)} rows -> {OUT_CSV}")
    print(f"[stress_test_gen_2] Wrote ground truth -> {GT_CSV}")
    print(f"[stress_test_gen_2] Fraud rows: {n_fraud} ({100*n_fraud/N_TOTAL:.1f}%) | Valid rows: {N_TOTAL - n_fraud}")
    print(f"[stress_test_gen_2] By type:\n{gt['fraud_type'].value_counts().to_string()}")
    print(f"[stress_test_gen_2] Rings: {n_ip_rings} IP_CLUSTER, {n_mob_rings} MOBILE_CLUSTER, {n_pin_rings} PINCODE_CLUSTER")


if __name__ == "__main__":
    main()
