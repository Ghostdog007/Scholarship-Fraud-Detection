"""
Recompute staged cohort outputs (stress_testing_1, sample_cohort) against the
CURRENT model checkpoint, using the exact same read-only code path as
POST /v3/monitoring/evaluate-dataset (src/api/handlers/monitoring.py) --
merge into raw, rebuild features/graph, score with the live checkpoint,
restore canonical files, write outputs/staged_*_<name>.* .

Run after any full retrain so cohort previews reflect the new model instead
of stale scores from the previous checkpoint.
"""
from pathlib import Path

from src.api.handlers.monitoring import evaluate_dataset
from src.api.schemas import EvaluateDatasetRequest

DATASETS = [
    "data/uploads/stress_testing_1.csv",
    "data/uploads/sample_cohort.csv",
]

for path in DATASETS:
    if not Path(path).exists():
        print(f"[recompute_cohorts] SKIP (not found): {path}")
        continue
    print(f"[recompute_cohorts] evaluate_dataset({path}) ...")
    resp = evaluate_dataset(EvaluateDatasetRequest(dataset_path=path))
    print(f"[recompute_cohorts]   -> {resp}")
