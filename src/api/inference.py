"""
Read-only scoring for staged / not-yet-committed datasets.

Loads the *existing* trained checkpoint and scores whatever is currently at
the canonical FINAL_CSV/GRAPH_PT paths (caller is responsible for having
merged + rebuilt those first — see dataset_ops.py). Does NOT train, does NOT
overwrite the checkpoint or outputs/hybrid_scores_v3.csv. Delegates to
hybrid_graphmcm_v3.compute_score_frame() — the single source of truth for the
hybrid_scores_v3.csv schema (including per_feature_predicted_json) — so
staged scores are schema- and normalization-identical to canonical ones.
"""
import pandas as pd


def score_dataset_only(app_ids_to_return: set[str] | None = None) -> pd.DataFrame:
    """
    Score every node currently in FINAL_CSV/GRAPH_PT with the existing checkpoint.
    If app_ids_to_return is given, the returned frame is filtered to just those
    rows — normalization is still computed over the full population, matching
    how the canonical pipeline would score them.
    """
    from src.hybrid_graphmcm_v3 import compute_score_frame, load_model_and_inputs

    model, x_all, edge_index_list, edge_type_tensor, isolated_mask, app_ids, features = load_model_and_inputs()

    out_df = compute_score_frame(model, x_all, edge_index_list, edge_type_tensor, isolated_mask, app_ids, features)
    out_df["application_id"] = out_df["application_id"].astype(str)

    if app_ids_to_return is not None:
        out_df = out_df[out_df["application_id"].isin(app_ids_to_return)].reset_index(drop=True)

    return out_df
