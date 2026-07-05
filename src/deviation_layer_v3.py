"""
deviation_layer_v3.py

T2 Deviation Layer — weak-supervision deviation score (DevNet/PReNet-style),
leakage-safe via out-of-fold (OOF) stacking.

Cold-start (hard stop #19 / D3): a category uses real-confirmed anomalies only
once it has >= DEV_MIN_CONFIRMED_PER_CATEGORY confirmed patterns; below that it
falls back to synthetic archetypes.

Leakage-safe (D2): real nodes get OOF scores for the fusion fit set. A final
model trained on all real normals + anomalies is returned so *unseen* nodes
(e.g. injected eval nodes) can be scored without retraining and without ever
being in the training set.

Outputs application_id, deviation_score, evidence_source (hard stop #18).
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold

from src.config_v3 import (
    DEVIATION_LAYER_ENABLED, DEV_MIN_CONFIRMED_PER_CATEGORY, DEV_OOF_FOLDS,
    DEV_HIDDEN, DEV_EPOCHS, DEV_LR, DEV_CONF_MARGIN, RANDOM_SEED
)

OUT_CSV = Path("outputs/deviation_scores_v3.csv")
DEV_IN_FEATURES = 78  # 68 node features + 5 dense-block + 5 degree


class DeviationNet(nn.Module):
    def __init__(self, in_features: int = DEV_IN_FEATURES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, DEV_HIDDEN),
            nn.ReLU(),
            nn.Linear(DEV_HIDDEN, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def build_dev_features(x_features, dense_block_df, degree_df) -> np.ndarray:
    """[node features | dense-block scores | degree features] -> (N, 78).
    Single source of truth for column order so train and score never disagree."""
    n = len(x_features)
    dense_cols = [c for c in dense_block_df.columns if c != "application_id"]
    dense_feats = dense_block_df[dense_cols].values if dense_cols else np.zeros((n, 5), dtype=np.float32)
    deg_cols = [c for c in degree_df.columns if c != "application_id"]
    deg_feats = degree_df[deg_cols].values if deg_cols else np.zeros((n, 5), dtype=np.float32)
    return np.concatenate([np.asarray(x_features), dense_feats, deg_feats], axis=1).astype(np.float32)


def _train_devnet(X_np: np.ndarray, y_np: np.ndarray, in_features: int) -> DeviationNet:
    """DevNet deviation loss: pull normals toward a N(0,1) reference; force labeled
    anomalies a margin beyond the normal mean (PReNet-style contrastive)."""
    model = DeviationNet(in_features)
    opt = optim.Adam(model.parameters(), lr=DEV_LR)
    Xt = torch.tensor(X_np, dtype=torch.float32)
    yt = torch.tensor(y_np, dtype=torch.float32)
    model.train()
    for _ in range(DEV_EPOCHS):
        opt.zero_grad()
        s = model(Xt)
        s_n = s[yt == 0.0]
        s_a = s[yt == 1.0]
        loss_n = torch.abs(s_n - torch.randn_like(s_n)).mean() if len(s_n) > 0 else torch.tensor(0.0)
        if len(s_a) > 0:
            mean_n = s_n.mean() if len(s_n) > 0 else torch.tensor(0.0)
            loss_a = torch.relu(DEV_CONF_MARGIN - (s_a - mean_n)).mean()
        else:
            loss_a = torch.tensor(0.0)
        loss = loss_n + loss_a
        loss.backward()
        opt.step()
    model.eval()
    return model


def score_devnet(model: DeviationNet, X_np: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X_np, dtype=torch.float32)).numpy()


def _category_anomalies(confirmed_patterns, synthetic_exposure, app_ids, app_id_to_idx):
    """Per-category label selection (D3). Returns (real anomaly indices,
    synthetic anomaly feature rows, evidence_source array)."""
    anomaly_indices = []
    synthetic_features = []
    evidence_source = np.full(len(app_ids), "neither", dtype=object)
    for category, syn_feats in synthetic_exposure.items():
        conf_ids = confirmed_patterns.get(category, [])
        if len(conf_ids) >= DEV_MIN_CONFIRMED_PER_CATEGORY:
            for aid in conf_ids:
                if aid in app_id_to_idx:
                    idx = app_id_to_idx[aid]
                    anomaly_indices.append(idx)
                    evidence_source[idx] = f"confirmed:{category}"
        else:
            for sf in syn_feats:
                synthetic_features.append(sf)
    return anomaly_indices, synthetic_features, evidence_source


def train_and_score_oof(x_features, dense_block_df, degree_df,
                        synthetic_exposure, confirmed_patterns, app_ids):
    """Returns (df[application_id, deviation_score, evidence_source], final_model).

    Real nodes get leakage-safe OOF scores. final_model is trained on all real
    normals + anomalies and is used by callers to score unseen nodes."""
    n_real = len(app_ids)
    app_id_to_idx = {aid: i for i, aid in enumerate(app_ids)}

    X_real = build_dev_features(x_features, dense_block_df, degree_df)

    anomaly_indices, synthetic_features, evidence_source = _category_anomalies(
        confirmed_patterns, synthetic_exposure, app_ids, app_id_to_idx
    )
    anomaly_set = set(anomaly_indices)

    # synthetic anomalies have no graph context -> pad dense+degree with zeros.
    # pad width is derived from the real feature matrix (68 node feats + N dense +
    # M degree), NOT hardcoded, so it stays correct when dense-block is gated to a
    # subset of relations (e.g. IP-only -> 1 dense col, not 5).
    if synthetic_features:
        pad_len = X_real.shape[1] - len(synthetic_features[0])
        X_syn = np.array(
            [np.concatenate([np.asarray(sf, dtype=np.float32), np.zeros(pad_len, dtype=np.float32)])
             for sf in synthetic_features],
            dtype=np.float32,
        )
    else:
        X_syn = np.zeros((0, X_real.shape[1]), dtype=np.float32)

    X_all = np.vstack([X_real, X_syn])
    y_all = np.zeros(len(X_all), dtype=np.float32)
    if anomaly_indices:
        y_all[anomaly_indices] = 1.0
    y_all[n_real:] = 1.0  # synthetic rows are anomalies
    anomaly_all_idx = np.where(y_all == 1.0)[0]

    normal_real_indices = np.array([i for i in range(n_real) if i not in anomaly_set], dtype=int)

    oof_scores = np.zeros(n_real, dtype=np.float32)
    if len(normal_real_indices) >= DEV_OOF_FOLDS:
        kf = KFold(n_splits=DEV_OOF_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        for train_idx, val_idx in kf.split(normal_real_indices):
            train_normals = normal_real_indices[train_idx]
            val_normals = normal_real_indices[val_idx]
            train_subset = np.concatenate([train_normals, anomaly_all_idx])
            m = _train_devnet(X_all[train_subset], y_all[train_subset], X_all.shape[1])
            oof_scores[val_normals] = score_devnet(m, X_all[val_normals])

    # final model on all real normals + anomalies (for scoring unseen nodes)
    final_train = np.concatenate([normal_real_indices, anomaly_all_idx])
    final_model = _train_devnet(X_all[final_train], y_all[final_train], X_all.shape[1])

    # real confirmed anomalies were in every fold's training set -> score with final model
    if anomaly_indices:
        oof_scores[np.array(anomaly_indices)] = score_devnet(final_model, X_real[np.array(anomaly_indices)])

    df = pd.DataFrame({
        "application_id": app_ids,
        "deviation_score": oof_scores,
        "evidence_source": evidence_source,
    })
    return df, final_model


def run_deviation_layer():
    print("[deviation_layer] run_deviation_layer() starting ...")
    if not DEVIATION_LAYER_ENABLED:
        print("[deviation_layer] DEVIATION_LAYER_ENABLED is False. Exiting.")
        return

    final_csv = Path("data/processed/engineered_features_v3.csv")
    df = pd.read_csv(final_csv)
    app_ids = df["application_id"].values

    with open("data/processed/v3_feature_schema.json") as f:
        schema = json.load(f)
    feat_cols = schema["features"]
    x_features = df[feat_cols].values.astype(np.float32)

    dense_csv = Path("outputs/dense_block_scores_v3.csv")
    dense_block_df = pd.read_csv(dense_csv) if dense_csv.exists() else pd.DataFrame({"application_id": app_ids})

    from src.config_v3 import DEGREE_FEATURES
    deg_cols = [c for c in DEGREE_FEATURES if c in df.columns]
    degree_df = df[["application_id"] + deg_cols]

    labels_json = Path("outputs/pseudo_labels_v3.json")
    confirmed_patterns = {}
    if labels_json.exists():
        labels_data = json.loads(labels_json.read_text())
        for r in labels_data["positive_set"]:
            if r.get("source") == "confirmed":
                cat = r.get("category", "UNKNOWN")
                confirmed_patterns.setdefault(cat, []).append(r["application_id"])

    from src.evaluate_model_v3 import INJECTION_FNS
    rng = np.random.default_rng(RANDOM_SEED)
    synthetic_exposure = {cat: fn(x_features, feat_cols, rng) for cat, fn in INJECTION_FNS.items()}

    out_df, _ = train_and_score_oof(
        x_features, dense_block_df, degree_df, synthetic_exposure, confirmed_patterns, app_ids
    )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"[deviation_layer] Saved {len(out_df)} scores -> {OUT_CSV}")


if __name__ == "__main__":
    run_deviation_layer()
