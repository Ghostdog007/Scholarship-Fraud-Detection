"""
ring_classifier_v3.py

Track M: T8c/d Ring Classifier
Trains a LightGBM classifier on subgraph fingerprints (positive vs negative rings).
Computes ring-probability and open-set novelty (distance to nearest positive prototype).
"""
import pickle
import numpy as np
import lightgbm as lgb
from pathlib import Path
import torch

from src.config_v3 import RANDOM_SEED, RING_N_NEG_SAMPLES
from src.ring_fingerprint_v3 import fingerprint, FINGERPRINT_DIM

MODEL_PATH = Path("models/ring_classifier_v3.pkl")

# Mirror fusion LGB params
LGB_PARAMS = {
    "objective":       "binary",
    "metric":          "average_precision",
    "num_leaves":      31,
    "learning_rate":   0.05,
    "n_estimators":    200,
    "min_child_samples": 5,
    "subsample":       0.8,
    "colsample_bytree": 0.8,
    "random_state":    RANDOM_SEED,
    "verbose":         -1,
}

class RingClassifier:
    def __init__(self):
        self.clf = lgb.LGBMClassifier(**LGB_PARAMS)
        self.positive_prototypes = [] # List of mean fingerprints per type/cluster
        self.is_fitted = False

    def fit(self, pos_subgraphs: list[dict], neg_subgraphs: list[dict]) -> None:
        """
        Fits the LightGBM model and stores positive prototypes for open-set novelty.
        """
        X = []
        y = []
        
        # Build positive prototypes (e.g. we can just store all positive fingerprints, 
        # or cluster them. Since the library is small, we can store all).
        self.positive_prototypes = []
        for sg in pos_subgraphs:
            fp = fingerprint(sg)
            X.append(fp)
            y.append(1)
            self.positive_prototypes.append(fp)
            
        for sg in neg_subgraphs:
            fp = fingerprint(sg)
            X.append(fp)
            y.append(0)
            
        if not X:
            print("[RingClassifier] No training data provided.")
            return
            
        X = np.vstack(X)
        y = np.array(y)
        
        self.clf.fit(X, y)
        self.is_fitted = True
        
        # Save model
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self, f)

    def score(self, subgraph: dict) -> float:
        """Returns the calibrated ring-probability in [0, 1]."""
        if not self.is_fitted:
            return 0.0
        fp = fingerprint(subgraph).reshape(1, -1)
        return float(self.clf.predict_proba(fp)[0, 1])

    def novelty(self, subgraph: dict) -> float:
        """
        Returns nearest-prototype distance (L2 or Euclidean).
        Larger distance = more novel.
        """
        if not self.positive_prototypes:
            return 0.0
            
        fp = fingerprint(subgraph)
        protos = np.vstack(self.positive_prototypes)
        
        distances = np.linalg.norm(protos - fp, axis=1)
        return float(np.min(distances))


def project_to_nodes(candidates: list[dict], classifier: RingClassifier, n_nodes: int) -> np.ndarray:
    """
    T8d: node projection helper.
    Every node inherits max ring-probability over the candidate rings it belongs to.
    """
    node_scores = np.zeros(n_nodes, dtype=np.float32)
    if not classifier.is_fitted:
        return node_scores
        
    for sg in candidates:
        score = classifier.score(sg)
        for node in sg["node_ids"]:
            if score > node_scores[node]:
                node_scores[node] = score
                
    return node_scores

def load_ring_classifier() -> RingClassifier:
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return RingClassifier()

