import pandas as pd
import numpy as np
from scipy.stats import genpareto
import json
import os

def fit_pot_gpd(scores, q=0.002, u_percentile=95):
    """
    Fits a Generalized Pareto Distribution to the tail of the scores using the Peak Over Threshold (POT) method.
    Derives the anomaly threshold corresponding to the (1 - q) quantile of the original distribution.
    """
    scores = scores[np.isfinite(scores)]
    
    # 1. Initial threshold u at u_percentile
    u = np.percentile(scores, u_percentile)
    
    # 2. Collect exceedances
    exceedances = scores[scores > u] - u
    
    if len(exceedances) == 0:
        print("Warning: No exceedances found above the initial threshold. Falling back to empirical quantile.")
        return float(np.quantile(scores, 1 - q))
    
    # 3. Fit GPD via MLE
    # floc=0 enforces that the location parameter is 0 (exceedances start at 0)
    c, loc, scale = genpareto.fit(exceedances, floc=0)
    
    # 4. Derive the anomaly threshold at target quantile (1 - q)
    p_u = 1.0 - u_percentile / 100.0
    
    if q >= p_u:
        threshold = np.quantile(scores, 1 - q)
    else:
        target_prob_in_tail = 1.0 - (q / p_u)
        excess_threshold = genpareto.ppf(target_prob_in_tail, c, loc=loc, scale=scale)
        threshold = u + excess_threshold
        
    return float(threshold)

def main():
    q = 0.002
    thresholds = {}
    
    vae_file = 'outputs/vae_v2_scores.csv'
    if os.path.exists(vae_file):
        df_vae = pd.read_csv(vae_file)
        if 'vae_anomaly_score' in df_vae.columns:
            scores = df_vae['vae_anomaly_score'].values
            thresh = fit_pot_gpd(scores, q=q, u_percentile=95)
            thresholds['vae_anomaly_score'] = {
                "threshold": round(thresh, 4),
                "q": q,
                "method": "POT-GPD"
            }
            print(f"VAE Anomaly Score Threshold: {thresh}")
        else:
            print(f"'vae_anomaly_score' not found in {vae_file}")
    else:
        print(f"{vae_file} not found")
            
    graph_file = 'outputs/graph_v2_scores.csv'
    if os.path.exists(graph_file):
        df_graph = pd.read_csv(graph_file)
        if 'graph_anomaly_score' in df_graph.columns:
            scores = df_graph['graph_anomaly_score'].values
            thresh = fit_pot_gpd(scores, q=q, u_percentile=95)
            thresholds['graph_anomaly_score'] = {
                "threshold": round(thresh, 4),
                "q": q,
                "method": "POT-GPD"
            }
            print(f"Graph Anomaly Score Threshold: {thresh}")
        else:
            print(f"'graph_anomaly_score' not found in {graph_file}")
    else:
        print(f"{graph_file} not found")

    with open('outputs/evt_thresholds_v2.json', 'w') as f:
        json.dump(thresholds, f, indent=2)
    
    print("evt_thresholds_v2.json created.")

if __name__ == "__main__":
    main()
