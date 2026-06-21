import pandas as pd
import numpy as np
import json
import torch
import os

def main():
    print("Loading risk scores...")
    df_risk = pd.read_csv('risk_scores_v2.csv')
    
    print("Loading VAE scores...")
    df_vae = pd.read_csv('vae_v2_scores.csv')
    
    # Filter flagged applications
    # e.g., lgbm_risk_score_v2 > 0.5 or label_source != 'negative'
    df_flagged = df_risk[(df_risk['lgbm_risk_score_v2'] > 0.5) | (df_risk['label_source'] != 'negative')]
    print(f"Found {len(df_flagged)} flagged applications.")
    
    print("Loading graph data...")
    graph_path = 'identity_graph.pt'
    graph = None
    if os.path.exists(graph_path):
        graph = torch.load(graph_path, weights_only=False)
        
    print("Loading node mapping...")
    # Node index in graph corresponds to order in engineered_features_v2.csv
    df_eng = pd.read_csv('engineered_features_v2.csv', low_memory=False)
    node_to_appid = df_eng['application_id'].values
    appid_to_node = {app_id: i for i, app_id in enumerate(node_to_appid)}
    
    print("Pre-processing graph edges...")
    # Pre-process graph edges for fast lookup
    adj_list = {} # node -> list of (neighbor_node, relation)
    if graph is not None:
        for edge_type in graph.edge_types:
            src, rel, dst = edge_type
            edge_index = graph[edge_type].edge_index.numpy()
            for i in range(edge_index.shape[1]):
                u = edge_index[0, i]
                v = edge_index[1, i]
                if u not in adj_list:
                    adj_list[u] = []
                adj_list[u].append((v, rel))
                if v not in adj_list:
                    adj_list[v] = []
                adj_list[v].append((u, rel))
    
    print("Pre-processing VAE recon errors...")
    recon_map = {}
    for _, row in df_vae.iterrows():
        app_id = row['application_id']
        recon_dict = json.loads(row['recon_error_vector'])
        sorted_recon = sorted(recon_dict.items(), key=lambda x: x[1], reverse=True)
        top_dims = [k for k, v in sorted_recon[:3]]
        recon_map[app_id] = top_dims
    
    explanation_cards = []
    print("Generating explanation cards...")
    for _, row in df_flagged.iterrows():
        app_id = row['application_id']
        risk_score = float(row['lgbm_risk_score_v2'])
        label_source = row['label_source']
        
        top_shap = []
        if pd.notna(row['top_shap_features']):
            top_shap = json.loads(row['top_shap_features'])
        
        channel = "tabular"
        if top_shap:
            first_feat = top_shap[0]
            if first_feat == "graph_anomaly_score" or first_feat.startswith("graph_"):
                channel = "graph"
            elif first_feat == "vae_anomaly_score" or first_feat.startswith("recon_"):
                channel = "reconstruction"
        
        # Get top graph neighbors
        neighbors_list = []
        node_idx = appid_to_node.get(app_id)
        if node_idx is not None and node_idx in adj_list:
            neighbors = adj_list[node_idx]
            seen = set()
            for n_idx, rel in neighbors:
                if n_idx not in seen and n_idx != node_idx:
                    neighbors_list.append({
                        "application_id": str(node_to_appid[n_idx]),
                        "relation": rel,
                        "weight": 1.0 # default weight as we don't have attention/GNNExplainer weights
                    })
                    seen.add(n_idx)
                if len(neighbors_list) >= 3:
                    break
        
        # Get top reconstruction dims
        top_dims = recon_map.get(app_id, [])
        
        # Build narrative
        narrative_parts = []
        if neighbors_list:
            rels = list(set([n['relation'] for n in neighbors_list]))
            narrative_parts.append(f"shares {', '.join(rels)} with {len(neighbors_list)} other(s)")
        if top_dims:
            narrative_parts.append(f"reconstructs poorly on {', '.join(top_dims)}")
        
        if narrative_parts:
            narrative = f"This application {' and '.join(narrative_parts)}."
        else:
            narrative = "Anomalous tabular features detected."
        
        card = {
            "application_id": app_id,
            "risk_score": risk_score,
            "label_source": label_source,
            "anomaly_channel": channel,
            "top_shap_features": top_shap,
            "top_graph_neighbors": neighbors_list,
            "top_reconstruction_dims": top_dims,
            "narrative": narrative
        }
        explanation_cards.append(card)
        
    with open('explanation_cards_v2.json', 'w') as f:
        json.dump(explanation_cards, f, indent=2)
        
    print(f"Saved {len(explanation_cards)} explanation cards to explanation_cards_v2.json")

if __name__ == '__main__':
    main()
