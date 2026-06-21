import pandas as pd
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import HeteroData
from sklearn.preprocessing import LabelEncoder
import json
import itertools

def build_graph():
    print("Loading data...")
    df = pd.read_csv('data/processed/engineered_features_v2.csv', low_memory=False)
    
    with open('data/processed/v2_feature_schema.json', 'r') as f:
        schema = json.load(f)
        
    feature_cols = schema['features'] + schema['aggregation_features']
    excluded = schema.get('excluded', [])
    valid_cols = [c for c in feature_cols if c not in excluded and c in df.columns]
    
    # We only keep numerical features for the tensor
    numeric_cols = df[valid_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    print("Extracting node features...")
    # Extract features
    features_df = df[numeric_cols].copy()
    features_df = features_df.fillna(0.0)
            
    # Convert to tensor
    x = torch.tensor(features_df.values, dtype=torch.float32)
    
    print("Building networkx graph...")
    # Typed edges: shares_mobile, shares_ip, shares_father_name, shares_mother_name, shares_pincode
    edge_types = {
        'mobile_no': 'shares_mobile',
        'ip_address': 'shares_ip',
        'father_name': 'shares_father_name',
        'mother_name': 'shares_mother_name',
        'permanent_pincode': 'shares_pincode'
    }
    
    G = nx.MultiGraph()
    for i in range(len(df)):
        G.add_node(i, application_id=df['application_id'].iloc[i])
        
    for col, edge_type in edge_types.items():
        if col not in df.columns:
            continue
        
        grouped = df.dropna(subset=[col]).groupby(col)
        for val, group in grouped:
            indices = group.index.tolist()
            if len(indices) > 1:
                edges = list(itertools.combinations(indices, 2))
                G.add_edges_from(edges, edge_type=edge_type)
                
    print("Networkx graph stats:")
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    
    print("Converting to PyG HeteroData...")
    data = HeteroData()
    data['application'].x = x
    
    for col, edge_type in edge_types.items():
        src = []
        dst = []
        if col in df.columns:
            grouped = df.dropna(subset=[col]).groupby(col)
            for val, group in grouped:
                indices = group.index.tolist()
                if len(indices) > 1:
                    for u, v in itertools.combinations(indices, 2):
                        src.extend([u, v])
                        dst.extend([v, u])
                        
        if src:
            edge_index = torch.tensor([src, dst], dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            
        data['application', edge_type, 'application'].edge_index = edge_index
        print(f"Edge type '{edge_type}': {edge_index.size(1)} directed edges")

    print("Saving identity_graph.pt...")
    torch.save(data, "data/processed/identity_graph.pt")
    print("Success")

if __name__ == "__main__":
    build_graph()
