import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import RGCNConv
import json
import itertools
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import average_precision_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# --- Model Definitions ---
class TabularVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super(TabularVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU()
        )
        self.fc_mu = nn.Linear(16, latent_dim)
        self.fc_logvar = nn.Linear(16, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, input_dim), nn.Sigmoid()
        )
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def decode(self, z):
        return self.decoder(z)
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class DOMINANT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations):
        super().__init__()
        self.encoder1 = RGCNConv(in_channels, hidden_channels, num_relations)
        self.encoder2 = RGCNConv(hidden_channels, out_channels, num_relations)
        self.attr_decoder = nn.Sequential(
            nn.Linear(out_channels, hidden_channels), nn.ReLU(),
            nn.Linear(hidden_channels, in_channels)
        )
    def encode(self, x, edge_index, edge_type):
        h1 = F.relu(self.encoder1(x, edge_index, edge_type))
        return self.encoder2(h1, edge_index, edge_type)
    def decode_attr(self, z):
        return self.attr_decoder(z)
    def decode_struct(self, z):
        return z @ z.t()

def evaluate_models():
    print("=== Phase F: Evaluation Harness ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load original data
    df_real = pd.read_csv('data/processed/engineered_features_v2.csv', low_memory=False)
    df_real['is_synthetic'] = 0
    df_real['synth_category'] = 'NORMAL'
    N_real = len(df_real)
    
    # Build Unseen Synthetic Test Set
    categories = ['AGE_VIOLATION', 'INCOME_VIOLATION', 'IP_CONCENTRATION', 'MOTHER_NAME_COLLISION', 'FEE_INFLATION']
    N_syn = 150
    synthetic_dfs = []
    
    for i, cat in enumerate(categories):
        # Use entirely different seeds than training (100+i instead of 42+i)
        np.random.seed(100 + i)
        df_cat = df_real.sample(n=N_syn, replace=True, random_state=100+i).copy()
        df_cat['application_id'] = [f"SYN_{cat}_{j}" for j in range(N_syn)]
        df_cat['is_synthetic'] = 1
        df_cat['synth_category'] = cat
        
        if cat == 'AGE_VIOLATION':
            for j in range(N_syn):
                if df_cat.iloc[j]['pre_post_matric'] == 1:
                    df_cat.iloc[j, df_cat.columns.get_loc('age_at_registration')] = np.random.uniform(21, 30)
                else:
                    df_cat.iloc[j, df_cat.columns.get_loc('age_at_registration')] = np.random.uniform(36, 50)
        elif cat == 'INCOME_VIOLATION':
            df_cat['annual_family_income'] = np.random.uniform(0, 999, size=N_syn)
        elif cat == 'IP_CONCENTRATION':
            # Create extreme clusters
            cluster_ips = [f"192.168.99.{k}" for k in range(N_syn // 15 + 1)]
            df_cat['ip_address'] = np.random.choice(cluster_ips, size=N_syn)
            df_cat['mobile_no'] = np.arange(9000000000, 9000000000 + N_syn)
            df_cat['ip_application_count'] = 15
        elif cat == 'MOTHER_NAME_COLLISION':
            df_cat['mother_name'] = "COLLISION_MOTHER_NAME"
            df_cat['father_name'] = "COLLISION_MOTHER_NAME"
            if 'is_father_name_eq_mother' in df_cat.columns:
                df_cat['is_father_name_eq_mother'] = 1
        elif cat == 'FEE_INFLATION':
            df_cat['annual_family_income'] = np.random.uniform(20001, 100000, size=N_syn)
            df_cat['fee_income_ratio'] = np.random.uniform(1.1, 5.0, size=N_syn)
            
        synthetic_dfs.append(df_cat)
        
    df_combined = pd.concat([df_real] + synthetic_dfs, ignore_index=True)
    
    with open('data/processed/v2_feature_schema.json', 'r') as f:
        schema = json.load(f)
    features = schema.get('features', []) + schema.get('aggregation_features', [])
    excluded = schema.get('excluded', [])
    valid_cols = [c for c in features if c not in excluded and c in df_combined.columns]
    numeric_cols = df_combined[valid_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    # Prepare Tabular Data
    X_comb = df_combined[numeric_cols].fillna(0.0).values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_comb)
    
    # 1. Evaluate Tabular VAE
    model_vae = TabularVAE(input_dim=len(numeric_cols), latent_dim=8).to(device)
    model_vae.load_state_dict(torch.load('models/tabular_vae_v2.pth', map_location=device))
    model_vae.eval()
    
    vae_scores = []
    with torch.no_grad():
        x_tens = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        recon_x, _, _ = model_vae(x_tens)
        mse_feature = torch.pow(recon_x - x_tens, 2)
        mse_total = mse_feature.mean(dim=1)
        anomaly_score = 1.0 - torch.exp(-mse_total)
        vae_scores = anomaly_score.cpu().numpy()
    df_combined['vae_anomaly_score'] = vae_scores
    
    # Prepare Graph Data
    import networkx as nx
    G = nx.MultiGraph()
    for i in range(len(df_combined)):
        G.add_node(i)
        
    edge_types = {'mobile_no': 'shares_mobile', 'ip_address': 'shares_ip', 
                  'father_name': 'shares_father_name', 'mother_name': 'shares_mother_name', 
                  'permanent_pincode': 'shares_pincode'}
    
    data = HeteroData()
    data['application'].x = x_tens.cpu()
    
    for col, edge_type in edge_types.items():
        src, dst = [], []
        if col in df_combined.columns:
            grouped = df_combined.dropna(subset=[col]).groupby(col)
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
        
    h_data = data.to_homogeneous().to(device)
    num_relations = h_data.edge_type.max().item() + 1
    
    # 2. Evaluate Graph AE
    graph_checkpoint = torch.load('models/graph_autoencoder_v2.pth', map_location=device)
    model_graph = DOMINANT(x_tens.size(1), 64, 32, num_relations).to(device)
    model_graph.load_state_dict(graph_checkpoint['model_state_dict'])
    centroid = graph_checkpoint['centroid'].to(device)
    model_graph.eval()
    
    N_comb = len(df_combined)
    A_dense = torch.zeros((N_comb, N_comb), device=device)
    A_dense[h_data.edge_index[0], h_data.edge_index[1]] = 1.0
    pos_weight = torch.tensor([(N_comb * N_comb - h_data.edge_index.size(1)) / max(h_data.edge_index.size(1), 1)], device=device)
    
    with torch.no_grad():
        z = model_graph.encode(h_data.x, h_data.edge_index, h_data.edge_type)
        attr_recon = model_graph.decode_attr(z)
        attr_loss_matrix = F.mse_loss(attr_recon, h_data.x, reduction='none')
        attr_recon_error = attr_loss_matrix.mean(dim=1).cpu().numpy()
        
        logits = model_graph.decode_struct(z)
        struct_loss_matrix = F.binary_cross_entropy_with_logits(logits, A_dense, pos_weight=pos_weight, reduction='none')
        struct_recon_error = struct_loss_matrix.mean(dim=1).cpu().numpy()
        
        dist_sq = torch.sum((z - centroid) ** 2, dim=1).cpu().numpy()
        
        def normalize(arr): return (arr - arr.mean()) / (arr.std() + 1e-8)
        graph_anomaly_score = normalize(attr_recon_error) + normalize(struct_recon_error) + normalize(dist_sq)
        
    df_combined['graph_anomaly_score'] = graph_anomaly_score
    
    # 3. Calculate PR-AUC vs Baselines
    print("\n--- PR-AUC Comparison: V2 Models vs V1 Baseline ---")
    baselines = {
        'INCOME_VIOLATION': 0.1162,
        'AGE_VIOLATION': 0.0506,
        'MOTHER_NAME_COLLISION': 0.0258,
        'FEE_INFLATION': 0.0264,
        'IP_CONCENTRATION': 0.0239
    }
    
    for cat in categories:
        df_sub = df_combined[(df_combined['synth_category'] == 'NORMAL') | (df_combined['synth_category'] == cat)]
        y_true = df_sub['is_synthetic'].values
        
        # Use VAE score for tabular, Graph score for relational
        if cat in ['AGE_VIOLATION', 'INCOME_VIOLATION']:
            y_score = df_sub['vae_anomaly_score'].values
            model_name = "Tabular VAE"
        else:
            y_score = df_sub['graph_anomaly_score'].values
            model_name = "Graph AE"
            
        pr_auc = average_precision_score(y_true, y_score)
        base = baselines.get(cat, 0.0)
        status = "PASSED" if pr_auc > base else "FAILED"
        print(f"{cat:<25} | V1: {base:.4f} | V2 {model_name}: {pr_auc:.4f} | {status}")

if __name__ == '__main__':
    evaluate_models()
