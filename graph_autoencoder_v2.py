import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import time
from torch_geometric.data import HeteroData
from torch_geometric.nn import RGCNConv

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class DOMINANT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations):
        super().__init__()
        # GCN encoder
        self.encoder1 = RGCNConv(in_channels, hidden_channels, num_relations)
        self.encoder2 = RGCNConv(hidden_channels, out_channels, num_relations)
        
        # Attribute decoder
        self.attr_decoder = nn.Sequential(
            nn.Linear(out_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, in_channels)
        )
        
    def encode(self, x, edge_index, edge_type):
        h1 = F.relu(self.encoder1(x, edge_index, edge_type))
        h2 = self.encoder2(h1, edge_index, edge_type)
        return h2
        
    def decode_attr(self, z):
        return self.attr_decoder(z)
        
    def decode_struct(self, z):
        # dot-product inner product reconstructs adjacency
        return z @ z.t()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading data...")
    # Load inputs
    data = torch.load('identity_graph.pt', weights_only=False)
    syn_x = torch.load('synthetic_exposure_set.pt', weights_only=False).to(device)
    
    df = pd.read_csv('engineered_features_v2.csv', low_memory=False)
    app_ids = df['application_id'].values
    
    # Extract homogeneous graph
    h_data = data.to_homogeneous().to(device)
    x = h_data.x
    edge_index = h_data.edge_index
    edge_type = h_data.edge_type
    num_relations = edge_type.max().item() + 1
    
    N_normal = x.size(0)
    N_syn = syn_x.size(0)
    
    # Combine normal and synthetic nodes for Stage 1 (synthetic nodes have no edges)
    full_x = torch.cat([x, syn_x], dim=0)
    
    # Hyperparameters
    in_channels = x.size(1)
    hidden_channels = 64
    out_channels = 32
    epochs_stage1 = 100
    epochs_stage2 = 100
    lr = 1e-3
    
    model = DOMINANT(in_channels, hidden_channels, out_channels, num_relations).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print("Initializing DeepSVDD centroid...")
    model.eval()
    with torch.no_grad():
        z_init = model.encode(full_x, edge_index, edge_type)
        centroid = z_init[:N_normal].mean(dim=0).detach()
    
    # We will use dense adjacency matrix for structure loss
    print("Creating dense adjacency for structure loss...")
    A_dense = torch.zeros((N_normal, N_normal), device=device)
    A_dense[edge_index[0], edge_index[1]] = 1.0
    
    E = edge_index.size(1)
    pos_weight = torch.tensor([(N_normal * N_normal - E) / max(E, 1)], device=device)
    
    # Training Loop
    # STAGE 1
    print("Starting Stage 1: Synthetic Anomaly Exposure")
    model.train()
    for epoch in range(epochs_stage1):
        optimizer.zero_grad()
        
        # lambda_t decays from 1.0 to 0.0
        lambda_t = 1.0 - (epoch / epochs_stage1)
        
        z = model.encode(full_x, edge_index, edge_type)
        z_normal = z[:N_normal]
        z_syn = z[N_normal:]
        
        # DOMINANT Attribute Reconstruction
        attr_recon = model.decode_attr(z_normal)
        attr_loss_matrix = F.mse_loss(attr_recon, x, reduction='none')
        attr_loss = attr_loss_matrix.mean()
        
        # DOMINANT Structure Reconstruction
        logits = model.decode_struct(z_normal)
        struct_loss_matrix = F.binary_cross_entropy_with_logits(logits, A_dense, pos_weight=pos_weight, reduction='none')
        struct_loss = struct_loss_matrix.mean()
        
        # DeepSVDD Loss (pull normal to centroid)
        dist_sq_normal = torch.sum((z_normal - centroid) ** 2, dim=1)
        svdd_loss = dist_sq_normal.mean()
        
        L_recon = attr_loss + struct_loss + svdd_loss
        
        # Exposure Loss (push synthetic away from centroid)
        dist_syn = torch.sum((z_syn - centroid) ** 2, dim=1)
        # Using exponential to avoid hard margin tuning
        L_exposure = torch.mean(torch.exp(-torch.sqrt(dist_syn + 1e-6)))
        
        loss = L_recon + lambda_t * L_exposure
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs_stage1} | Loss: {loss.item():.4f} | Recon: {L_recon.item():.4f} | Exposure: {L_exposure.item():.4f}")
            
    # STAGE 2
    print("Starting Stage 2: Free Reconstruction")
    for epoch in range(epochs_stage2):
        optimizer.zero_grad()
        
        # only normal nodes matter now
        z = model.encode(x, edge_index, edge_type)
        
        attr_recon = model.decode_attr(z)
        attr_loss_matrix = F.mse_loss(attr_recon, x, reduction='none')
        attr_loss = attr_loss_matrix.mean()
        
        logits = model.decode_struct(z)
        struct_loss_matrix = F.binary_cross_entropy_with_logits(logits, A_dense, pos_weight=pos_weight, reduction='none')
        struct_loss = struct_loss_matrix.mean()
        
        dist_sq_normal = torch.sum((z - centroid) ** 2, dim=1)
        svdd_loss = dist_sq_normal.mean()
        
        loss = attr_loss + struct_loss + svdd_loss
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs_stage2} | Loss: {loss.item():.4f} | Attr: {attr_loss.item():.4f} | Struct: {struct_loss.item():.4f}")

    print("Training complete. Generating scores...")
    model.eval()
    with torch.no_grad():
        z = model.encode(x, edge_index, edge_type)
        
        # Attribute recon error per node
        attr_recon = model.decode_attr(z)
        attr_loss_matrix = F.mse_loss(attr_recon, x, reduction='none')
        attr_recon_error = attr_loss_matrix.mean(dim=1).cpu().numpy()
        
        # Structure recon error per node
        logits = model.decode_struct(z)
        struct_loss_matrix = F.binary_cross_entropy_with_logits(logits, A_dense, pos_weight=pos_weight, reduction='none')
        struct_recon_error = struct_loss_matrix.mean(dim=1).cpu().numpy()
        
        # SVDD error (distance to centroid)
        dist_sq = torch.sum((z - centroid) ** 2, dim=1).cpu().numpy()
        
        # Normalize errors before combining
        def normalize(arr):
            return (arr - arr.mean()) / (arr.std() + 1e-8)
            
        # Higher = more anomalous
        graph_anomaly_score = normalize(attr_recon_error) + normalize(struct_recon_error) + normalize(dist_sq)

    # Save to CSV
    out_df = pd.DataFrame({
        'application_id': app_ids,
        'graph_anomaly_score': graph_anomaly_score,
        'attr_recon_error': attr_recon_error,
        'struct_recon_error': struct_recon_error
    })
    
    out_file = 'graph_v2_scores.csv'
    out_df.to_csv(out_file, index=False)
    print(f"Wrote scores to {out_file}")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'centroid': centroid
    }, 'graph_autoencoder_v2.pth')
    print("Saved to graph_autoencoder_v2.pth")

if __name__ == '__main__':
    main()
