import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class TabularVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super(TabularVAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(16, latent_dim)
        self.fc_logvar = nn.Linear(16, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.Sigmoid()
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
        x_recon = self.decode(z)
        return x_recon, mu, logvar

def elbo_loss(recon_x, x, mu, logvar):
    mse = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + kld, mse, kld

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading features...")
    df = pd.read_csv('data/processed/engineered_features_v2.csv', low_memory=False)
    
    with open('data/processed/v2_feature_schema.json', 'r') as f:
        schema = json.load(f)
        
    features = schema.get('features', []) + schema.get('aggregation_features', [])
    excluded = schema.get('excluded', [])
    valid_cols = [c for c in features if c not in excluded and c in df.columns]
    
    numeric_cols = df[valid_cols].select_dtypes(include=[np.number]).columns.tolist()
    print(f"Number of numeric features: {len(numeric_cols)}")
    
    X = df[numeric_cols].fillna(0.0).values
    app_ids = df['application_id'].values
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Load synthetic tensor
    print("Loading synthetic exposure tensor...")
    synth_tensor = torch.load('data/processed/synthetic_exposure_set.pt', map_location='cpu')
    synth_X = scaler.transform(synth_tensor.numpy())
    
    batch_size = 256
    normal_data = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    synth_data = torch.tensor(synth_X, dtype=torch.float32).to(device)
    
    normal_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(normal_data), 
        batch_size=batch_size, shuffle=True
    )
    
    model = TabularVAE(input_dim=len(numeric_cols), latent_dim=8).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs_stage1 = 20
    epochs_stage2 = 30
    margin = 5.0
    
    print("Starting Stage 1: Synthetic anomaly exposure pretraining")
    for epoch in range(epochs_stage1):
        model.train()
        total_loss = 0
        lambda_t = 1.0 - (epoch / epochs_stage1)
        
        synth_idx = 0
        
        for batch_idx, (batch_x,) in enumerate(normal_loader):
            optimizer.zero_grad()
            
            recon_x, mu, logvar = model(batch_x)
            loss_recon, mse, kld = elbo_loss(recon_x, batch_x, mu, logvar)
            
            batch_synth = synth_data[synth_idx : synth_idx + batch_x.size(0)]
            if batch_synth.size(0) < batch_x.size(0):
                synth_idx = 0
                batch_synth = synth_data[synth_idx : synth_idx + batch_x.size(0)]
            synth_idx += batch_x.size(0)
            
            _, mu_synth, _ = model(batch_synth)
            
            centroid = mu.mean(dim=0).detach()
            dist = torch.norm(mu_synth - centroid, dim=1)
            loss_exp = torch.mean(torch.relu(margin - dist))
            
            loss = loss_recon / batch_x.size(0) + lambda_t * loss_exp
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs_stage1}, Loss: {total_loss/len(normal_loader):.4f}, Lambda: {lambda_t:.2f}")

    print("Starting Stage 2: Free reconstruction discovery")
    for epoch in range(epochs_stage2):
        model.train()
        total_loss = 0
        for batch_idx, (batch_x,) in enumerate(normal_loader):
            optimizer.zero_grad()
            recon_x, mu, logvar = model(batch_x)
            loss_recon, mse, kld = elbo_loss(recon_x, batch_x, mu, logvar)
            loss = loss_recon / batch_x.size(0)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs_stage2}, Loss: {total_loss/len(normal_loader):.4f}")
            
    print("Scoring...")
    model.eval()
    scores = []
    recon_vectors = []
    
    score_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(normal_data), 
        batch_size=256, shuffle=False
    )
    
    with torch.no_grad():
        for (batch_x,) in score_loader:
            recon_x, _, _ = model(batch_x)
            mse_feature = torch.pow(recon_x - batch_x, 2)
            mse_total = mse_feature.mean(dim=1)
            
            # Anomaly score = exp(-MSE), inverted so higher = more anomalous.
            # Using 1 - exp(-MSE) ensures higher MSE -> higher score
            anomaly_score = 1.0 - torch.exp(-mse_total)
            
            scores.extend(anomaly_score.cpu().numpy().tolist())
            recon_vectors.extend(mse_feature.cpu().numpy().tolist())
            
    df_out = pd.DataFrame({
        'application_id': app_ids,
        'vae_anomaly_score': scores,
    })
    
    recon_json = [json.dumps(dict(zip(numeric_cols, vec))) for vec in recon_vectors]
    df_out['recon_error_vector'] = recon_json
    
    df_out.to_csv('outputs/vae_v2_scores.csv', index=False)
    print("Saved to vae_v2_scores.csv")
    
    torch.save(model.state_dict(), 'models/tabular_vae_v2.pth')
    print("Saved to tabular_vae_v2.pth")

if __name__ == '__main__':
    main()
