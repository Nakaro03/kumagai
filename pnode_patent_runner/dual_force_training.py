import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Set
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
from pnode_patent_runner.unified_training import sample_hard_negatives_v2, decode_bipartite

def compute_dual_force_loss(
    model: torch.nn.Module,
    data_t: Data,
    data_t1: Data,
    num_authors: int,
    historical_edges: Set[Tuple[int, int]],
    beta: float = 0.01,
    pos_weight: float = 5.0,
    latent_pred_weight: float = 1.0,
    future_link_weight: float = 10.0,
    num_neg_recon: int = 800,
    num_neg_future: int = 400,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    device = data_t.x.device
    
    # 1. Encode current year
    z_t, mu_t, logvar_t = model.encode(data_t.x, data_t.edge_index)
    
    # 2. Actual mu at t+1 for supervision
    with torch.no_grad():
        _, mu_t1_actual, _ = model.encode(data_t1.x, data_t1.edge_index)
    
    # 3. Reconstruction loss at t
    pos_edge_index = data_t.edge_index
    active_authors = torch.unique(pos_edge_index[0][pos_edge_index[0] < num_authors])
    active_topics = torch.unique(pos_edge_index[1][pos_edge_index[1] >= num_authors])
    pos_set = {tuple(p.tolist()) for p in pos_edge_index.t()}
    
    neg_edge_index = sample_hard_negatives_v2(
        model, z_t, active_authors, active_topics, pos_set, historical_edges, num_samples=num_neg_recon
    )
    
    recon_loss = torch.tensor(0.0, device=device)
    if neg_edge_index is not None:
        pos_pred = model.decode(z_t, pos_edge_index)
        neg_pred = model.decode(z_t, neg_edge_index)
        recon_loss = -torch.log(pos_pred + 1e-15).mean() * pos_weight - torch.log(1 - neg_pred + 1e-15).mean()
        
    # 4. KL Divergence
    kl_loss = -0.5 * torch.mean(1 + logvar_t - mu_t.pow(2) - logvar_t.exp())
    
    # 5. Latent Prediction Loss (ODE)
    # Dual-Force P-NODE uses topic info from data_t
    z_t1_pred = model.predict_future([z_t], data_t)
    latent_pred_loss = F.mse_loss(z_t1_pred, mu_t1_actual.detach())
    
    # 6. Future Link Prediction Loss
    future_link_loss = torch.tensor(0.0, device=device)
    pos_edge_index_t1 = data_t1.edge_index
    if pos_edge_index_t1.size(1) > 0:
        active_a_t1 = torch.unique(pos_edge_index_t1[0][pos_edge_index_t1[0] < num_authors])
        active_t_t1 = torch.unique(pos_edge_index_t1[1][pos_edge_index_t1[1] >= num_authors])
        pos_set_t1 = {tuple(p.tolist()) for p in pos_edge_index_t1.t()}
        
        neg_t1 = sample_hard_negatives_v2(
            model, z_t1_pred, active_a_t1, active_t_t1, pos_set_t1, historical_edges, num_samples=num_neg_future
        )
        if neg_t1 is not None:
            pos_pred_t1 = model.decode(z_t1_pred, pos_edge_index_t1)
            neg_pred_t1 = model.decode(z_t1_pred, neg_t1)
            future_link_loss = -torch.log(pos_pred_t1 + 1e-15).mean() * pos_weight - torch.log(1 - neg_pred_t1 + 1e-15).mean()
            
    total_loss = recon_loss + beta * kl_loss + latent_pred_weight * latent_pred_loss + future_link_weight * future_link_loss
    
    breakdown = {
        "total": float(total_loss.item()),
        "recon": float(recon_loss.item()),
        "kl": float(kl_loss.item()),
        "latent_pred": float(latent_pred_loss.item()),
        "future_link": float(future_link_loss.item()),
    }
    return total_loss, breakdown

def train_dual_force(
    model, graphs, num_authors, hist_edges, num_epochs=20, lr=0.001
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    years = sorted(graphs.keys())
    device = next(model.parameters()).device
    history = []
    
    for epoch in range(num_epochs):
        model.train()
        total_epoch_loss = 0
        epoch_breakdown = {}
        
        for i in range(len(years) - 1):
            y_t, y_t1 = years[i], years[i+1]
            data_t = graphs[y_t].to(device)
            data_t1 = graphs[y_t1].to(device)
            
            optimizer.zero_grad()
            loss, breakdown = compute_dual_force_loss(model, data_t, data_t1, num_authors, hist_edges)
            loss.backward()
            optimizer.step()
            total_epoch_loss += loss.item()
            
            for k, v in breakdown.items():
                epoch_breakdown[k] = epoch_breakdown.get(k, 0.0) + v
        
        # 平均値を計算
        num_steps = len(years) - 1
        avg_breakdown = {k: v / num_steps for k, v in epoch_breakdown.items()}
        history.append(avg_breakdown)
            
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_epoch_loss/num_steps:.4f}")
    
    return model, history
