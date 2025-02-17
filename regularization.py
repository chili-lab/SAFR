import torch
import numpy as np

def polysemanticity_loss(weights):
    """
    Calculate the polysemanticity regularization loss.
    
    Args:
        weights: Tensor of shape (B, L, E)
        
    Returns:
        A scalar loss value.
    """
    B, L, E = weights.shape
    W_norm = weights / (1e-5 + torch.linalg.norm(weights, dim=-1, keepdim=True))
    interference = torch.einsum('ble,bme->blm', W_norm, weights)
    mask = torch.eye(L, device=weights.device).unsqueeze(0).expand(B, -1, -1).bool()
    interference.masked_fill_(mask, 0)
    polysemanticity = torch.linalg.norm(interference, dim=-1)
    polysemanticity_values = polysemanticity / np.sqrt(E)
    loss = torch.mean(polysemanticity_values)
    return loss

def interference_attention(attn_weights):
    """
    Calculate interference attention from the attention weights.
    
    Args:
        attn_weights: Tensor of shape (B, H, L, L)
        
    Returns:
        Tensor of interference values.
    """
    B, H, L, _ = attn_weights.shape
    interference = torch.einsum('bhij,bhkj->bhik', attn_weights, attn_weights)
    mask = torch.eye(L, device=attn_weights.device)\
                .unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1).bool()
    interference.masked_fill_(mask, 0)
    return interference

def correlation_interference_loss(all_attn_weights, attn_correlation):
    """
    Calculate the correlation interference loss.
    
    Args:
        all_attn_weights: Tensor of all attention weights.
        attn_correlation: Normalized attention correlations.
        
    Returns:
        A scalar loss value.
    """
    interference = interference_attention(all_attn_weights)
    loss = attn_correlation * (1 - interference)
    return torch.mean(loss)
