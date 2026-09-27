#!/usr/bin/env python3
"""
Cross-Modal Consistency Regularization

Core Innovation:
- Trains lightweight residual projector on test image-text pairs
- Uses bidirectional contrastive loss for self-supervised adaptation
- Freezes pretrained vision-language encoders

CRITICAL: This is TRANSDUCTIVE test-time adaptation - we train on test pairs!
The paper (Section 4.2, Critical Acknowledgment) explicitly addresses this:
- Tiny projector capacity (0.17% of backbone) prevents memorization
- Short training (2 epochs) limits overfitting
- Zero-shot ranking evaluation (all N×M pairs)
- Cross-domain generalization (RSITMD success despite domain shift)

Reference: ACE Paper Section 4.2, Algorithm 1, Equations 6-9
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class ConsistencyProjector(nn.Module):
    """
    Lightweight residual projector for cross-modal consistency.
    
    Architecture: Two-layer MLP with residual connection
    - Input: d-dimensional frozen features
    - Hidden: d/2 dimensions
    - Output: d dimensions (residual added to input)
    """
    def __init__(self, input_dim=512, hidden_dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, d] normalized features
        Returns:
            z: [B, d] adapted normalized features
        """
        residual = self.proj(x)  # [B, d]
        z = x + residual  # Residual connection preserves pretrained knowledge
        z = F.normalize(z, dim=-1)  # L2 normalization
        return z


def bidirectional_contrastive_loss(img_features, txt_features, temperature=0.07):
    """
    Compute bidirectional contrastive loss (Equations 7-9).
    
    Treats each (image, text) pair as positive within the batch.
    Optimizes both I2T and T2I directions simultaneously.
    
    Args:
        img_features: [B, d] normalized image features
        txt_features: [B, d] normalized text features
        temperature: temperature parameter τ
    
    Returns:
        loss: scalar bidirectional contrastive loss
    """
    B = img_features.shape[0]
    
    # Compute similarity matrix
    logits = (img_features @ txt_features.T) / temperature  # [B, B]
    
    # Labels: diagonal elements are positive pairs
    labels = torch.arange(B, device=logits.device)
    
    # I2T loss: for each image, rank its corresponding text
    loss_i2t = F.cross_entropy(logits, labels)
    
    # T2I loss: for each text, rank its corresponding image
    loss_t2i = F.cross_entropy(logits.T, labels)
    
    # Bidirectional average (Equation 6)
    loss = (loss_i2t + loss_t2i) / 2.0
    
    return loss


def train_consistency_projector(
    image_features,
    text_features,
    input_dim=512,
    hidden_dim=256,
    num_epochs=2,
    batch_size=32,
    lr=1e-3,
    temperature=0.07,
    device='cuda'
):
    """
    Train consistency projector using test image-text pairs (Algorithm 1).
    
    CRITICAL: This trains on TEST DATA in the transductive setting.
    The paper acknowledges this raises memorization concerns but argues:
    - Projector capacity is tiny (0.17% of backbone)
    - Training is short (2 epochs)
    - Evaluation is zero-shot (ranks all N×M pairs)
    - Method generalizes across domains (e.g., RSITMD)
    
    Args:
        image_features: [N, d] frozen image features from test set
        text_features: [N, d] frozen text features from test set
        input_dim: feature dimension
        hidden_dim: projector hidden dimension (default d/2)
        num_epochs: training epochs (paper uses 2)
        batch_size: batch size (paper uses 32)
        lr: learning rate (paper uses 1e-3)
        temperature: contrastive temperature τ (paper uses 0.07)
        device: computation device
    
    Returns:
        projector: trained ConsistencyProjector module
    """
    N = image_features.shape[0]
    
    # Initialize projector
    projector = ConsistencyProjector(input_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(projector.parameters(), lr=lr)
    
    # Move features to device
    image_features = image_features.to(device)
    text_features = text_features.to(device)
    
    # Training loop
    projector.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        # Shuffle and batch
        indices = torch.randperm(N)
        
        for i in range(0, N, batch_size):
            batch_idx = indices[i:i+batch_size]
            
            # Get batch
            img_batch = image_features[batch_idx]
            txt_batch = text_features[batch_idx]
            
            # Forward pass
            img_adapted = projector(img_batch)
            txt_adapted = projector(txt_batch)
            
            # Compute loss
            loss = bidirectional_contrastive_loss(
                img_adapted, txt_adapted, temperature
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    projector.eval()
    return projector


def apply_consistency_adaptation(
    image_features,
    text_features,
    projector,
    device='cuda'
):
    """
    Apply trained projector to adapt features.
    
    Args:
        image_features: [N, d] frozen image features
        text_features: [M, d] frozen text features
        projector: trained ConsistencyProjector
        device: computation device
    
    Returns:
        adapted_img_features: [N, d] adapted image features
        adapted_txt_features: [M, d] adapted text features
    """
    projector.eval()
    with torch.no_grad():
        image_features = image_features.to(device)
        text_features = text_features.to(device)
        
        adapted_img = projector(image_features)
        adapted_txt = projector(text_features)
    
    return adapted_img.cpu(), adapted_txt.cpu()


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("Cross-Modal Consistency Regularization Demo")
    print("="*80)
    
    # Simulate test data
    torch.manual_seed(42)
    N, d = 1000, 512
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Frozen features from pretrained encoder
    image_features = F.normalize(torch.randn(N, d), dim=1)
    text_features = F.normalize(torch.randn(N, d), dim=1)
    
    print(f"\nTest set: {N} image-text pairs")
    print(f"Feature dimension: {d}")
    print(f"Device: {device}")
    
    # Train projector on test pairs (transductive adaptation)
    print("\n⚠️  TRAINING ON TEST DATA (Transductive Setting)")
    print("   Paper Section 4.2 Critical Acknowledgment addresses memorization concerns")
    
    projector = train_consistency_projector(
        image_features,
        text_features,
        input_dim=d,
        hidden_dim=d//2,
        num_epochs=2,
        batch_size=32,
        lr=1e-3,
        device=device
    )
    
    # Apply adaptation
    adapted_img, adapted_txt = apply_consistency_adaptation(
        image_features, text_features, projector, device
    )
    
    print(f"\n✓ Adapted image features: {adapted_img.shape}")
    print(f"✓ Adapted text features: {adapted_txt.shape}")
    
    # Check projector capacity
    total_params = sum(p.numel() for p in projector.parameters())
    print(f"\nProjector parameters: {total_params:,}")
    print(f"Backbone parameters (CLIP ViT-B/32): 151,000,000")
    print(f"Projector / Backbone: {total_params / 151e6 * 100:.2f}%")
