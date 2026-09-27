#!/usr/bin/env python3
"""
Dual-End Confidence Fusion for Cross-Modal Retrieval

Core Innovation:
- Combines image-to-text (I2T) and text-to-image (T2I) retrieval directions
- Per-direction z-score normalization to handle distributional differences
- Exploits negative error correlation when it exists (ρ=-0.31 on COCO)

Reference: ACE Paper Section 4.1, Equations 3-5
"""
import torch
import torch.nn.functional as F
import numpy as np


def compute_dual_end_fusion(image_features, text_features, alpha=0.5):
    """
    Compute dual-end fusion scores with per-direction z-score normalization.
    
    Args:
        image_features: [N, d] normalized image features
        text_features: [M, d] normalized text features
        alpha: fusion weight (default 0.5 for equal weighting)
    
    Returns:
        fused_scores: [N, M] fused similarity matrix
        i2t_scores: [N, M] I2T scores (for analysis)
        t2i_scores: [N, M] T2I scores (for analysis)
    """
    N = image_features.shape[0]
    M = text_features.shape[0]
    
    # Step 1: Compute raw bidirectional similarities (Equations 1-2)
    # Although both use the same dot product, they operate in different ranking contexts
    similarity_matrix = image_features @ text_features.T  # [N, M]
    
    s_i2t = similarity_matrix  # I2T: rank texts for each image
    s_t2i = similarity_matrix.T  # T2I: rank images for each text
    
    # Step 2: Per-direction z-score normalization (Equations 3-4)
    # For I2T: normalize over all texts for each image
    mu_i = s_i2t.mean(dim=1, keepdim=True)  # [N, 1]
    sigma_i = s_i2t.std(dim=1, keepdim=True)  # [N, 1]
    s_i2t_norm = (s_i2t - mu_i) / (sigma_i + 1e-8)  # [N, M]
    
    # For T2I: normalize over all images for each text
    mu_j = s_t2i.mean(dim=1, keepdim=True)  # [M, 1]
    sigma_j = s_t2i.std(dim=1, keepdim=True)  # [M, 1]
    s_t2i_norm = (s_t2i - mu_j) / (sigma_j + 1e-8)  # [M, N]
    s_t2i_norm = s_t2i_norm.T  # [N, M] to match I2T dimension
    
    # Step 3: Fuse both directions (Equation 5)
    fused_scores = alpha * s_i2t_norm + (1 - alpha) * s_t2i_norm  # [N, M]
    
    return fused_scores, s_i2t_norm, s_t2i_norm


def compute_error_correlation(i2t_scores, t2i_scores, ground_truth_pairs, k=10):
    """
    Compute error correlation between I2T and T2I directions (post-hoc analysis).
    
    This measures whether errors are complementary (ρ < 0) or redundant (ρ ≈ 0).
    Used for understanding when dual-end fusion provides benefit.
    
    Args:
        i2t_scores: [N, M] I2T similarity scores
        t2i_scores: [N, M] T2I similarity scores  
        ground_truth_pairs: [(img_idx, txt_idx)] list of true pairs
        k: recall cutoff (default 10)
    
    Returns:
        rho: Pearson correlation of error indicators
    """
    N = i2t_scores.shape[0]
    
    errors_i2t = []
    errors_t2i = []
    
    for img_idx, txt_idx in ground_truth_pairs:
        # I2T error: does true text rank beyond top-k?
        i2t_ranks = (-i2t_scores[img_idx]).argsort()
        i2t_rank = (i2t_ranks == txt_idx).nonzero(as_tuple=True)[0].item()
        errors_i2t.append(1 if i2t_rank >= k else 0)
        
        # T2I error: does true image rank beyond top-k?
        t2i_ranks = (-t2i_scores[:, txt_idx]).argsort()
        t2i_rank = (t2i_ranks == img_idx).nonzero(as_tuple=True)[0].item()
        errors_t2i.append(1 if t2i_rank >= k else 0)
    
    # Pearson correlation
    errors_i2t = np.array(errors_i2t)
    errors_t2i = np.array(errors_t2i)
    
    if errors_i2t.std() == 0 or errors_t2i.std() == 0:
        return 0.0
    
    rho = np.corrcoef(errors_i2t, errors_t2i)[0, 1]
    return rho


# Example usage
if __name__ == "__main__":
    # Simulate features
    torch.manual_seed(42)
    N, M, d = 100, 100, 512
    
    image_features = F.normalize(torch.randn(N, d), dim=1)
    text_features = F.normalize(torch.randn(M, d), dim=1)
    
    # Compute fusion
    fused, i2t, t2i = compute_dual_end_fusion(image_features, text_features)
    
    print("Dual-End Fusion Demo")
    print(f"Image features: {image_features.shape}")
    print(f"Text features: {text_features.shape}")
    print(f"Fused scores: {fused.shape}")
    print(f"I2T normalized scores: {i2t.shape}")
    print(f"T2I normalized scores: {t2i.shape}")
    
    # Compute error correlation (requires ground truth)
    ground_truth = [(i, i) for i in range(min(N, M))]  # Assume diagonal pairing
    rho = compute_error_correlation(i2t, t2i, ground_truth, k=10)
    print(f"\nError correlation ρ = {rho:.3f}")
    print("(Negative ρ indicates complementary errors, enabling fusion benefit)")
