#!/usr/bin/env python3
"""
Adaptive Pseudo-Labeling for Test-Time Adaptation

Core Innovation:
- Entropy-based sample selection (not fixed confidence threshold)
- Adapts selection budget to model uncertainty
- Prevents capacity-dependent failure modes

Reference: ACE Paper Section 4.3, Algorithm 2, Equation 10
"""
import torch
import torch.nn.functional as F
import numpy as np


def compute_entropy(similarity_scores, temperature=0.07):
    """
    Compute prediction entropy for uncertainty estimation (Equation 8 in paper).
    
    Args:
        similarity_scores: [N, M] similarity matrix
        temperature: temperature for softmax normalization
    
    Returns:
        entropy: scalar average entropy
    """
    N = similarity_scores.shape[0]
    
    # Convert to probability distribution via softmax
    probs = F.softmax(similarity_scores / temperature, dim=1)  # [N, M]
    
    # Compute entropy: H = -sum(p * log(p))
    log_probs = torch.log(probs + 1e-8)
    entropy_per_sample = -(probs * log_probs).sum(dim=1)  # [N]
    
    # Average over all samples
    avg_entropy = entropy_per_sample.mean().item()
    
    return avg_entropy


def select_pseudo_label_budget(entropy):
    """
    Determine selection budget k* based on prediction entropy H (Equation 10).
    
    High-entropy models (H > 2.0): diffuse predictions, need more samples (20%)
    Low-entropy models (H < 1.0): peaked predictions, need fewer samples (5%)
    
    Args:
        entropy: scalar average entropy H
    
    Returns:
        selection_ratio: fraction of samples to select
    """
    if entropy > 2.0:
        return 0.20  # Top-20% for high-uncertainty models
    elif entropy >= 1.0:
        return 0.10  # Top-10% for medium-uncertainty models
    else:
        return 0.05  # Top-5% for low-uncertainty models


def adaptive_pseudo_labeling(
    image_features,
    text_features,
    projector,
    temperature=0.07,
    device='cuda'
):
    """
    Adaptive pseudo-labeling with entropy-based sample selection.
    
    Steps:
    1. Compute adapted features using projector
    2. Calculate similarity scores and entropy
    3. Determine selection budget based on entropy
    4. Select top-k* most confident samples
    5. Assign pseudo-labels
    
    Args:
        image_features: [N, d] frozen image features
        text_features: [M, d] frozen text features
        projector: trained consistency projector
        temperature: temperature for entropy calculation
        device: computation device
    
    Returns:
        selected_indices: indices of selected samples
        pseudo_labels: text indices for selected images
        selection_ratio: ratio of samples selected
    """
    N = image_features.shape[0]
    
    # Step 1: Apply projector to get adapted features
    projector.eval()
    with torch.no_grad():
        image_features = image_features.to(device)
        text_features = text_features.to(device)
        
        img_adapted = projector(image_features)
        txt_adapted = projector(text_features)
        
        # Compute similarity matrix
        similarity = img_adapted @ txt_adapted.T  # [N, M]
        
        # Step 2: Calculate entropy
        H = compute_entropy(similarity, temperature)
        
        # Step 3: Determine selection budget based on entropy
        selection_ratio = select_pseudo_label_budget(H)
        k_star = int(np.ceil(selection_ratio * N))
        
        # Step 4: Select top-k* most confident samples
        # Confidence = max similarity per image
        confidences, _ = similarity.max(dim=1)  # [N]
        top_k_values, top_k_indices = torch.topk(confidences, k_star)
        
        # Step 5: Assign pseudo-labels (argmax for selected samples)
        pseudo_labels = similarity[top_k_indices].argmax(dim=1)
    
    print(f"Entropy H = {H:.3f}")
    print(f"Selection ratio: {selection_ratio*100:.0f}% (Top-{k_star}/{N})")
    
    return top_k_indices.cpu(), pseudo_labels.cpu(), selection_ratio


def refine_projector_with_pseudo_labels(
    image_features,
    text_features,
    projector,
    selected_indices,
    pseudo_labels,
    num_epochs=1,
    lr=1e-3,
    temperature=0.07,
    device='cuda'
):
    """
    Refine projector using pseudo-labeled samples (one additional epoch).
    
    Args:
        image_features: [N, d] frozen image features
        text_features: [M, d] frozen text features
        projector: consistency projector to refine
        selected_indices: indices of high-confidence samples
        pseudo_labels: text indices for selected images
        num_epochs: refinement epochs (paper uses 1)
        lr: learning rate
        temperature: contrastive temperature
        device: computation device
    
    Returns:
        refined_projector: updated projector
    """
    optimizer = torch.optim.Adam(projector.parameters(), lr=lr)
    
    # Extract selected subset
    selected_images = image_features[selected_indices].to(device)
    selected_texts = text_features[pseudo_labels].to(device)
    
    projector.train()
    for epoch in range(num_epochs):
        # Forward pass
        img_adapted = projector(selected_images)
        txt_adapted = projector(selected_texts)
        
        # Contrastive loss (treat pseudo-pairs as positives)
        logits = (img_adapted @ txt_adapted.T) / temperature
        labels = torch.arange(len(selected_indices), device=device)
        loss = F.cross_entropy(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Pseudo-label refinement epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
    
    projector.eval()
    return projector


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("Adaptive Pseudo-Labeling Demo")
    print("="*80)
    
    # Simulate data
    torch.manual_seed(42)
    N, d = 1000, 512
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Frozen features
    image_features = F.normalize(torch.randn(N, d), dim=1)
    text_features = F.normalize(torch.randn(N, d), dim=1)
    
    # Pretrained projector (from consistency regularization)
    from consistency_regularization import ConsistencyProjector
    projector = ConsistencyProjector(d, d//2).to(device)
    
    print(f"\nDataset: {N} samples")
    print(f"Feature dimension: {d}")
    
    # Test different entropy scenarios
    print("\n" + "="*80)
    print("Scenario 1: High-entropy model (H > 2.0) → Select 20%")
    print("="*80)
    # Simulate high-entropy predictions
    high_entropy_sim = torch.randn(N, N) * 0.1  # Diffuse similarities
    H_high = compute_entropy(high_entropy_sim)
    ratio_high = select_pseudo_label_budget(H_high)
    print(f"Entropy: {H_high:.3f}, Selection ratio: {ratio_high*100:.0f}%")
    
    print("\n" + "="*80)
    print("Scenario 2: Low-entropy model (H < 1.0) → Select 5%")
    print("="*80)
    # Simulate low-entropy predictions
    low_entropy_sim = torch.randn(N, N)
    low_entropy_sim += torch.eye(N) * 5  # Peaked similarities
    H_low = compute_entropy(low_entropy_sim)
    ratio_low = select_pseudo_label_budget(H_low)
    print(f"Entropy: {H_low:.3f}, Selection ratio: {ratio_low*100:.0f}%")
    
    # Run adaptive pseudo-labeling
    print("\n" + "="*80)
    print("Running adaptive pseudo-labeling")
    print("="*80)
    selected_idx, pseudo_lbl, ratio = adaptive_pseudo_labeling(
        image_features, text_features, projector, device=device
    )
    
    print(f"\nSelected {len(selected_idx)} samples for refinement")
    print(f"Pseudo-labels assigned: {len(pseudo_lbl)}")
