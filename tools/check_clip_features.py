"""Check if original CLIP features have diversity."""
import torch
import torch.nn.functional as F

fd = torch.load("data/cifar_features.pt", weights_only=True)
v_all, t_all = fd['v'][:100], fd['t'][:100]

print("=== Original CLIP Features ===")
print(f"Vision: shape={v_all.shape}, mean={v_all.mean():.4f}, std={v_all.std():.4f}")
print(f"Text: shape={t_all.shape}, mean={t_all.mean():.4f}, std={t_all.std():.4f}")

# Check diversity
v_norm = F.normalize(v_all, dim=-1)
t_norm = F.normalize(t_all, dim=-1)

v_sim = v_norm @ v_norm.t()
t_sim = t_norm @ t_norm.t()

print(f"\nVision self-similarity: mean={v_sim.mean():.4f}, std={v_sim.std():.4f}")
print(f"Text self-similarity: mean={t_sim.mean():.4f}, std={t_sim.std():.4f}")

# Check unique text features
unique_texts = []
for i in range(len(t_all)):
    is_unique = True
    for u in unique_texts:
        if torch.allclose(t_all[i], u, atol=1e-5):
            is_unique = False
            break
    if is_unique:
        unique_texts.append(t_all[i])

print(f"\nNumber of unique text features: {len(unique_texts)} / {len(t_all)}")

# Check first 10 samples
print("\nFirst 10 text features (first 5 dims):")
for i in range(10):
    print(f"  [{i}]: {t_all[i, :5].numpy()}")

# Cross-modal similarity
cross_sim = v_norm @ t_norm.t()
print(f"\nCross-modal similarity (original CLIP):")
print(f"  Diagonal (correct pairs): mean={cross_sim.diag().mean():.4f}, std={cross_sim.diag().std():.4f}")
print(f"  Off-diagonal (wrong pairs): mean={cross_sim[~torch.eye(100, dtype=bool)].mean():.4f}, std={cross_sim[~torch.eye(100, dtype=bool)].std():.4f}")
