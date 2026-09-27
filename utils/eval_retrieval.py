"""Proper cross-modal retrieval evaluation on CIFAR-10 CLIP features."""
import torch
import numpy as np
from pathlib import Path

# Load pre-computed CLIP features (20K samples)
print("Loading features...")
fd = torch.load("data/cifar_features.pt", weights_only=True)
v_all = fd['v'].cpu()
t_all = fd['t'].cpu()
n_total = len(v_all)
print("Total samples:", n_total)

# Split: 80% train, 20% val (stratified by class)
import random
random.seed(42)
indices = list(range(n_total))
np.random.seed(42)
np.random.shuffle(indices)
n_val = min(200, max(50, n_total // 10))
train_idx, val_idx = indices[:-n_val], indices[-n_val:]

print("Train: %d, Val: %d" % (len(train_idx), len(val_idx)))

# Get features for train/val split
v_train = v_all[train_idx]
t_train = t_all[train_idx]
v_val = v_all[val_idx]
t_val = t_all[val_idx]

# ── Baseline: No Bridge (raw CLIP with projection) ─────────────────────
print("\n=== Baseline: Raw CLIP (no bridge, projecting both to 512) ===")

from transformers import CLIPModel, CLIPTextModel
text_proj = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").text_projection

# Project vision features from 768->512 using CLIP's own projection
proj_linear = torch.nn.Linear(768, 512)
torch.nn.init.xavier_uniform_(proj_linear.weight)

# Simple projection to same dim for comparison  
v_proj = proj_linear(v_val).cpu()    # [200, 512]
t_set = t_train.cpu()                # [19800, 512]

sim = v_proj @ t_set.t()              # [200, 19800]
preds = torch.argsort(sim, dim=1, descending=True)
r1_raw = (preds[:, :1] == torch.arange(len(v_proj)).unsqueeze(1)).sum().item() / len(v_proj)
print("Image->Text R@1: %.1f%%" % (r1_raw * 100))

# Also text-to-image
t2i_r1 = compute_r_at_k(t_val, v_train)['R@1']
print("Text->Img R@1: %.1f%%" % (t2i_r1 * 100))

# ── Bridge with different ranks ───────────────────────────────────────

class BridgeModule(torch.nn.Module):
    def __init__(self, d=768, r=32):
        super().__init__()
        self.proj = torch.nn.Linear(d, 512) if d != 512 else torch.nn.Identity()
        self.use_lowrank = (r < 512)
        if self.use_lowrank:
            self.U = torch.nn.Parameter(torch.empty(512, r))
            self.V = torch.nn.Parameter(torch.empty(512, r))
            self.s = torch.nn.Parameter(torch.ones(r))
            torch.nn.init.xavier_uniform_(self.U)
            torch.nn.init.xavier_uniform_(self.V)
        else:
            self.proj2 = torch.nn.Linear(512, 512)
            torch.nn.init.xavier_uniform_(self.proj2.weight)

    def forward(self, x):
        y = self.proj(x)
        if self.use_lowrank:
            return (y @ self.V * self.s) @ self.U.t()
        else:
            return self.proj2(y)

# Evaluate bridges (just random initialization - no training)
print("\n=== Untrained Bridges (random init, no training) ===")
for rank in [32, 64, 128, 256]:
    bv = BridgeModule(768, r=rank)
    bt = BridgeModule(512, r=rank)
    
    with torch.no_grad():
        zv = bv(v_val).cpu()
        zt = bt(t_train).cpu()
    
    sim = (zv @ zt.t())
    preds = torch.argsort(sim, dim=1, descending=True)
    r1 = (preds[:, :1] == torch.arange(len(zv)).unsqueeze(1)).sum().item() / len(zv)
    
    # Check output statistics
    print("Rank %d: R@1=%.1f%% | out_std=%.4f" % (rank, r1*100, zv.std().item()))

# ── Key question: are val indices matched? ─────────────────────────────
print("\n=== Checking index alignment ===")
# In the retrieval setup, each val image at index i should match val caption at index i
# But we compare val images vs TRAIN captions!
# Check: do val images find their MATCHING train caption or random train caption?

val_labels = [int(idx // 200) for idx in val_idx]   # approximate class labels
train_same_class = []
for vi, vl in enumerate(val_idx):
    v_label = vl // 200
    same_class_train = [ti for ti in train_idx if ti // 200 == v_label]
    train_same_class.append(len(same_class_train))

print("Avg same-class train samples per val image:", np.mean(train_same_class))
print("This affects R@1 - having many same-class captions makes retrieval easier!")

# ── Better evaluation: random text set as negatives ─────────────────────
print("\n=== Evaluation with 1000 random negative captions ===")

# For each val image, compare against its matching train caption + 1000 random train captions
np.random.seed(42)
neg_indices = np.random.choice(train_idx, size=min(1000, len(train_idx)), replace=False)

for vi in range(min(50, len(val_idx))):  # Test first 50 val images
    v_feat = v_val[vi:vi+1]  # [1, 768]
    
    # Positive caption (index-matched train caption)
    pos_caption_idx = val_idx[vi] - train_idx[0] if val_idx[vi] in train_idx else 0
    t_pos = t_train[pos_caption_idx:pos_caption_idx+1]
    
    # Negative captions
    neg_cap_indices = np.random.choice(len(t_train), size=99, replace=False)
    t_negs = t_train[neg_cap_indices]
    
    # All candidates
    all_caps = torch.cat([t_pos, t_negs], dim=0)   # [100, 512]
    
    # Compute similarity
    sim = v_feat @ all_caps.t()   # [1, 100]
    rank = torch.argsort(sim, dim=1, descending=True)[0, 0].item() + 1
    
    if vi < 10:
        print("Val[%d]: caption rank=%d" % (vi, rank))
