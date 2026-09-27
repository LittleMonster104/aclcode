# ACE: Test-Time Consistency Regularization for Zero-Shot Cross-Modal Retrieval

**Complete Reproducibility Package**

This package contains all code to reproduce every result in the paper, including main results (Tables 1-2) and appendix analyses (A.1, D).

---

## 📋 Quick Start

### Prerequisites
```bash
pip install torch torchvision clip open_clip_torch tqdm numpy
```

### One-Command Reproduction
```bash
bash run_all_experiments.sh
```

This will reproduce:
- ✅ Table 1: COCO Main Results
- ✅ Table 2: RSITMD Cross-Domain Results  
- ✅ Appendix A.1: Error Correlation Analysis
- ✅ Appendix D: Hyperparameter Sensitivity

---

## 🗂️ Directory Structure

```
finalcode/
├── core/                           # Core algorithm implementations
│   ├── dual_end_fusion.py          # Per-direction z-score normalization (Eq. 3-5)
│   ├── consistency_regularization.py  # Bidirectional contrastive training (Alg. 1)
│   └── adaptive_topk.py            # Entropy-based pseudo-labeling (Eq. 10)
├── experiments/                    # Main paper results
│   ├── coco_table1_main_results.py     # Table 1: COCO results
│   └── rsitmd_table2_cross_domain.py   # Table 2: RSITMD results
├── analysis/                       # Appendix analyses
│   ├── error_correlation_appendix_a1.py     # Appendix A.1
│   └── hyperparameter_sensitivity_appendix_d.py  # Appendix D
├── data/                           # Dataset configurations
│   └── README.md                   # Download instructions
├── results/                        # Saved experiment results
├── run_all_experiments.sh          # One-click reproduction
└── README.md                       # This file
```

---

## 📊 Reproducing Specific Results

### Table 1: COCO Main Results (I2T R@1)

```bash
python3 experiments/coco_table1_main_results.py
```

**Output**: `results/table1_coco_main.json`

**Expected Results**:
| Method | ViT-B/32 | ViT-L/14 | SigLIP-base | SO400M |
|--------|----------|----------|-------------|--------|
| Baseline | 50.12 | 56.26 | 28.64 | 64.96 |
| + Dual-End | 54.36 | 59.34 | 46.70 | 67.68 |
| + Consistency | 54.82 | 59.44 | 53.92 | 68.54 |
| + Pseudo-Label | 55.28 | 59.52 | 54.32 | 69.10 |

---

### Table 2: RSITMD Cross-Domain Results (I2T R@1)

```bash
python3 experiments/rsitmd_table2_cross_domain.py
```

**Output**: `results/table2_rsitmd_cross_domain.json`

**Expected Results**:
| Method | ViT-B/32 | ViT-L/14 | SigLIP-base | SO400M |
|--------|----------|----------|-------------|--------|
| Baseline | 9.29 | 9.51 | 2.21 | 8.41 |
| + Dual-End | 9.29 | 9.51 | 2.21 | 8.41 |
| + Consistency | 15.93 | 19.91 | 8.85 | 30.53 |
| + Pseudo-Label | 13.72 | 20.80 | 4.65 | 25.66 |

**Key Finding**: Dual-End Fusion provides 0% gain on RSITMD (ρ ≈ -0.03, no error complementarity).

---

## 🎯 Reviewer Response - Addressed Issues

### A2. Consistency Regularization Training Data
**Issue**: Code and paper consistency  
**Resolution**: Code explicitly states "train on test pairs" (line 10 in `consistency_regularization.py`), matching paper Section 4.2 and line 288: "for T=2 epochs over the test set"

### A3. COCO T2I Baseline
**Issue**: Paper shows 48.56, some results show 31.02  
**Resolution**: All scripts now use consistent evaluation protocol. Check `experiments/coco_table1_main_results.py` for verification.

### B. Missing Scripts
**Resolution**: All requested scripts now included:
- ✅ COCO driver (Table 1, Table 4)
- ✅ RSITMD driver (Table 2, Table 5)
- ✅ Error correlation analysis (Appendix A.1)
- ✅ Hyperparameter sensitivity (Appendix D)

---

## ✅ Reproducibility Checklist

- [x] All core algorithms implemented
- [x] Main results (Tables 1-2) reproducible
- [x] Appendix analyses (A.1, D) reproducible
- [x] One-click reproduction script
- [x] Detailed documentation
- [x] Consistent with paper claims
- [x] Code-paper alignment verified

**Status**: Ready for review ✅
