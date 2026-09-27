#!/usr/bin/env python3
"""
RSITMD Cross-Domain Results - Table 2 in Paper
Reproduces I2T R@1 results on RSITMD for all 4 models
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from pathlib import Path
import json
import numpy as np
from core.dual_end_fusion import compute_dual_end_fusion, compute_error_correlation
from core.consistency_regularization import train_consistency_projector, apply_consistency_adaptation
from core.adaptive_topk import adaptive_pseudo_labeling, refine_projector_with_pseudo_labels

device = "cuda" if torch.cuda.is_available() else "cpu"

# Data paths
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "../data"))
RSITMD_DIR = DATA_ROOT / "rsitmd"

def load_rsitmd_features(model_name):
    """Load precomputed RSITMD features"""
    features_path = RSITMD_DIR / f"features_{model_name}.pt"
    if features_path.exists():
        data = torch.load(features_path)
        return data['image_features'], data['text_features']
    else:
        print(f"⚠️  Features not found: {features_path}")
        print("   Please run: python extract_rsitmd_features.py first")
        return None, None

def evaluate_retrieval(img_features, txt_features):
    """Compute I2T R@1"""
    N = img_features.shape[0]
    similarity = img_features @ txt_features.T
    
    i2t_ranks = []
    for i in range(N):
        sorted_indices = torch.argsort(similarity[i], descending=True)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item()
        i2t_ranks.append(rank)
    
    i2t_ranks = np.array(i2t_ranks)
    i2t_r1 = (i2t_ranks < 1).mean() * 100
    
    return {'I2T_R@1': i2t_r1}

def run_ace_pipeline_rsitmd(img_features, txt_features):
    """Run ACE pipeline on RSITMD"""
    results = {}
    
    # Baseline
    print(f"\n1. Baseline")
    results['baseline'] = evaluate_retrieval(img_features, txt_features)
    print(f"   I2T R@1: {results['baseline']['I2T_R@1']:.2f}%")
    
    # Step 1: Dual-End Fusion
    print(f"\n2. + Dual-End Fusion")
    fused_sim, i2t_scores, t2i_scores = compute_dual_end_fusion(
        img_features, txt_features, alpha=0.5
    )
    # Check error correlation
    ground_truth = [(i, i) for i in range(len(img_features))]
    rho = compute_error_correlation(i2t_scores, t2i_scores, ground_truth, k=10)
    print(f"   Error correlation ρ = {rho:.3f}")
    
    results['dual_end'] = evaluate_retrieval(img_features, txt_features)
    print(f"   I2T R@1: {results['dual_end']['I2T_R@1']:.2f}%")
    print(f"   Δ = {results['dual_end']['I2T_R@1'] - results['baseline']['I2T_R@1']:.2f}%")
    
    # Step 2: Consistency Regularization
    print(f"\n3. + Consistency Regularization")
    projector = train_consistency_projector(
        img_features, txt_features,
        num_epochs=2, batch_size=32, lr=1e-3, device=device
    )
    img_adapted, txt_adapted = apply_consistency_adaptation(
        img_features, txt_features, projector, device
    )
    results['consistency'] = evaluate_retrieval(img_adapted, txt_adapted)
    print(f"   I2T R@1: {results['consistency']['I2T_R@1']:.2f}%")
    gain = results['consistency']['I2T_R@1'] - results['dual_end']['I2T_R@1']
    print(f"   Δ = +{gain:.2f}%")
    
    # Step 3: Adaptive Pseudo-Labeling
    print(f"\n4. + Pseudo-Labeling")
    selected_idx, pseudo_labels, ratio = adaptive_pseudo_labeling(
        img_adapted, txt_adapted, projector, device=device
    )
    projector_refined = refine_projector_with_pseudo_labels(
        img_adapted, txt_adapted, projector,
        selected_idx, pseudo_labels, num_epochs=1, device=device
    )
    img_final, txt_final = apply_consistency_adaptation(
        img_features, txt_features, projector_refined, device
    )
    results['pseudo_label'] = evaluate_retrieval(img_final, txt_final)
    print(f"   I2T R@1: {results['pseudo_label']['I2T_R@1']:.2f}%")
    gain = results['pseudo_label']['I2T_R@1'] - results['consistency']['I2T_R@1']
    print(f"   Δ = {gain:+.2f}%")
    
    return results

def main():
    print("="*60)
    print("RSITMD Cross-Domain Results - Table 2")
    print("="*60)
    
    models = {
        'ViT-B/32': 'vitb32',
        'ViT-L/14': 'vitl14',
        'SigLIP-base': 'siglip_base',
        'SigLIP-SO400M': 'siglip_so400m'
    }
    
    all_results = {}
    
    for model_display, model_key in models.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_display}")
        print(f"{'='*60}")
        
        img_feat, txt_feat = load_rsitmd_features(model_key)
        if img_feat is None:
            print(f"⚠️  Skipping {model_display} (features not found)")
            continue
        
        results = run_ace_pipeline_rsitmd(img_feat, txt_feat)
        all_results[model_display] = results
    
    # Print Table 2
    print("\n" + "="*60)
    print("Table 2: RSITMD Results (I2T R@1)")
    print("="*60)
    print(f"{'Method':<20} {'ViT-B/32':<12} {'ViT-L/14':<12} {'SigLIP-base':<12} {'SO400M':<12}")
    print("-"*80)
    
    for method in ['baseline', 'dual_end', 'consistency', 'pseudo_label']:
        method_name = {
            'baseline': 'Baseline',
            'dual_end': '+ Dual-End',
            'consistency': '+ Consistency',
            'pseudo_label': '+ Pseudo-Label'
        }[method]
        
        row = [method_name]
        for model in models.keys():
            if model in all_results and method in all_results[model]:
                val = all_results[model][method]['I2T_R@1']
                row.append(f"{val:.2f}")
            else:
                row.append("N/A")
        
        print(f"{row[0]:<20} {row[1]:<12} {row[2]:<12} {row[3]:<12} {row[4]:<12}")
    
    # Save results
    output_file = Path("results/table2_rsitmd_cross_domain.json")
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Results saved to: {output_file}")

if __name__ == "__main__":
    main()
