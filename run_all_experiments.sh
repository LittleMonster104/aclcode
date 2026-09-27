#!/bin/bash
# One-click script to reproduce all paper results
# Run this after extracting features for all datasets and models

set -e

echo "=========================================="
echo "ACE Paper - Complete Reproducibility"
echo "=========================================="
echo ""

# Check Python environment
if ! command -v python3 &> /dev/null; then
    echo "❌ python3 not found. Please install Python 3.7+"
    exit 1
fi

echo "✓ Python environment found"
echo ""

# Create results directory
mkdir -p results

# Main results
echo "=========================================="
echo "1. Main Results"
echo "=========================================="
echo ""

echo "[1/2] Table 1 - COCO Main Results..."
python3 experiments/coco_table1_main_results.py
echo ""

echo "[2/2] Table 2 - RSITMD Cross-Domain Results..."
python3 experiments/rsitmd_table2_cross_domain.py
echo ""

# Analysis
echo "=========================================="
echo "2. Appendix Analysis"
echo "=========================================="
echo ""

echo "[1/2] Appendix A.1 - Error Correlation Analysis..."
python3 analysis/error_correlation_appendix_a1.py
echo ""

echo "[2/2] Appendix D - Hyperparameter Sensitivity..."
python3 analysis/hyperparameter_sensitivity_appendix_d.py
echo ""

# Summary
echo "=========================================="
echo "✓ All experiments completed!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - results/table1_coco_main.json"
echo "  - results/table2_rsitmd_cross_domain.json"
echo "  - results/appendix_a1_error_correlation.json"
echo "  - results/appendix_d_hyperparameter_sensitivity.json"
echo ""
echo "To reproduce specific tables:"
echo "  python3 experiments/coco_table1_main_results.py"
echo "  python3 experiments/rsitmd_table2_cross_domain.py"
echo ""
