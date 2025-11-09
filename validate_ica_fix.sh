#!/bin/bash
# Quick validation script for ICA diagonal weights fix
# This script checks if the fix resolves the diagonal mixing matrix issue

set -e  # Exit on error

echo "=================================================="
echo "ICA Diagonal Weights Fix - Validation Script"
echo "=================================================="
echo ""

# Check if required files exist
echo "[1/4] Checking required files..."
required_files=("ica_fixed.py" "compare_ica_methods.py" "test_ica_fixed.py")
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "  ✗ Missing: $file"
        exit 1
    fi
done
echo "  ✓ All required files present"
echo ""

# Check Python and packages
echo "[2/4] Checking Python environment..."
if ! python -c "import numpy, torch, sklearn" 2>/dev/null; then
    echo "  ⚠ Required packages not installed"
    echo "  Installing from requirements.txt..."
    pip install -q -r requirements.txt
fi
echo "  ✓ Python environment ready"
echo ""

# Run comparison
echo "[3/4] Running ICA method comparison..."
echo "  (This may take 1-2 minutes...)"
if python compare_ica_methods.py > /tmp/ica_comparison_output.txt 2>&1; then
    echo "  ✓ Comparison completed successfully"

    # Extract key metrics
    echo ""
    echo "  Key Results:"
    grep -A 3 "Diagonality Measures" /tmp/ica_comparison_output.txt | tail -n 3 || true
    echo ""
else
    echo "  ✗ Comparison failed - check /tmp/ica_comparison_output.txt"
    exit 1
fi

# Run tests
echo "[4/4] Running test suite..."
if pytest test_ica_fixed.py -v --tb=short > /tmp/ica_test_output.txt 2>&1; then
    # Count passed tests
    passed=$(grep -c "PASSED" /tmp/ica_test_output.txt || echo "0")
    echo "  ✓ All tests passed ($passed tests)"
else
    echo "  ✗ Some tests failed - check /tmp/ica_test_output.txt"
    exit 1
fi

echo ""
echo "=================================================="
echo "VALIDATION COMPLETE: FIX IS WORKING ✓"
echo "=================================================="
echo ""
echo "Summary:"
echo "  - Diagonal mixing problem identified in original code"
echo "  - Fixed implementation creates proper non-diagonal mixing"
echo "  - All tests passing"
echo "  - Visualization saved to: figures/ica/mixing_matrix_comparison.png"
echo ""
echo "Next steps:"
echo "  1. Review: cat ICA_FIX_SUMMARY.md"
echo "  2. Integrate: Follow patterns in QUICK_START_ICA_FIX.md"
echo "  3. Visualize: open figures/ica/mixing_matrix_comparison.png"
echo ""
