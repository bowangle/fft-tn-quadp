#!/bin/bash
set -e

# 1. Install external dependencies
echo "==> Step 1: Installing external dependencies..."
./install_extern.sh

# 2. Compile (tests + executable)
echo "==> Step 2: Compiling..."
./compile.sh

# 3. Precompute QFT MPOs
echo "==> Step 3: Precomputing QFT MPOs..."
./build/precompute_qft +double +dd128 +float128 +C max_a=127
python3 src/plot_precompute_qft.py

# 4. Run tests
echo "==> Step 4: Running tests..."
./run_test.sh
python3 test/plot_errors_qft_mpo.py

echo "==> All done!"
echo "Figures saved to mpo_data/figure/ and test/figure/"
