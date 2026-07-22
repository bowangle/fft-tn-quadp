#!/bin/bash
# Remove existing test files (if any)
rm -f test/test_qft_mpo.txt test/test_fft_vanilla.txt

# Run tests in parallel and redirect output
./build/Test_qft_mpo       > test/test_qft_mpo.txt       2>&1 &
./build/Test_fft_vanilla   > test/test_fft_vanilla.txt   2>&1 &

# Wait for all background processes to finish
wait
echo "All tests done. Output saved to individual files."
