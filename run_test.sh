#!/bin/bash

set -u

enable_qrsvd=false
enable_float128=false

usage() {
    cat <<'EOF'
Usage: ./run_test.sh [options]

Options:
  --qrsvd     Include QR-SVD cases in the DD128 and float128 FFT tests.
  --float128  Run the float128 FFT test (disabled by default because it is slow).
  -h, --help  Show this help message.
EOF
}

while (($# > 0)); do
    case "$1" in
        --qrsvd)
            enable_qrsvd=true
            ;;
        --float128)
            enable_float128=true
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

rm -f \
    test/test_qft_mpo.txt \
    test/test_fft_vanilla.txt \
    test/test_fft_double.txt \
    test/test_fft_convergence_double.txt \
    test/test_fft_convergence_dd128.txt \
    test/test_fft_dd128.txt \
    test/test_fft_float128.txt

status=0
run_and_record() {
    local output_file=$1
    shift

    echo "Running $*"
    if "$@" >"$output_file" 2>&1; then
        echo "  passed; output saved to $output_file"
    else
        local test_status=$?
        echo "  failed with status $test_status; output saved to $output_file" >&2
        status=1
    fi
}

# Run serially so the MPO-MPS timing summaries are not distorted by tests
# competing for CPU time and memory bandwidth.
run_and_record test/test_qft_mpo.txt \
    ./build/fft_tn_test_magic_qft_mpo
run_and_record test/test_fft_vanilla.txt \
    ./build/fft_tn_test_fft_vanilla
run_and_record test/test_fft_double.txt \
    ./build/fft_tn_test_fft_double
run_and_record test/test_fft_convergence_double.txt \
    ./build/fft_tn_test_fft_convergence_double
run_and_record test/test_fft_convergence_dd128.txt \
    ./build/fft_tn_test_fft_convergence_dd128

qrsvd_args=()
if $enable_qrsvd; then
    qrsvd_args+=(--qrsvd)
fi

run_and_record test/test_fft_dd128.txt \
    ./build/fft_tn_test_fft_dd128 "${qrsvd_args[@]}"

if $enable_float128; then
    run_and_record test/test_fft_float128.txt \
        ./build/fft_tn_test_fft_float128 "${qrsvd_args[@]}"
else
    echo "Skipping float128 FFT test (use --float128 to enable it)."
fi

if ((status == 0)); then
    echo "All requested tests passed. Output saved to individual files."
else
    echo "One or more tests failed. See the individual output files." >&2
fi

exit "$status"
