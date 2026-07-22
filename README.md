# fft-tn-quadp

Quad-precision QFT MPO precomputation using tensor networks.

## Quick Start

```bash
# 1. Install external dependencies
./install_extern.sh

# 2. Compile (tests + executable)
./compile.sh

# 3. Run tests
./run_test.sh

# 4. Precompute QFT MPOs
./build/calc_MPO_QFT +double +dd128 +float128 +C max_a=127

# 5. Verify QFT MPO quality (plots)
python src/plot_precompute_qft.py

# Figures are saved to mpo_data/figure/:
#   - precompute_qft_chi_err.*   : chi & relative error vs nBit
#   - precompute_qft_timing.*    : t_construction & t_all vs nBit
```

## Dependencies

- CMake ≥ 3.16
- C++17 compiler (GCC or Clang)
- OpenMP

`install_extern.sh` fetches and builds all external dependencies (Eigen, QD, TensorQuadOperation, etc.) into `extern/`.

## Build Targets

| Target            | Description                          |
|-------------------|--------------------------------------|
| `Test_qft_mpo`    | Unit tests for the QFT MPO builder   |
| `calc_MPO_QFT`    | Precompute MPOs and save to disk     |

## CLI Usage — `calc_MPO_QFT`

```
Usage: calc_MPO_QFT [+double] [+float128] [+dd128] [+all]
                     [+compress|+C] [min_a=N] [max_a=N] [chi=N] [eps_factor=N]
```

### Type toggles (all off by default)

| Flag         | Description            |
|--------------|------------------------|
| `+double`    | `std::complex<double>` |
| `+float128`  | `boost::float128`      |
| `+dd128`     | double-double (qd)     |
| `+all`       | enable all three       |

### Other flags

| Flag              | Default | Description                                |
|-------------------|---------|--------------------------------------------|
| `+compress`, `+C` | off     | enable SVD compression                     |
| `min_a=N`         | 10      | minimum number of bits                     |
| `max_a=N`         | 100     | maximum number of bits                     |
| `chi=N`           | 35      | bond dimension for MPO construction        |
| `eps_factor=N`    | 2       | tolerance = 10^N × ε (used for compression)|

### Examples

```bash
# Precompute with all types, compression, up to 127 bits
./build/calc_MPO_QFT +double +dd128 +float128 +C max_a=127

# Quick run with just double, no compression, 10-30 bits
./build/calc_MPO_QFT +double min_a=10 max_a=30 chi=20
```

## Output

Precomputed MPOs and info files are saved to `mpo_data/data/`. Figures are saved to `mpo_data/figure/`.
