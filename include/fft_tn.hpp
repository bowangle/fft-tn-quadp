#pragma once

#include <string>
#include <cmath>

#include <mps_base.h>
#include <mpo_base.h>
#include <grid.h>
#include "magic_tensor_qft.hpp"

template <typename Cscalar, typename Sint>
class FFTmps{
public:
    using RealScalar = typename Eigen::NumTraits<Cscalar>::Real;

    // constructor where we provide everything (mps, grid and mpo)
    FFTmps(
        MPS<Cscalar> mps_,
        QTGrid<RealScalar, Sint> grid_,
        MPO<Cscalar> qft_mpo_,
        int padding_bit_ = 0,
        int max_chi_ = 200,
        RealScalar reltol_ = Eigen::NumTraits<RealScalar>::epsilon() * 100)
    :
    mps(std::move(mps_)),
    grid(std::move(grid_)),
    QFT_E_T(std::move(qft_mpo_)),
    padding_bit(padding_bit_),
    reltol(reltol_),
    max_chi(max_chi_)
    {
        _add_n_padding(mps, grid, padding_bit);
    }
    
    // constructor where we provide path to file:
    // assume mps armadillo file format with .tt extension for tt and grid file format in _grid.tt
    FFTmps(
        const std::string& prefix_mps,
        const std::string& folder_qft_data = "mpo_data/data/",
        int padding_bit_ = 0,
        int max_chi_ = 200,
        RealScalar reltol_ = Eigen::NumTraits<RealScalar>::epsilon() * 100)
    :
    mps(MPS<Cscalar>(prefix_mps + ".tt", max_chi_, reltol_, /*w_=*/-1)),
    grid(QTGrid<RealScalar, Sint>(prefix_mps + "_grid_E.json")),
    QFT_E_T(magic_tensor_qft::load_compressed_qft_mpo<Cscalar>(mps.get_size() + padding_bit_, folder_qft_data, reltol_, max_chi_)),
    padding_bit(padding_bit_),
    reltol(reltol_),
    max_chi(max_chi_)
    {
        _add_n_padding(mps, grid, padding_bit);
    }

    // --- reverse cores: swap left↔right on each core, then reverse order ---
    // TODO we need to move this to TensorQuadOperation
    static std::vector<Tensor3D<Cscalar>> reverse_cores(const std::vector<Tensor3D<Cscalar>>& cores)
    {
        std::vector<Tensor3D<Cscalar>> out;
        out.reserve(cores.size());
        for (int i = static_cast<int>(cores.size()) - 1; i >= 0; --i) {
            auto const& src = cores[i];
            Tensor3D<Cscalar> dst(src.n_right, src.n_phys, src.n_left);
            for (Eigen::Index l = 0; l < src.n_left; ++l)
                for (Eigen::Index p = 0; p < src.n_phys; ++p)
                    for (Eigen::Index r = 0; r < src.n_right; ++r)
                        dst(r, p, l) = src(l, p, r);
            out.push_back(std::move(dst));
        }
        return out;
    }

    // --- vanilla FFT (no trapezoid rule) ---
    void fft_vanilla(bool do_shift = true)
    {
        // 1. Reverse cores for endian consistency with the QFT MPO
        mps = MPS<Cscalar>(reverse_cores(mps.get_core()), mps.get_max_bond_dim(), mps.get_reltol(), mps.get_w());

        // 2. Apply QFT MPO
        mps = QFT_E_T._mul(mps);

        // 3. Normalisation: *√N
        mps *= magic_tensor_qft::real_sqrt(RealScalar(grid.get_N()));

        // 4. Phase factor: *dx / (2π)
        mps *= grid.get_dx() / (RealScalar(2) * pi<RealScalar>());

        // 5. Time-dependent phase: exp(-1j * t * a)
        RealScalar dt = RealScalar(2) * pi<RealScalar>() / (RealScalar(grid.get_N()) * grid.get_dx());
        auto mps_phase = _phase_mps_exp_1j_t_a(grid.get_nBits(), grid.get_a(), dt);
        mps_phase = MPS<Cscalar>(mps_phase.get_core(), max_chi, reltol, mps_phase.get_w());
        auto phase_mpo = MPO<Cscalar>::from_mps(mps_phase);
        mps = phase_mpo._mul(mps);

        // 6. Optional fftshift
        if (do_shift)
            mps = _fft_shift(mps);

        // 7. Update grid to dual grid
        grid = grid.build_dual_grid(do_shift);
    }

    Eigen::Index get_chi() const { return mps.get_chi(); }

private:
    MPS<Cscalar> mps;
    QTGrid<RealScalar, Sint> grid;
    MPO<Cscalar> QFT_E_T;
    int padding_bit;
    RealScalar reltol = Eigen::NumTraits<RealScalar>::epsilon() * 100;
    int max_chi;

    // --- padding -----------------------------------------------------------
    static void _add_one_slow_bit(MPS<Cscalar>& mps, QTGrid<RealScalar, Sint>& grid)
    {
        auto cores = mps.get_core();
        Eigen::Index chi_right = cores.back().n_right;
        Tensor3D<Cscalar> new_core(chi_right, 2, 1);
        for (Eigen::Index r = 0; r < chi_right; ++r)
            new_core(r, 0, 0) = Cscalar(1, 0);
        cores.push_back(std::move(new_core));
        mps = MPS<Cscalar>(std::move(cores), mps.get_max_bond_dim(), mps.get_reltol(), mps.get_w());
        grid.update_padding_1h_bit();
    }

    static void _add_n_padding(MPS<Cscalar>& mps, QTGrid<RealScalar, Sint>& grid, int n)
    {
        for (int i = 0; i < n; ++i)
            _add_one_slow_bit(mps, grid);
    }

    // --- fftshift: swap |0⟩ ↔ |1⟩ on the last (MSB) core ---
    static MPS<Cscalar> _fft_shift(MPS<Cscalar> const& mps)
    {
        auto cores = mps.get_core();
        auto& last = cores.back();
        Tensor3D<Cscalar> swapped(last.n_left, last.n_phys, last.n_right);
        for (Eigen::Index l = 0; l < last.n_left; ++l)
            for (Eigen::Index r = 0; r < last.n_right; ++r) {
                swapped(l, 0, r) = last(l, 1, r);
                swapped(l, 1, r) = last(l, 0, r);
            }
        cores.back() = std::move(swapped);
        return MPS<Cscalar>(std::move(cores), mps.get_max_bond_dim(), mps.get_reltol(), mps.get_w());
    }

    // --- phase MPS: exp(-1j * t_m * a) ------------------------------------
    static MPS<Cscalar> _phase_mps_exp_1j_t_a(int nBits, RealScalar a, RealScalar dt)
    {
        std::vector<Tensor3D<Cscalar>> cores;
        cores.reserve(nBits);
        RealScalar theta_factor = a * dt;
        RealScalar pow2 = RealScalar(1);
        for (int k = 0; k < nBits; ++k) {
            RealScalar theta = theta_factor * pow2;
            pow2 *= RealScalar(2);
            int sign = (k < nBits - 1) ? -1 : 1;
            Tensor3D<Cscalar> core(1, 2, 1);
            core(0, 0, 0) = Cscalar(1, 0);
            core(0, 1, 0) = std::exp(Cscalar(0, (sign == 1) ? theta : -theta));
            cores.push_back(std::move(core));
        }
        return MPS<Cscalar>(std::move(cores));
    }

};
