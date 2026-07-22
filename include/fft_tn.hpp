#pragma once

#include <string>

#include <mps_base.h>
#include <mpo_base.h>
#include <grid.h>

template <typename Cscalar, typename Sint>
class FFTmps{
public:
    using RealScalar = typename Eigen::NumTraits<Cscalar>::Real;

    // constructor where we provide everything (mps, grid and mpo)
    FFTmps(
        MPS<Cscalar> mps_,
        QTGrid<RealScalar, Sint> grid_,
        MPO<Cscalar> qft_mpo_,
        RealScalar reltol_ = Eigen::NumTraits<RealScalar>::epsilon() * 100,
        int padding_bit_ = 0,
        int max_chi_ = 200)
    :
    mps(std::move(mps_)),
    grid(std::move(grid_)),
    QFT_E_T(std::move(qft_mpo_)),
    padding_bit(padding_bit_),
    reltol(reltol_),
    max_chi(max_chi_)
    {}
    
    // constructor where we provide path to file:
    // assume mps armadillo file format with .tt extension for tt and grid file format in _grid.tt
    FFTmps(
        const std::string& prefix_mps,
        const std::string& folder_qft_data = "mpo_data/data/",
        RealScalar reltol_ = Eigen::NumTraits<RealScalar>::epsilon() * 100,
        int padding_bit_ = 0,
        int max_chi_ = 200)
    :
    mps(MPS<Cscalar>(prefix_mps + ".tt")),                                                              // load .tt to an mps
    grid(QTGrid<RealScalar, Sint>(prefix_mps + "_grid_E.json")),                                          // load grid
    QFT_E_T(magic_tensor_qft::load_compressed_qft_mpo<Cscalar>(mps.get_size() + padding_bit_, folder_qft_data)),  // load mpo
    padding_bit(padding_bit_),
    reltol(reltol_),
    max_chi(max_chi_)
    {}

private:
    MPS<Cscalar> mps;
    QTGrid<RealScalar, Sint> grid;
    MPO<Cscalar> QFT_E_T;
    int padding_bit;
    RealScalar reltol = Eigen::NumTraits<RealScalar>::epsilon() * 100;
    int max_chi;
};