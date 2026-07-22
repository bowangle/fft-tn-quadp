#pragma once

#include <mps_base.h>
#include <mpo_base.h>
#include <grid.h>

template <typename Cscalar, typename Sint>
class FFTmps{
public:
    using RealScalar = typename Eigen::NumTraits<Scalar>::Real;

    // constructor where we provide everything (mps, grid and mpo)
    FFTmps(
        MPS mps_,
        QTGrid grid_,
        MPO QFT_E_T_,
        RealScalar reltol = Eigen::NumTraits<RealScalar>::epsilon() * 100,
        int padding_bit_ = 0,
        int max_chi_ = 200)
    :
    mps(mps_),
    grid(grid_),
    QFT_E_T(QFT_E_T_),
    padding_bit(padding_bit_),
    reltol(reltol_),
    max_chi(max_chi_)
    {}
    
    // constructor where we provide path to file:
    // assume mps armadillo file format with .tt extension for tt and grid file format in _grid.tt
    FFTmps(
        const std::string& prefix_mps_,
        MPO QFT_E_T_,
        RealScalar reltol = Eigen::NumTraits<RealScalar>::epsilon() * 100,
        int padding_bit_ = 0,
        int max_chi_ = 200)
    :
    mps(mps_),
    grid(grid_),
    QFT_E_T(QFT_E_T_),
    padding_bit(padding_bit_),
    reltol(reltol_),
    max_chi(max_chi_)
    {}

private:
    MPS mps;
    QTGrid grid;
    MPO QFT_E_T;
    int padding_bit;
    RealScalar reltol = Eigen::NumTraits<RealScalar>::epsilon() * 100;
    int max_chi;
};