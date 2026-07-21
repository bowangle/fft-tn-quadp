#pragma once

#include <cmath>
#include <vector>
#include <complex>
#include <cstddef>

#include "tt_base.h"
#include "type_double_double.h"
#include "type_float128_boost.h"
#include <boost/math/constants/constants.hpp>

namespace magic_tensor_qft {

// ============================================================
// Math helpers — ADL-dispatched so they work for double,
// dd_128 and float128 alike.
// ============================================================
template<typename Real>
Real real_cos(Real x) { using std::cos; return cos(x); }

template<typename Real>
Real real_sin(Real x) { using std::sin; return sin(x); }

template<typename Real>
Real real_sqrt(Real x) { using std::sqrt; return sqrt(x); }

// Primary template — explodes with a clear message for unknown types
template<typename Real>
Real pi() {
    static_assert(sizeof(Real) == 0,
        "pi() not specialised for this type — add a specialisation or include the right type header");
    return Real(0);
}

template<> inline double     pi<double>()     { return M_PI; }
template<> inline dd_128     pi<dd_128>()     { return dd_real::_pi; }
template<> inline float128   pi<float128>()   { return boost::math::constants::pi<float128>(); }

// ============================================================
// chebyshev_nodes_on_01
//   c_a = [1 - cos(pi a / (chi-1))] / 2,   a = 0..chi-1
// ============================================================
template<typename Real>
std::vector<Real> chebyshev_nodes_on_01(int chi) {
    std::vector<Real> nodes(chi);
    Real pi_val = pi<Real>();
    Real denom = Real(chi - 1);
    for (int a = 0; a < chi; ++a) {
        nodes[a] = Real(0.5) * (Real(1) - real_cos(pi_val * Real(a) / denom));
    }
    return nodes;
}

// ============================================================
// lagrange_basis_values
//   Evaluate all Lagrange basis polynomials P_beta(x) for
//   given nodes.  P_beta(x) = Π_{γ≠β} (x - c_γ)/(c_β - c_γ)
// ============================================================
template<typename Real>
std::vector<std::complex<Real>> lagrange_basis_values(
    const std::vector<Real>& nodes, Real x)
{
    using Complex = std::complex<Real>;
    int chi = static_cast<int>(nodes.size());
    std::vector<Complex> P(chi, Complex(1, 0));
    for (int beta = 0; beta < chi; ++beta) {
        Complex denom(1, 0);
        Complex numer(1, 0);
        Real c_beta = nodes[beta];
        for (int gamma = 0; gamma < chi; ++gamma) {
            if (gamma == beta) continue;
            denom *= Complex(c_beta - nodes[gamma]);
            numer *= Complex(x - nodes[gamma]);
        }
        P[beta] = numer / denom;
    }
    return P;
}

// ============================================================
// build_qft_mpo_tensors
//   Construct local MPO tensors W[n] with shape (2,2,chi,chi)
//   and return them as a vector of 4D arrays (flat-packed).
//
//   M^n_{αβ} = exp(-iπ (t_n ω_n + c_α t_n)) * P_β((c_α+ω_n)/2)
// ============================================================
template<typename ComplexT>
std::vector<std::vector<ComplexT>>
build_qft_mpo_tensors(int r, int chi)
{
    using Real = typename ComplexT::value_type;
    auto nodes = chebyshev_nodes_on_01<Real>(chi);
    Real pi_val = pi<Real>();

    // Each W[n] stored as a flat vector of length 2*2*chi*chi,
    // with index = ((tbit*2 + wbit)*chi + alpha)*chi + beta
    std::vector<std::vector<ComplexT>> W(r);

    for (int n = 0; n < r; ++n) {
        auto& Wn = W[n];
        Wn.resize(2 * 2 * chi * chi);

        for (int tbit = 0; tbit < 2; ++tbit) {
            for (int wbit = 0; wbit < 2; ++wbit) {
                Real wbit_r = Real(wbit);
                Real tbit_r = Real(tbit);

                for (int alpha = 0; alpha < chi; ++alpha) {
                    Real c_alpha = nodes[alpha];
                    Real x = Real(0.5) * (c_alpha + wbit_r);
                    auto P = lagrange_basis_values<Real>(nodes, x);
                    Real theta = pi_val * (tbit_r * wbit_r + c_alpha * tbit_r);
                    ComplexT phase(real_cos(theta), -real_sin(theta));

                    for (int beta = 0; beta < chi; ++beta) {
                        std::size_t idx =
                            ((static_cast<std::size_t>(tbit) * 2 + static_cast<std::size_t>(wbit)) * chi
                             + static_cast<std::size_t>(alpha)) * chi
                            + static_cast<std::size_t>(beta);
                        Wn[idx] = phase * P[beta];
                    }
                }
            }
        }
    }
    return W;
}

// ============================================================
// build_mpo
//   Assemble the MPO cores from the pre-computed W tensors
//   into a TT object.
//
//   Core layout (before boundary contraction):
//       core[alpha, phys, beta]  with phys = wbit*2 + tbit
//   After left/right boundary:
//       site 0:     (1, 4, chi)
//       site r-1:   (chi, 4, 1)
//       otherwise:  (chi, 4, chi)
//
//   Overall normalisation 1 / 2^{r/2} is folded into core[0].
// ============================================================
template<typename ComplexT>
TT<ComplexT> build_mpo(
    const std::vector<std::vector<ComplexT>>& W,
    int chi)
{
    using Real = typename ComplexT::value_type;
    int r = static_cast<int>(W.size());

    using VecTT = std::vector<Tensor3D<ComplexT>>;
    VecTT cores;
    cores.reserve(r);

    for (int n = 0; n < r; ++n) {
        const auto& Wn = W[n];

        Eigen::Index left_dim  = (n == 0)     ? 1 : chi;
        Eigen::Index right_dim = (n == r - 1) ? 1 : chi;

        Tensor3D<ComplexT> core(left_dim, 4, right_dim);

        for (int tbit = 0; tbit < 2; ++tbit) {
            for (int wbit = 0; wbit < 2; ++wbit) {
                int phys = wbit * 2 + tbit;

                for (int alpha = 0; alpha < chi; ++alpha) {
                    for (int beta = 0; beta < chi; ++beta) {
                        std::size_t idx =
                            ((static_cast<std::size_t>(tbit) * 2 + static_cast<std::size_t>(wbit)) * chi
                             + static_cast<std::size_t>(alpha)) * chi
                            + static_cast<std::size_t>(beta);
                        ComplexT val = Wn[idx];

                        if (n == 0) {
                            if (alpha == 0)
                                core(0, phys, beta) = val;
                        } else if (n == r - 1) {
                            core(alpha, phys, 0) += val;
                        } else {
                            core(alpha, phys, beta) = val;
                        }
                    }
                }
            }
        }

        cores.push_back(std::move(core));
    }

    // Normalisation:  divide core[0] by 2^{r/2} = (√2)^r
    Real sqrt2 = real_sqrt(Real(2));
    Real norm = Real(1);
    for (int i = 0; i < r; ++i)
        norm *= sqrt2;

    for (auto& x : cores[0].data)
        x /= norm;

    return TT<ComplexT>(std::move(cores));
}

// ============================================================
// build_qft_mpo_magic
//   Convenience:  build W tensors, then assemble the TT MPO.
// ============================================================
template<typename ComplexT>
TT<ComplexT> build_qft_mpo_magic(int r, int chi)
{
    auto W = build_qft_mpo_tensors<ComplexT>(r, chi);
    return build_mpo<ComplexT>(W, chi);
}

} // namespace magic_tensor_qft
