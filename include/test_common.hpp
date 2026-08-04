#pragma once

#include <cmath>
#include <complex>
#include <functional>
#include <string_view>
#include <vector>
#include <string>
#include <tuple>
#include <sstream>
#include <fstream>

#include <xf-qd-runner.hpp>
#include <grid.h>
#include <random>
#include <filesystem>

#include "magic_tensor_qft.hpp"

namespace fft_tn_test {

// One entry from the function battery used by the FFT tests.  Real is the
// coordinate type (double, dd_128 or float128); the function is complex-valued
// so that the same interface also covers the Lorentzian test case.
template <typename ComplexT>
struct FFTTestFunction {

    using RealT = typename Eigen::NumTraits<ComplexT>::Real;

    using Function = std::function<ComplexT(RealT)>;

    std::string_view name;
    Function function;
    RealT a_E;
    RealT b_E;
    std::vector<RealT> E_discontinuity;

    ComplexT operator()(RealT x) const { return function(x); }
};

namespace detail {

// Keep calls unqualified after importing std overloads: dd_128 and float128
// then select their overloads through argument-dependent lookup.
template <typename Real>
Real cos_value(const Real& x)
{
    using std::cos;
    return cos(x);
}

template <typename Real>
Real exp_value(const Real& x)
{
    using std::exp;
    return exp(x);
}

template <typename Real>
Real abs_value(const Real& x)
{
    using std::abs;
    return abs(x);
}

} // namespace detail

// The functions below mirror TEST_FUNCTIONS in syst_1/test/test_fft_mpo.py.
// They are kept as named functions in addition to being collected in the
// battery, which makes it possible for focused tests to use one directly.

template <typename ComplexT>
ComplexT cos_simple(typename Eigen::NumTraits<ComplexT>::Real x)
{
    using Real = typename Eigen::NumTraits<ComplexT>::Real;
    return {detail::cos_value(x), Real(0)};
}

template <typename ComplexT>
ComplexT cos_high_freq(typename Eigen::NumTraits<ComplexT>::Real x)
{
    using Real = typename Eigen::NumTraits<ComplexT>::Real;
    return {detail::cos_value(Real(5) * x), Real(0)};
}

template <typename ComplexT>
ComplexT sum_of_cos(typename Eigen::NumTraits<ComplexT>::Real x)
{
    using Real = typename Eigen::NumTraits<ComplexT>::Real;
    const Real two_pi = Real(2) * magic_tensor_qft::pi<Real>();
    return {detail::cos_value(x) + detail::cos_value(two_pi * x), Real(0)};
}

template <typename ComplexT>
ComplexT sum_of_cos_three(typename Eigen::NumTraits<ComplexT>::Real x)
{
    using Real = typename Eigen::NumTraits<ComplexT>::Real;
    const Real value =
        detail::cos_value(x / Real(2))
        + detail::cos_value(Real(2) * x) / Real(2)
        + detail::cos_value(Real(4) * x) / Real(4);
    return {value, Real(0)};
}

template <typename ComplexT>
ComplexT gaussian(typename Eigen::NumTraits<ComplexT>::Real x)
{
    using Real = typename Eigen::NumTraits<ComplexT>::Real;
    return {detail::exp_value(-(x * x) / Real(2)), Real(0)};
}

template <typename ComplexT>
ComplexT gaussian_shifted(typename Eigen::NumTraits<ComplexT>::Real x)
{
    using Real = typename Eigen::NumTraits<ComplexT>::Real;
    const Real shifted = x - Real(2);
    return {detail::exp_value(-(shifted * shifted) / Real(2)), Real(0)};
}

template <typename ComplexT>
ComplexT exp_decay(typename Eigen::NumTraits<ComplexT>::Real x)
{
    using Real = typename Eigen::NumTraits<ComplexT>::Real;
    return {detail::exp_value(-detail::abs_value(x)), Real(0)};
}

template <typename ComplexT>
ComplexT cos_times_gaussian(typename Eigen::NumTraits<ComplexT>::Real x)
{
    using Real = typename Eigen::NumTraits<ComplexT>::Real;
    const Real value = detail::cos_value(Real(2) * x)
                     * detail::exp_value(-(x * x) / Real(2));
    return {value, Real(0)};
}

template <typename ComplexT>
ComplexT cos_times_exp_decay(typename Eigen::NumTraits<ComplexT>::Real x)
{
    using Real = typename Eigen::NumTraits<ComplexT>::Real;
    const Real value = detail::cos_value(x)
                     * detail::exp_value(-detail::abs_value(x));
    return {value, Real(0)};
}

template <typename ComplexT>
ComplexT sum_cos_gaussian(typename Eigen::NumTraits<ComplexT>::Real x)
{
    using Real = typename Eigen::NumTraits<ComplexT>::Real;
    const Real two_pi = Real(2) * magic_tensor_qft::pi<Real>();
    const Real value = (detail::cos_value(x) + detail::cos_value(two_pi * x))
                     * detail::exp_value(-(x * x) / Real(2));
    return {value, Real(0)};
}

template <typename ComplexT>
ComplexT lorentzian(typename Eigen::NumTraits<ComplexT>::Real x)
{
    using Real = typename Eigen::NumTraits<ComplexT>::Real;
    return ComplexT(Real(1), Real(0))
         / ComplexT(x - Real(1) / Real(2), Real(1) / Real(10));
}

template <typename ComplexT>
std::vector<FFTTestFunction<ComplexT>> make_fft_test_functions()
{
    using Real = typename Eigen::NumTraits<ComplexT>::Real;
    const Real eps = Eigen::NumTraits<Real>::epsilon() * Real(10);
    const Real a_E = Real(-10) + eps;
    const Real b_E = Real(10) - eps;
    const std::vector<Real> E_discontinuity = {Real(-1.5), Real(7.2)};

    return {
        {"cos_simple",            cos_simple<ComplexT>,            a_E, b_E, E_discontinuity},
        {"cos_high_freq",         cos_high_freq<ComplexT>,         a_E, b_E, E_discontinuity},
        {"sum_of_cos",            sum_of_cos<ComplexT>,            a_E, b_E, E_discontinuity},
        {"sum_of_cos_three",      sum_of_cos_three<ComplexT>,      a_E, b_E, E_discontinuity},
        {"gaussian",              gaussian<ComplexT>,              a_E, b_E, E_discontinuity},
        {"gaussian_shifted",      gaussian_shifted<ComplexT>,      a_E, b_E, E_discontinuity},
        {"exp_decay",             exp_decay<ComplexT>,             a_E, b_E, E_discontinuity},
        {"cos_times_gaussian",    cos_times_gaussian<ComplexT>,    a_E, b_E, E_discontinuity},
        {"cos_times_exp_decay",   cos_times_exp_decay<ComplexT>,   a_E, b_E, E_discontinuity},
        {"sum_cos_gaussian",      sum_cos_gaussian<ComplexT>,      a_E, b_E, E_discontinuity},
        {"lorentzian",            lorentzian<ComplexT>,            a_E, b_E, E_discontinuity},
    };
}

// Construct each typed battery only once per program.
template <typename ComplexT>
const std::vector<FFTTestFunction<ComplexT>>& fft_test_functions()
{
    static const auto functions = make_fft_test_functions<ComplexT>();
    return functions;
}

template <typename ComplexT, typename Sint>
void tci_function_for_test(
    const std::string& path,
    const std::string& type_name,
    FFTTestFunction<ComplexT> function_1,
    int nBit_,
    int nIter_,
    bool fullPiv_= true,
    CacheLevel cache = CacheLevel::runner)
{
    // use xfac runner on the function. 

    using RealT = typename Eigen::NumTraits<ComplexT>::Real;
    int kBondDim = 100;
    int kNbPointRes = 1000;

    // ---- grid ----
    QTGrid<RealT, Sint> grid(function_1.a_E, function_1.b_E, nBit_);

    // ---- 10 random additional pivots with fixed seed ----
    std::mt19937 rng(42);
    std::uniform_real_distribution<RealT> dist(function_1.a_E, function_1.b_E);
    std::vector<RealT> additional_pivot(10);
    for (auto& p : additional_pivot) p = dist(rng);

    // ---- a non-default pivot1, to verify propagation ----
    std::vector<int> pivot1_custom = grid.coord_to_id(RealT(0.0));

    const std::vector<RealT>& E_discontinuity = function_1.E_discontinuity;

    TCI2_1D_runner_opts<ComplexT> opts{
        .reltol  = Eigen::NumTraits<RealT>::epsilon() * RealT(100),
        .pivot1  = pivot1_custom,
        .fullPiv = fullPiv_,
        .cache   = cache
    };

    TCI2_1D_runner_param<ComplexT> param(nBit_, nIter_, kBondDim, opts);

    TCI2_1D_Runner<ComplexT, Sint> runner(grid, param, function_1.function);

    // resolve the prefix
    const std::string prefix = path
                             + std::string("/")
                             + type_name
                             + std::string("/")
                             + std::string(function_1.name)
                             + std::string("_nB")
                             + std::to_string(nBit_);

    std::filesystem::create_directories(
        std::filesystem::path(prefix).parent_path());

    runner.fit(additional_pivot, /*verbose=*/true, /*do_save=*/true, prefix, kNbPointRes, E_discontinuity);
}

// Compare two MPS on the same grid, evaluating at nb_point points.
// If use_logspace is true, points are distributed logarithmically (requires a > 0).
template <typename ComplexT, typename Sint>
TTErrorOnGrid<typename Eigen::NumTraits<ComplexT>::Real> error_TT_on_grid_point(
    const MPS<ComplexT>& tt_ref,
    const MPS<ComplexT>& tt,
    const QTGrid<typename Eigen::NumTraits<ComplexT>::Real, Sint>& grid,
    int nb_point,
    bool use_logspace = false)
{
    using Real = typename Eigen::NumTraits<ComplexT>::Real;

    TTErrorOnGrid<Real> out;
    out.real_abs.resize(nb_point);
    out.real_rel.resize(nb_point);
    out.imag_abs.resize(nb_point);
    out.imag_rel.resize(nb_point);
    out.abs_abs.resize(nb_point);
    out.abs_rel.resize(nb_point);

    std::vector<Real> l_point(nb_point);

    if (use_logspace) {
        Real log_a = std::log(grid.get_a());
        Real log_b = std::log(grid.get_b());
        for (int i = 0; i < nb_point; ++i) {
            Real t = Real(i) / Real(nb_point - 1);
            l_point[i] = std::exp(log_a + t * (log_b - log_a));
        }
    } else {
        l_point = linspace(grid.get_a(), grid.get_b(), nb_point);
    }

    std::vector<ComplexT> l_ref_i(nb_point);
    std::vector<ComplexT> l_res_tt(nb_point);

    // Build the list of grid IDs once
    std::vector<std::vector<int>> ids(nb_point);
    for (int i = 0; i < nb_point; ++i)
        ids[i] = grid.coord_to_id(l_point[i]);

    // Parallel evaluation of both MPS
    l_ref_i = tt_ref.eval_list(ids);
    l_res_tt = tt.eval_list(ids);

    for (int i = 0; i < nb_point; i++) {
        ComplexT ref_i = l_ref_i[i];
        ComplexT res_tt = l_res_tt[i];
        ComplexT diff = ref_i - res_tt;

        Real abs_diff = std::abs(diff);
        Real ref_norm = std::abs(ref_i);

        out.real_abs[i] = diff.real();
        out.imag_abs[i] = diff.imag();
        out.abs_abs[i]  = abs_diff;

        if (ref_norm != Real(0)) {
            out.real_rel[i] = diff.real() / ref_norm;
            out.imag_rel[i] = diff.imag() / ref_norm;
            out.abs_rel[i]  = abs_diff / ref_norm;
        } else {
            out.real_rel[i] = 0;
            out.imag_rel[i] = 0;
            out.abs_rel[i]  = 0;
        }
    }
    out.l_point = l_point;
    out.l_ref = l_ref_i;
    out.l_tt = l_res_tt;

    out.compute_summary();
    return out;
}

// Data saved by tci_function_for_test for one function.
template <typename ComplexT, typename Sint>
struct FFTTestData {
    using Real = typename Eigen::NumTraits<ComplexT>::Real;

    MPS<ComplexT> mps;
    QTGrid<Real, Sint> grid;
    ComplexT f_b;
    std::vector<Real> discontinuities;
    std::vector<ComplexT> f_discontinuities;
};

// Load previously saved MPS, grid, and function boundary values.
// Reconstructs the same file prefix that tci_function_for_test used:
//     path/type_name/function_name_nB<nBit>

template <typename ComplexT, typename Sint>
FFTTestData<ComplexT, Sint>
load_mps_and_grid(
    const std::string& path,
    const std::string& type_name,
    const std::string& function_name,
    int nBit)
{
    using Real = typename Eigen::NumTraits<ComplexT>::Real;

    const std::string prefix = path
                             + std::string("/")
                             + type_name
                             + std::string("/")
                             + function_name
                             + std::string("_nB")
                             + std::to_string(nBit);

    FFTTestData<ComplexT, Sint> data{
        MPS<ComplexT>(prefix + ".tt"),
        QTGrid<Real, Sint>(prefix + "_grid_E.json"),
        {}, {}, {}
    };

    std::tie(data.f_b, data.discontinuities, data.f_discontinuities) =
        load_fvalues_from_json<ComplexT>(prefix + "_f_values.json");

    return data;
}

// Verify loaded data against the original function.
template <typename ComplexT, typename Sint>
void verify_loaded_data(
    const FFTTestData<ComplexT, Sint>& data,
    const FFTTestFunction<ComplexT>& f,
    int nBit)
{
    using Real = typename Eigen::NumTraits<ComplexT>::Real;

    auto mismatch = [&](const std::string& what, auto expected, auto actual) {
        auto fmt = [](const auto& v) {
            std::ostringstream oss;
            oss << std::setprecision(std::numeric_limits<Real>::max_digits10)
                << std::scientific;
            if constexpr (Eigen::NumTraits<decltype(v)>::IsComplex != 0)
                oss << v.real() << "+" << v.imag() << "j";
            else
                oss << v;
            return oss.str();
        };
        std::cerr << "[" << f.name << "] " << what << " mismatch:\n"
                  << "  expected: " << fmt(expected) << "\n"
                  << "  actual:   " << fmt(actual) << std::endl;
        throw std::runtime_error(std::string(f.name) + ": " + what + " mismatch");
    };

    // Grid parameters
    if (data.grid.get_a() != f.a_E)
        mismatch("grid a", f.a_E, data.grid.get_a());
    if (data.grid.get_b() != f.b_E)
        mismatch("grid b", f.b_E, data.grid.get_b());
    if (data.grid.get_nBits() != nBit)
        mismatch("grid nBits", nBit, data.grid.get_nBits());

    // f_value at right boundary
    ComplexT expected_f_b = f.function(f.b_E);
    if (data.f_b != expected_f_b)
        mismatch("f_b", expected_f_b, data.f_b);

    // f_discontinuities
    if (data.discontinuities.size() != f.E_discontinuity.size())
        mismatch("discontinuity count", f.E_discontinuity.size(), data.discontinuities.size());
    for (size_t i = 0; i < f.E_discontinuity.size(); ++i) {
        if (data.discontinuities[i] != f.E_discontinuity[i])
            mismatch("discontinuity[" + std::to_string(i) + "]",
                     f.E_discontinuity[i], data.discontinuities[i]);
        if (data.f_discontinuities[i] != f.function(f.E_discontinuity[i]))
            mismatch("f_discontinuity[" + std::to_string(i) + "]",
                     f.function(f.E_discontinuity[i]), data.f_discontinuities[i]);
    }
}


} // namespace fft_tn_test
