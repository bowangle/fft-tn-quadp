#include "magic_tensor_qft.hpp"
#include "type_double_double.h"
#include "type_float128_boost.h"
#include <iostream>
#include <cmath>

template<typename RealT>
double to_double(RealT const& v) { return static_cast<double>(v); }

template<>
double to_double<dd_128>(dd_128 const& v) { return v.x[0]; }

template<>
double to_double<float128>(float128 const& v) { return v.convert_to<double>(); }

template<typename ComplexT>
void test_build(const std::string& name, int r, int chi, int n_points = 10)
{
    using Real = typename ComplexT::value_type;
    std::cout << "\n=== Testing " << name << " (r=" << r << ", chi=" << chi << ") ===\n";

    // 1. chebyshev nodes
    auto nodes = magic_tensor_qft::chebyshev_nodes_on_01<Real>(chi);
    std::cout << "  chebyshev: [0]=" << to_double<Real>(nodes[0])
              << " [" << chi-1 << "]=" << to_double<Real>(nodes[chi-1]) << "\n";

    // 2. lagrange
    Real x_mid = Real(0.5);
    auto P = magic_tensor_qft::lagrange_basis_values<Real>(nodes, x_mid);
    std::cout << "  lagrange at 0.5: sum = "
              << to_double<Real>(std::real(P[0] + P[1]))
              << " ...\n";

    // 3. Build MPO tensors (flat-packed)
    auto W = magic_tensor_qft::build_qft_mpo_tensors<ComplexT>(r, chi);
    std::cout << "  W.size() = " << W.size() << "\n";
    std::cout << "  W[0].size() = " << W[0].size() << "\n";
    std::cout << "  W[0][0] = ("
              << to_double<Real>(std::real(W[0][0])) << ","
              << to_double<Real>(std::imag(W[0][0])) << ")\n";

    // 4. Build MPO from W
    auto mpo = magic_tensor_qft::build_mpo<ComplexT>(W, chi);
    std::cout << "  MPO nBit = " << mpo.get_size() << "\n";
    std::cout << "  MPO chi  = " << mpo.get_chi() << "\n";

    auto shape = mpo.get_shape();
    std::cout << "  MPO shape: ";
    for (auto const& [l, p, rr] : shape)
        std::cout << "(" << l << "," << p << "," << rr << ") ";
    std::cout << "\n";

    // 5. Evaluate at a few points
    auto points = mpo.generate_points(n_points);
    auto vals = mpo.eval_list(points);
    std::cout << "  eval " << n_points << " points:";
    for (int i = 0; i < n_points && i < 3; ++i) {
        std::cout << " (" << to_double<Real>(std::real(vals[i]))
                  << "," << to_double<Real>(std::imag(vals[i])) << ")";
    }
    std::cout << "\n";

    // 6. Convenience function
    auto mpo2 = magic_tensor_qft::build_qft_mpo_magic<ComplexT>(r, chi);
    std::cout << "  build_qft_mpo_magic: nBit=" << mpo2.get_size()
              << " chi=" << mpo2.get_chi() << "\n";
}

template<typename ComplexT>
void test_error(const std::string& name, int r, int chi_max, int n_points = 1000){
    for (int i=1; i<chi_max+1; i++){
        MPO<ComplexT> mpo = magic_tensor_qft::build_qft_mpo_magic<ComplexT>(r, i);
        auto err = magic_tensor_qft::compute_qft_mpo_error<ComplexT>(mpo, n_points);
        std::cout << name << " " << i <<"\n";
        std::cout << "  Error vs exact QFT (" << n_points << " samples):\n";
        std::cout << "    max |MPO - F|  = " << err[0] << "\n";
        std::cout << "    rms |MPO - F|  = " << err[1] << "\n";
        std::cout << "    rms |F|        = " << err[2] << "\n";
        std::cout << "    relative rms   = " << err[1]/err[2] << "\n";
    }
}

int main()
{
    int r = 30;
    int chi = 12;

    test_build<std::complex<double>>("complex<double>", r, chi);
    test_build<Cdd_128>("Cdd_128", r, chi);
    test_build<Cfloat128>("Cfloat128", r, chi);

    test_error<std::complex<double>>("complex<double>", r, 30);
    test_error<Cdd_128>("Cdd_128", r, 30);
    test_error<Cfloat128>("Cfloat128", r, 30);
    std::cout << "\nAll tests passed!\n";
    return 0;
}
