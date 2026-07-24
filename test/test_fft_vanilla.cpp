#include "fft_tn.hpp"
#include "type_int128.h"
#include "type_double_double.h"
#include "type_float128_boost.h"
#include <iostream>
#include <chrono>

template<typename Cscalar, typename Sint>
void test_fft_construction(const std::string& type_name, const Sint padding_bit_)
{
    std::cout << "Testing FFTmps<" << type_name << ">...\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    try {
        FFTmps<Cscalar, Sint> fft(
            "test/gE<_site[[1, 0], [0, 0]]",    // prefix_mps
            "mpo_data/data/",                   // folder_qft_data
            padding_bit_                        // padding bit
        );
        auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << "  Constructed successfully.\n";
        std::cout << "  Time: " << std::chrono::duration<double>(t1 - t0).count() << " s\n";
    } catch (const std::exception& e) {
        std::cout << "  Failed: " << e.what() << "\n";
    }
}

template<typename Cscalar, typename Sint>
void test_fft_fft_vanilla_execution(const std::string& type_name, const Sint padding_bit_)
{
    std::cout << "Testing FFTmps fft_vanilla run<" << type_name << ">...\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    try {
        FFTmps<Cscalar, Sint> fft(
            "test/gE<_site[[1, 0], [0, 0]]",    // prefix_mps
            "mpo_data/data/",                   // folder_qft_data
            padding_bit_                        // padding bit
        );
        fft.fft_vanilla();
        auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << "  fft_vanilla run successfully.\n";
        std::cout << "  Time: " << std::chrono::duration<double>(t1 - t0).count() << " s\n";
        std::cout << "  Max bond dim: " << fft.get_chi() << "\n";
    } catch (const std::exception& e) {
        std::cout << "  Failed: " << e.what() << "\n";
    }
}

int main()
{
    test_fft_construction<std::complex<double>, util::i128>("complex<double>", util::i128(0));
    test_fft_construction<Cdd_128, util::i128>("Cdd_128", util::i128(0));
    test_fft_construction<Cfloat128, util::i128>("Cfloat128", util::i128(0));

    test_fft_construction<std::complex<double>, util::i128>("complex<double>", util::i128(10));
    test_fft_construction<Cdd_128, util::i128>("Cdd_128", util::i128(10));
    test_fft_construction<Cfloat128, util::i128>("Cfloat128", util::i128(10));


    test_fft_fft_vanilla_execution<std::complex<double>, util::i128>("complex<double>", util::i128(0));
    test_fft_fft_vanilla_execution<Cdd_128, util::i128>("Cdd_128>", util::i128(0));
    test_fft_fft_vanilla_execution<Cfloat128, util::i128>("Cfloat128", util::i128(0));      // too long

    test_fft_fft_vanilla_execution<std::complex<double>, util::i128>("complex<double>", util::i128(10));
    test_fft_fft_vanilla_execution<Cdd_128, util::i128>("Cdd_128>", util::i128(10));
    //test_fft_fft_vanilla_execution<Cfloat128, util::i128>("Cfloat128", util::i128(0));    // too long

    return 0;
}
