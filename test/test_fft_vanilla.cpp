#include "fft_tn.hpp"
#include "type_int128.h"
#include "type_double_double.h"
#include "type_float128_boost.h"
#include <iostream>

template<typename Cscalar, typename Sint>
void test_fft_construction(const std::string& type_name)
{
    std::cout << "Testing FFTmps<" << type_name << ">...\n";
    try {
        FFTmps<Cscalar, Sint> fft(
            "test/gE<_site[[1, 0], [0, 0]]",   // prefix_mps
            "mpo_data/data/"                      // folder_qft_data
        );
        std::cout << "  Constructed successfully.\n";
    } catch (const std::exception& e) {
        std::cout << "  Failed: " << e.what() << "\n";
    }
}

int main()
{
    test_fft_construction<std::complex<double>, util::i128>("complex<double>");
    test_fft_construction<Cdd_128, util::i128>("Cdd_128");
    test_fft_construction<Cfloat128, util::i128>("Cfloat128");

    return 0;
}
