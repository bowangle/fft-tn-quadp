#include <filesystem>

#include "type_int128.h"
#include "test_common.hpp"

int main(){
    using Sint = util::i128;
    using RealT = double;
    using ComplexT = std::complex<RealT>;

    std::filesystem::create_directories("test/output_fft_test");

    // tci all function
    const std::vector<fft_tn_test::FFTTestFunction<ComplexT>>& functions = fft_tn_test::fft_test_functions<ComplexT>();

    for (const auto& f : functions) {
        fft_tn_test::tci_function_for_test<ComplexT, Sint>(
            "test/output_fft_test", "double", f, /*nBit=*/30, /*nIter=*/10);
    }

    // Reload and verify
    for (const auto& f : functions) {
        auto data = fft_tn_test::load_mps_and_grid<ComplexT, Sint>(
            "test/output_fft_test", "double", std::string(f.name), /*nBit=*/30);
        fft_tn_test::verify_loaded_data<ComplexT, Sint>(data, f, /*nBit=*/30);
    }

    return 0;
}