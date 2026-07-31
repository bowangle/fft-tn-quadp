cmake_minimum_required(VERSION 3.21)
project(fft-tn-quadp LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(FFT_TN_BUILD_TESTS
       "Build fft-tn-quadp's own tests"
       ${PROJECT_IS_TOP_LEVEL})

# ---- dependency: TensorQuadOperation manages itself (transitively everything) -
# Its tests, and those of every layer below it, stay off unless the matching
# <LAYER>_BUILD_TESTS option is explicitly set.
if(NOT TARGET tensorquad_deps)
    add_subdirectory(extern/TensorQuadOperation)
endif()

# ---- own paths ----------------------------------------------------------------
set(FFT_TN_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ---- interface target: carries our headers + transitive deps ------------------
add_library(fft_tn_deps INTERFACE)

target_include_directories(fft_tn_deps INTERFACE
    ${FFT_TN_INCLUDE_DIR})

target_compile_features(fft_tn_deps INTERFACE cxx_std_20)

target_link_libraries(fft_tn_deps INTERFACE tensorquad_deps)

# ---- executables (real deliverables: always built) ----------------------------
# One entry per src/<name>.cpp -> target <name>
set(_fft_tn_execs
    precompute_qft)

foreach(_name IN LISTS _fft_tn_execs)
    add_executable(${_name} src/${_name}.cpp)
    target_link_libraries(${_name} PRIVATE fft_tn_deps)
    target_compile_options(${_name} PRIVATE
        -Wall -Wextra -Wreorder -Wshadow)
endforeach()

# ---- tests (leaf targets: nothing may ever depend on these) -------------------
if(FFT_TN_BUILD_TESTS)
    enable_testing()

    # One entry per test/test_<name>.cpp -> target fft_tn_test_<name>
    set(_fft_tn_tests
        magic_qft_mpo
        fft_vanilla)

    foreach(_name IN LISTS _fft_tn_tests)
        add_executable(fft_tn_test_${_name} test/test_${_name}.cpp)
        target_link_libraries(fft_tn_test_${_name} PRIVATE fft_tn_deps)
        target_compile_options(fft_tn_test_${_name} PRIVATE
            -Wall -Wextra -Wreorder -Wshadow -UNDEBUG)
        add_test(NAME fft_tn_test_${_name}
                 COMMAND fft_tn_test_${_name})
    endforeach()
endif()
