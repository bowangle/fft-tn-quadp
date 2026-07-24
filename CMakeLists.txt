cmake_minimum_required(VERSION 3.16)
project(fft-tn-quadp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---- dependency: TensorQuadOperation manages itself (transitively everything) -
add_subdirectory(extern/TensorQuadOperation EXCLUDE_FROM_ALL)

# ---- own paths ----------------------------------------------------------------
set(PROJECT_INCLUDE_DIR     ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ---- our dependency bundle: TQO + our own headers -----------------------------
add_library(project_deps INTERFACE)
target_include_directories(project_deps INTERFACE ${PROJECT_INCLUDE_DIR})
target_link_libraries(project_deps INTERFACE tensorquad_deps)

# ---- test targets -------------------------------------------------------------
set(TEST_TARGETS
    Test_qft_mpo
    Test_fft_vanilla
)

add_executable(Test_qft_mpo test/test_magic_qft_mpo.cpp)
add_executable(Test_fft_vanilla test/test_fft_vanilla.cpp)

foreach(_tgt IN LISTS TEST_TARGETS)
    target_link_libraries(${_tgt} PRIVATE project_deps)
    target_compile_options(${_tgt} PRIVATE -Wall -Wextra -Wreorder -Wshadow)
endforeach()

set(exec_TARGETS
    calc_MPO_QFT
)

add_executable(calc_MPO_QFT src/precompute_qft.cpp)

foreach(_tgt IN LISTS exec_TARGETS)
    target_link_libraries(${_tgt} PRIVATE project_deps)
    target_compile_options(${_tgt} PRIVATE -Wall -Wextra -Wreorder -Wshadow)
endforeach()
