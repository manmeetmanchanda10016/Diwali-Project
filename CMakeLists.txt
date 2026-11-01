
cmake_minimum_required(VERSION 3.16)

project(MiniTensor VERSION 0.1.0 LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler warnings
function(enable_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /permissive-)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE
            -Wall
            -Wextra
            -Wpedantic
        )
    endif()
endfunction()

# OpenMP
find_package(OpenMP REQUIRED)

# Main library
add_library(minitensor
    src/Shape.cpp
    src/Storage.cpp
    src/Tensor.cpp
    src/Ops.cpp
)

target_include_directories(minitensor
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(minitensor PUBLIC OpenMP::OpenMP_CXX)

enable_warnings(minitensor)

# Tests
enable_testing()

# Phase 1 tests
add_executable(test_phase1
    tests/test_phase1.cpp
)

target_link_libraries(test_phase1 PRIVATE minitensor)
enable_warnings(test_phase1)

add_test(NAME Phase1Tests COMMAND test_phase1)

# Phase 2 tests
add_executable(test_phase2
    tests/test_phase2.cpp
)

target_link_libraries(test_phase2 PRIVATE minitensor)
enable_warnings(test_phase2)

add_test(NAME Phase2Tests COMMAND test_phase2)

# Phase 3 tests
add_executable(test_phase3
    tests/test_phase3.cpp
)

target_link_libraries(test_phase3 PRIVATE minitensor)
enable_warnings(test_phase3)

add_test(NAME Phase3Tests COMMAND test_phase3)

# Phase 4 tests
add_executable(test_phase4
    tests/test_phase4.cpp
)

target_link_libraries(test_phase4 PRIVATE minitensor)
enable_warnings(test_phase4)

add_test(NAME Phase4Tests COMMAND test_phase4)

# Keep assertions enabled in all test builds
if(MSVC)
    target_compile_options(test_phase1 PRIVATE /UNDEBUG)
    target_compile_options(test_phase2 PRIVATE /UNDEBUG)
    target_compile_options(test_phase3 PRIVATE /UNDEBUG)
    target_compile_options(test_phase4 PRIVATE /UNDEBUG)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(test_phase1 PRIVATE -UNDEBUG)
    target_compile_options(test_phase2 PRIVATE -UNDEBUG)
    target_compile_options(test_phase3 PRIVATE -UNDEBUG)
    target_compile_options(test_phase4 PRIVATE -UNDEBUG)
endif()

# Basic example
add_executable(basic_example
    examples/main.cpp
)

target_link_libraries(basic_example PRIVATE minitensor)
enable_warnings(basic_example)

# Benchmark V1
add_executable(benchmark_phase1
    benchmarks/benchmark_phase1.cpp
)

target_link_libraries(benchmark_phase1 PRIVATE minitensor)
enable_warnings(benchmark_phase1)

# Benchmark V2
add_executable(benchmark_v2
    benchmarks/benchmark_v2.cpp
)

target_link_libraries(benchmark_v2 PRIVATE minitensor)
enable_warnings(benchmark_v2)

# Matrix multiplication benchmark
add_executable(benchmark_matmul
    benchmarks/benchmark_matmul.cpp
)

target_link_libraries(benchmark_matmul PRIVATE minitensor)
enable_warnings(benchmark_matmul)
