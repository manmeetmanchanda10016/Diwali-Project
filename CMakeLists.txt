
cmake_minimum_required(VERSION 3.16)

project(MiniTensor
    VERSION 0.1.0
    DESCRIPTION "A lightweight C++ tensor library"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CheckIPOSupported)

# Build options

option(MINITENSOR_ENABLE_NATIVE
    "Optimize specifically for the current CPU"
    OFF
)

option(MINITENSOR_ENABLE_LTO
    "Enable Link Time Optimization"
    ON
)

# Link Time Optimization

set(IPO_SUPPORTED FALSE)

if(MINITENSOR_ENABLE_LTO)
    check_ipo_supported(
        RESULT IPO_SUPPORTED
        OUTPUT IPO_ERROR
        LANGUAGES CXX
    )

    if(IPO_SUPPORTED)
        message(STATUS "Link Time Optimization: ENABLED")
    else()
        message(WARNING "LTO is not supported: ${IPO_ERROR}")
    endif()
endif()

# Compiler optimizations

function(enable_optimizations target)

    if(MSVC)
        target_compile_options(${target} PRIVATE
            $<$<CONFIG:Release>:/O2>
        )
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE
            $<$<CONFIG:Release>:-O3>
        )
    endif()

    if(MINITENSOR_ENABLE_NATIVE AND
       CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")

        target_compile_options(${target} PRIVATE
            $<$<CONFIG:Release>:-march=native>
            $<$<CONFIG:Release>:-mtune=native>
        )
    endif()

    if(MINITENSOR_ENABLE_LTO AND IPO_SUPPORTED)
        set_property(
            TARGET ${target}
            PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE
        )
    endif()

endfunction()

# Compiler warnings

function(enable_warnings target)

    if(MSVC)
        target_compile_options(${target} PRIVATE
            /W4
            /permissive-
        )
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

add_library(MiniTensor::minitensor ALIAS minitensor)

set_target_properties(minitensor PROPERTIES
    EXPORT_NAME minitensor
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

target_include_directories(minitensor
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(minitensor
    PUBLIC
        OpenMP::OpenMP_CXX
)

enable_warnings(minitensor)
enable_optimizations(minitensor)

# Testing

enable_testing()

function(add_minitensor_test target source test_name)

    add_executable(${target} ${source})

    target_link_libraries(${target} PRIVATE minitensor)

    enable_warnings(${target})
    enable_optimizations(${target})

    add_test(
        NAME ${test_name}
        COMMAND ${target}
    )

    # Keep assertions enabled in Release builds.

    if(MSVC)
        target_compile_options(${target} PRIVATE /UNDEBUG)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE -UNDEBUG)
    endif()

endfunction()

add_minitensor_test(test_phase1 tests/test_phase1.cpp Phase1Tests)
add_minitensor_test(test_phase2 tests/test_phase2.cpp Phase2Tests)
add_minitensor_test(test_phase3 tests/test_phase3.cpp Phase3Tests)
add_minitensor_test(test_phase4 tests/test_phase4.cpp Phase4Tests)
add_minitensor_test(test_phase5 tests/test_phase5.cpp Phase5Tests)
add_minitensor_test(test_phase6 tests/test_phase6.cpp Phase6Tests)

# Examples

add_executable(basic_example
    examples/main.cpp
)

target_link_libraries(basic_example PRIVATE minitensor)

enable_warnings(basic_example)
enable_optimizations(basic_example)

add_executable(tensor_application
    examples/tensor_application.cpp
)

target_link_libraries(tensor_application PRIVATE minitensor)

enable_warnings(tensor_application)
enable_optimizations(tensor_application)

# Benchmarks

function(add_minitensor_benchmark target source)

    add_executable(${target} ${source})

    target_link_libraries(${target} PRIVATE minitensor)

    enable_warnings(${target})
    enable_optimizations(${target})

endfunction()

add_minitensor_benchmark(benchmark_phase1 benchmarks/benchmark_phase1.cpp)
add_minitensor_benchmark(benchmark_v2 benchmarks/benchmark_v2.cpp)
add_minitensor_benchmark(benchmark_matmul benchmarks/benchmark_matmul.cpp)
add_minitensor_benchmark(benchmark_elementwise benchmarks/benchmark_elementwise.cpp)
add_minitensor_benchmark(benchmark_transpose benchmarks/benchmark_transpose.cpp)
add_minitensor_benchmark(benchmark_broadcast benchmarks/benchmark_broadcast.cpp)
add_minitensor_benchmark(benchmark_broadcast_v2 benchmarks/benchmark_broadcast_v2.cpp)
add_minitensor_benchmark(benchmark_broadcast_v3 benchmarks/benchmark_broadcast_v3.cpp)
add_minitensor_benchmark(benchmark_remaining benchmarks/benchmark_remaining.cpp)

# Installation and packaging

install(
    TARGETS minitensor
    EXPORT MiniTensorTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
    DIRECTORY include/minitensor
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
    EXPORT MiniTensorTargets
    FILE MiniTensorTargets.cmake
    NAMESPACE MiniTensor::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MiniTensor
)

configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/MiniTensorConfig.cmake.in
    ${PROJECT_BINARY_DIR}/MiniTensorConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MiniTensor
)

write_basic_package_version_file(
    ${PROJECT_BINARY_DIR}/MiniTensorConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(
    FILES
        ${PROJECT_BINARY_DIR}/MiniTensorConfig.cmake
        ${PROJECT_BINARY_DIR}/MiniTensorConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MiniTensor
)

# Configuration summary

message(STATUS "====================================")
message(STATUS "MiniTensor Build Configuration")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "C++ Standard: C++17")
message(STATUS "OpenMP: ENABLED")
message(STATUS "Native CPU optimization: ${MINITENSOR_ENABLE_NATIVE}")
message(STATUS "LTO requested: ${MINITENSOR_ENABLE_LTO}")
message(STATUS "====================================")

# License
