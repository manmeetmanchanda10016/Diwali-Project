cmake_minimum_required(VERSION 3.16)

project(MiniTensor VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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

find_package(OpenMP REQUIRED)

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

enable_testing()

add_executable(test_phase1 tests/test_phase1.cpp)
target_link_libraries(test_phase1 PRIVATE minitensor)
enable_warnings(test_phase1)
add_test(NAME Phase1Tests COMMAND test_phase1)

add_executable(test_phase2 tests/test_phase2.cpp)
target_link_libraries(test_phase2 PRIVATE minitensor)
enable_warnings(test_phase2)
add_test(NAME Phase2Tests COMMAND test_phase2)

add_executable(test_phase3 tests/test_phase3.cpp)
target_link_libraries(test_phase3 PRIVATE minitensor)
enable_warnings(test_phase3)
add_test(NAME Phase3Tests COMMAND test_phase3)

add_executable(test_phase4 tests/test_phase4.cpp)
target_link_libraries(test_phase4 PRIVATE minitensor)
enable_warnings(test_phase4)
add_test(NAME Phase4Tests COMMAND test_phase4)

add_executable(test_phase5 tests/test_phase5.cpp)
target_link_libraries(test_phase5 PRIVATE minitensor)
enable_warnings(test_phase5)
add_test(NAME Phase5Tests COMMAND test_phase5)

# Keep assertions enabled in every test build, including Release.
if(MSVC)
    foreach(test_target test_phase1 test_phase2 test_phase3 test_phase4 test_phase5)
        target_compile_options(${test_target} PRIVATE /UNDEBUG)
    endforeach()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    foreach(test_target test_phase1 test_phase2 test_phase3 test_phase4 test_phase5)
        target_compile_options(${test_target} PRIVATE -UNDEBUG)
    endforeach()
endif()

add_executable(basic_example examples/main.cpp)
target_link_libraries(basic_example PRIVATE minitensor)
enable_warnings(basic_example)

add_executable(benchmark_phase1 benchmarks/benchmark_phase1.cpp)
target_link_libraries(benchmark_phase1 PRIVATE minitensor)
enable_warnings(benchmark_phase1)

add_executable(benchmark_v2 benchmarks/benchmark_v2.cpp)
target_link_libraries(benchmark_v2 PRIVATE minitensor)
enable_warnings(benchmark_v2)

add_executable(benchmark_matmul benchmarks/benchmark_matmul.cpp)
target_link_libraries(benchmark_matmul PRIVATE minitensor)
enable_warnings(benchmark_matmul)