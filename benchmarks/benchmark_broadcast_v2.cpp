
#include "minitensor/Ops.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace minitensor;
using Clock = std::chrono::steady_clock;

struct Add {
    static double apply(double a, double b) { return a + b; }
};

struct Subtract {
    static double apply(double a, double b) { return a - b; }
};

struct Multiply {
    static double apply(double a, double b) { return a * b; }
};

template <typename Op>
Tensor precomputed_broadcast(const Tensor& a, const Tensor& b) {
    const auto& ad = a.shape().dimensions();
    const auto& bd = b.shape().dimensions();

    const std::size_t rank = std::max(ad.size(), bd.size());

    std::vector<std::size_t> output_dims(rank, 1);
    std::vector<std::size_t> a_strides(rank, 0);
    std::vector<std::size_t> b_strides(rank, 0);

    std::size_t stride = 1;

    for (std::size_t i = 0; i < rank; ++i) {
        const std::size_t ai = i < rank - ad.size()
            ? 1 : ad[i - (rank - ad.size())];

        const std::size_t bi = i < rank - bd.size()
            ? 1 : bd[i - (rank - bd.size())];

        if (ai != bi && ai != 1 && bi != 1) {
            throw std::invalid_argument("Incompatible broadcast shapes");
        }

        output_dims[i] = std::max(ai, bi);
    }

    for (std::size_t i = rank; i-- > 0;) {
        const std::size_t ai = i < rank - ad.size()
            ? 1 : ad[i - (rank - ad.size())];

        const std::size_t bi = i < rank - bd.size()
            ? 1 : bd[i - (rank - bd.size())];

        a_strides[i] = ai == 1 ? 0 : stride;
        b_strides[i] = bi == 1 ? 0 : stride;

        if (ai != 1) {
            stride *= ai;
        }
    }

    std::vector<std::size_t> actual_a(rank, 0);
    std::vector<std::size_t> actual_b(rank, 0);

    std::size_t a_stride = 1;
    std::size_t b_stride = 1;

    for (std::size_t i = ad.size(); i-- > 0;) {
        actual_a[rank - ad.size() + i] =
            ad[i] == 1 ? 0 : a_stride;
        a_stride *= ad[i];
    }

    for (std::size_t i = bd.size(); i-- > 0;) {
        actual_b[rank - bd.size() + i] =
            bd[i] == 1 ? 0 : b_stride;
        b_stride *= bd[i];
    }

    Tensor result(output_dims);

    const std::size_t count = result.size();

    for (std::size_t flat = 0; flat < count; ++flat) {
        std::size_t remaining = flat;
        std::size_t offset_a = 0;
        std::size_t offset_b = 0;

        for (std::size_t d = rank; d-- > 0;) {
            const std::size_t coordinate =
                remaining % output_dims[d];

            remaining /= output_dims[d];

            offset_a += coordinate * actual_a[d];
            offset_b += coordinate * actual_b[d];
        }

        result.data()[flat] =
            Op::apply(a.data()[offset_a], b.data()[offset_b]);
    }

    return result;
}

template <typename Op>
Tensor specialized_broadcast(const Tensor& a, const Tensor& b) {
    const auto& ad = a.shape().dimensions();
    const auto& bd = b.shape().dimensions();

    if (ad == bd) {
        Tensor result(ad);

        const double* ap = a.data();
        const double* bp = b.data();
        double* rp = result.data();

        #pragma omp simd
        for (std::size_t i = 0; i < result.size(); ++i) {
            rp[i] = Op::apply(ap[i], bp[i]);
        }

        return result;
    }

    if (ad.size() == 2 && bd.size() == 2) {
        const std::size_t rows = std::max(ad[0], bd[0]);
        const std::size_t cols = std::max(ad[1], bd[1]);

        const bool a_row = ad[0] == 1 && ad[1] == cols;
        const bool b_row = bd[0] == 1 && bd[1] == cols;

        const bool a_col = ad[0] == rows && ad[1] == 1;
        const bool b_col = bd[0] == rows && bd[1] == 1;

        if (a_row && bd[0] == rows && bd[1] == cols) {
            Tensor result({rows, cols});

            for (std::size_t i = 0; i < rows; ++i) {
                #pragma omp simd
                for (std::size_t j = 0; j < cols; ++j) {
                    result.data()[i * cols + j] =
                        Op::apply(a.data()[j], b.data()[i * cols + j]);
                }
            }

            return result;
        }

        if (b_row && ad[0] == rows && ad[1] == cols) {
            Tensor result({rows, cols});

            for (std::size_t i = 0; i < rows; ++i) {
                #pragma omp simd
                for (std::size_t j = 0; j < cols; ++j) {
                    result.data()[i * cols + j] =
                        Op::apply(a.data()[i * cols + j], b.data()[j]);
                }
            }

            return result;
        }

        if (a_col && bd[0] == rows && bd[1] == cols) {
            Tensor result({rows, cols});

            for (std::size_t i = 0; i < rows; ++i) {
                const double value = a.data()[i];

                #pragma omp simd
                for (std::size_t j = 0; j < cols; ++j) {
                    result.data()[i * cols + j] =
                        Op::apply(value, b.data()[i * cols + j]);
                }
            }

            return result;
        }

        if (b_col && ad[0] == rows && ad[1] == cols) {
            Tensor result({rows, cols});

            for (std::size_t i = 0; i < rows; ++i) {
                const double value = b.data()[i];

                #pragma omp simd
                for (std::size_t j = 0; j < cols; ++j) {
                    result.data()[i * cols + j] =
                        Op::apply(a.data()[i * cols + j], value);
                }
            }

            return result;
        }
    }

    return precomputed_broadcast<Op>(a, b);
}

double median_time(const std::function<void()>& function) {
    std::vector<double> times;

    for (int i = 0; i < 2; ++i) {
        function();
    }

    for (int i = 0; i < 7; ++i) {
        const auto start = Clock::now();
        function();
        const auto end = Clock::now();

        times.push_back(
            std::chrono::duration<double, std::milli>(
                end - start
            ).count()
        );
    }

    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

bool correct(const Tensor& a, const Tensor& b) {
    if (a.shape().dimensions() != b.shape().dimensions()) {
        return false;
    }

    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::abs(a.data()[i] - b.data()[i]) > 1e-9) {
            return false;
        }
    }

    return true;
}

template <typename Op>
void run_case(
    const std::string& name,
    const Tensor& a,
    const Tensor& b,
    const std::function<Tensor(const Tensor&, const Tensor&)>& current
) {
    Tensor reference = current(a, b);

    Tensor precomputed = precomputed_broadcast<Op>(a, b);
    Tensor specialized = specialized_broadcast<Op>(a, b);

    if (!correct(reference, precomputed) ||
        !correct(reference, specialized)) {
        std::cout << name << ": CORRECTNESS FAILED\n";
        return;
    }

    const double current_time = median_time([&]() {
        Tensor result = current(a, b);
    });

    const double precomputed_time = median_time([&]() {
        Tensor result = precomputed_broadcast<Op>(a, b);
    });

    const double specialized_time = median_time([&]() {
        Tensor result = specialized_broadcast<Op>(a, b);
    });

    std::cout << "\n" << name << "\n";
    std::cout << "Current:       " << current_time << " ms\n";
    std::cout << "Precomputed:   " << precomputed_time << " ms\n";
    std::cout << "Specialized:   " << specialized_time << " ms\n";

    std::cout << "Precomputed speedup: "
              << current_time / precomputed_time << "x\n";

    std::cout << "Specialized speedup: "
              << current_time / specialized_time << "x\n";
}

int main() {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "MINITENSOR BROADCASTING BENCHMARK V2\n";

    Tensor a({512, 512});
    Tensor b({512, 512});

    for (std::size_t i = 0; i < a.size(); ++i) {
        a.data()[i] = static_cast<double>(i % 100) * 0.1;
        b.data()[i] = static_cast<double>(i % 50) * 0.2;
    }

    Tensor row({1, 512});
    Tensor column({512, 1});

    for (std::size_t i = 0; i < 512; ++i) {
        row.data()[i] = static_cast<double>(i) * 0.1;
        column.data()[i] = static_cast<double>(i) * 0.2;
    }

    std::cout << "\n========== ADD ==========\n";

    run_case<Add>(
        "Same shape",
        a, b,
        [](const Tensor& x, const Tensor& y) {
            return add_broadcast(x, y);
        }
    );

    run_case<Add>(
        "Row broadcasting",
        a, row,
        [](const Tensor& x, const Tensor& y) {
            return add_broadcast(x, y);
        }
    );

    run_case<Add>(
        "Column broadcasting",
        a, column,
        [](const Tensor& x, const Tensor& y) {
            return add_broadcast(x, y);
        }
    );

    Tensor high_a({256, 256, 16});
    Tensor high_b({1, 1, 16});

    for (std::size_t i = 0; i < high_a.size(); ++i) {
        high_a.data()[i] = static_cast<double>(i % 100) * 0.1;
    }

    for (std::size_t i = 0; i < high_b.size(); ++i) {
        high_b.data()[i] = static_cast<double>(i) * 0.2;
    }

    run_case<Add>(
        "Higher-dimensional broadcasting",
        high_a, high_b,
        [](const Tensor& x, const Tensor& y) {
            return add_broadcast(x, y);
        }
    );

    std::cout << "\n========== SUBTRACT ==========\n";

    run_case<Subtract>(
        "Row broadcasting",
        a, row,
        [](const Tensor& x, const Tensor& y) {
            return subtract_broadcast(x, y);
        }
    );

    std::cout << "\n========== MULTIPLY ==========\n";

    run_case<Multiply>(
        "Row broadcasting",
        a, row,
        [](const Tensor& x, const Tensor& y) {
            return multiply_broadcast(x, y);
        }
    );

    std::cout << "\nBenchmark completed.\n";
    return 0;
}
