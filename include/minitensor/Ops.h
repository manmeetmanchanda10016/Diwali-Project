
#ifndef MINITENSOR_OPS_H
#define MINITENSOR_OPS_H

#include <cstddef>
#include "minitensor/Tensor.h"

namespace minitensor {

Tensor add(const Tensor& a, const Tensor& b);
Tensor subtract(const Tensor& a, const Tensor& b);
Tensor multiply(const Tensor& a, const Tensor& b);

Tensor add_parallel(const Tensor& a, const Tensor& b);
Tensor subtract_parallel(const Tensor& a, const Tensor& b);
Tensor multiply_parallel(const Tensor& a, const Tensor& b);

Tensor matmul(const Tensor& a, const Tensor& b);
Tensor matmul_parallel(const Tensor& a, const Tensor& b);

Tensor matmul_blocked(
    const Tensor& a,
    const Tensor& b,
    std::size_t block_size = 128
);

Tensor matmul_parallel_blocked(
    const Tensor& a,
    const Tensor& b,
    std::size_t block_size = 128
);

Tensor transpose(const Tensor& a);

Tensor add_broadcast(const Tensor& a, const Tensor& b);
Tensor subtract_broadcast(const Tensor& a, const Tensor& b);
Tensor multiply_broadcast(const Tensor& a, const Tensor& b);

double sum(const Tensor& a);
double mean(const Tensor& a);
double min(const Tensor& a);
double max(const Tensor& a);

Tensor operator+(const Tensor& a, const Tensor& b);
Tensor operator-(const Tensor& a, const Tensor& b);
Tensor operator*(const Tensor& a, const Tensor& b);

}

#endif
