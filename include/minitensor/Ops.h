#ifndef MINITENSOR_OPS_H
#define MINITENSOR_OPS_H

#include "minitensor/Tensor.h"

namespace minitensor {

Tensor add(const Tensor& a, const Tensor& b);
Tensor subtract(const Tensor& a, const Tensor& b);
Tensor multiply(const Tensor& a, const Tensor& b);

Tensor matmul(const Tensor& a, const Tensor& b);
Tensor matmul_parallel(const Tensor& a, const Tensor& b);

Tensor transpose(const Tensor& a);

// Phase 4: Broadcasting
Tensor add_broadcast(const Tensor& a, const Tensor& b);
Tensor subtract_broadcast(const Tensor& a, const Tensor& b);
Tensor multiply_broadcast(const Tensor& a, const Tensor& b);

// Phase 4: Reduction Operations
double sum(const Tensor& a);
double mean(const Tensor& a);
double min(const Tensor& a);
double max(const Tensor& a);

Tensor operator+(const Tensor& a, const Tensor& b);
Tensor operator-(const Tensor& a, const Tensor& b);
Tensor operator*(const Tensor& a, const Tensor& b);

}

#endif