
#ifndef MINITENSOR_OPS_H
#define MINITENSOR_OPS_H

#include "minitensor/Tensor.h"

namespace minitensor {

Tensor add(const Tensor& a, const Tensor& b);
Tensor subtract(const Tensor& a, const Tensor& b);
Tensor multiply(const Tensor& a, const Tensor& b);

Tensor matmul(const Tensor& a, const Tensor& b);
Tensor matmul_parallel(const Tensor& a, const Tensor& b);

Tensor operator+(const Tensor& a, const Tensor& b);
Tensor operator-(const Tensor& a, const Tensor& b);
Tensor operator*(const Tensor& a, const Tensor& b);

}

#endif
