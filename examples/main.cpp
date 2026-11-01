
#include "minitensor/Tensor.h"

#include <iostream>

int main() {
    minitensor::Tensor tensor({2, 3}, 5.0);

    std::cout << "Tensor rank: " << tensor.rank() << '\n';

    std::cout << "Tensor size: " << tensor.size() << '\n';

    std::cout << "Tensor shape: ";

    for (const auto dimension : tensor.shape().dimensions()) {
        std::cout << dimension << ' ';
    }

    std::cout << "\nTensor data: ";

    for (std::size_t i = 0; i < tensor.size(); ++i) {
        std::cout << tensor.data()[i] << ' ';
    }

    std::cout << '\n';

    return 0;
}
