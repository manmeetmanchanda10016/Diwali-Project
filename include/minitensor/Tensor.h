#ifndef MINITENSOR_TENSOR_H
#define MINITENSOR_TENSOR_H

#include "minitensor/Shape.h"
#include "minitensor/Storage.h"

#include <cstddef>
#include <initializer_list>
#include <vector>

namespace minitensor {

namespace detail { struct TensorAccess; }

class Tensor {
private:
    friend struct detail::TensorAccess;
    struct UninitializedTag {};
    Tensor(UninitializedTag, const std::vector<std::size_t>& dimensions);

    Shape shape_;
    Storage storage_;

    std::size_t offset(
        const std::size_t* indices,
        std::size_t count
    ) const;

public:
    explicit Tensor(
        const std::vector<std::size_t>& dimensions,
        double initial_value = 0.0
    );

    Tensor(
        std::initializer_list<std::size_t> dimensions,
        double initial_value = 0.0
    );

    Tensor(const Tensor&) = default;
    Tensor& operator=(const Tensor&) = default;

    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(Tensor&&) noexcept = default;

    std::size_t size() const noexcept;
    std::size_t rank() const noexcept;

    const Shape& shape() const noexcept;

    double* data() noexcept;
    const double* data() const noexcept;

    void fill(double value);

    double& at(
        const std::vector<std::size_t>& indices
    );

    const double& at(
        const std::vector<std::size_t>& indices
    ) const;

    double& at(
        std::initializer_list<std::size_t> indices
    );

    const double& at(
        std::initializer_list<std::size_t> indices
    ) const;

    double& operator()(
        std::initializer_list<std::size_t> indices
    );

    const double& operator()(
        std::initializer_list<std::size_t> indices
    ) const;

    double& operator()(
        const std::vector<std::size_t>& indices
    );

    const double& operator()(
        const std::vector<std::size_t>& indices
    ) const;

    double& flat_at(std::size_t index);
    const double& flat_at(std::size_t index) const;

    void reshape(
        const std::vector<std::size_t>& dimensions
    );

    void reshape(
        std::initializer_list<std::size_t> dimensions
    );
};

} // namespace minitensor

#endif
