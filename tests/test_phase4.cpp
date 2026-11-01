
#include "minitensor/Ops.h"

#include <cassert>
#include <stdexcept>
#include <vector>
#include <iostream>

using namespace minitensor;

void test_transpose()
{
    Tensor a({2, 3});

    double values[] = {1, 2, 3, 4, 5, 6};

    for (std::size_t i = 0; i < 6; ++i)
    {
        a.data()[i] = values[i];
    }

    Tensor result = transpose(a);

    assert(result.shape().dimensions() ==
           std::vector<std::size_t>({3, 2}));

    double expected[] = {1, 4, 2, 5, 3, 6};

    for (std::size_t i = 0; i < 6; ++i)
    {
        assert(result.data()[i] == expected[i]);
    }

    assert(a.data()[0] == 1);
    assert(a.data()[5] == 6);
}

void test_square_transpose()
{
    Tensor a({2, 2});

    a.data()[0] = 1;
    a.data()[1] = 2;
    a.data()[2] = 3;
    a.data()[3] = 4;

    Tensor result = transpose(a);

    assert(result.data()[0] == 1);
    assert(result.data()[1] == 3);
    assert(result.data()[2] == 2);
    assert(result.data()[3] == 4);
}

void test_invalid_transpose()
{
    Tensor a({2, 3, 4});

    bool threw = false;

    try
    {
        transpose(a);
    }
    catch (const std::invalid_argument&)
    {
        threw = true;
    }

    assert(threw);
}

void test_row_broadcasting()
{
    Tensor a({2, 3});
    Tensor b({3});

    double a_values[] = {1, 2, 3, 4, 5, 6};
    double b_values[] = {10, 20, 30};

    for (std::size_t i = 0; i < 6; ++i)
    {
        a.data()[i] = a_values[i];
    }

    for (std::size_t i = 0; i < 3; ++i)
    {
        b.data()[i] = b_values[i];
    }

    Tensor result = add_broadcast(a, b);

    double expected[] = {11, 22, 33, 14, 25, 36};

    for (std::size_t i = 0; i < 6; ++i)
    {
        assert(result.data()[i] == expected[i]);
    }
}

void test_column_broadcasting()
{
    Tensor a({2, 3});
    Tensor b({2, 1});

    double a_values[] = {1, 2, 3, 4, 5, 6};

    a.data()[0] = a_values[0];
    a.data()[1] = a_values[1];
    a.data()[2] = a_values[2];
    a.data()[3] = a_values[3];
    a.data()[4] = a_values[4];
    a.data()[5] = a_values[5];

    b.data()[0] = 10;
    b.data()[1] = 20;

    Tensor result = add_broadcast(a, b);

    double expected[] = {11, 12, 13, 24, 25, 26};

    for (std::size_t i = 0; i < 6; ++i)
    {
        assert(result.data()[i] == expected[i]);
    }
}

void test_broadcast_subtraction()
{
    Tensor a({2, 3});
    Tensor b({3});

    for (std::size_t i = 0; i < 6; ++i)
    {
        a.data()[i] = 10;
    }

    b.data()[0] = 1;
    b.data()[1] = 2;
    b.data()[2] = 3;

    Tensor result = subtract_broadcast(a, b);

    assert(result.data()[0] == 9);
    assert(result.data()[1] == 8);
    assert(result.data()[2] == 7);
    assert(result.data()[3] == 9);
}

void test_broadcast_multiplication()
{
    Tensor a({2, 3});
    Tensor b({3});

    for (std::size_t i = 0; i < 6; ++i)
    {
        a.data()[i] = 2;
    }

    b.data()[0] = 3;
    b.data()[1] = 4;
    b.data()[2] = 5;

    Tensor result = multiply_broadcast(a, b);

    assert(result.data()[0] == 6);
    assert(result.data()[1] == 8);
    assert(result.data()[2] == 10);
    assert(result.data()[3] == 6);
    assert(result.data()[4] == 8);
    assert(result.data()[5] == 10);
}

void test_scalar_broadcasting()
{
    Tensor a({2, 2});
    Tensor scalar(std::vector<std::size_t>{});

    for (std::size_t i = 0; i < 4; ++i)
    {
        a.data()[i] = static_cast<double>(i + 1);
    }

    scalar.data()[0] = 10;

    Tensor result = add_broadcast(a, scalar);

    assert(result.data()[0] == 11);
    assert(result.data()[1] == 12);
    assert(result.data()[2] == 13);
    assert(result.data()[3] == 14);
}

void test_incompatible_broadcasting()
{
    Tensor a({2, 3});
    Tensor b({2, 2});

    bool threw = false;

    try
    {
        add_broadcast(a, b);
    }
    catch (const std::invalid_argument&)
    {
        threw = true;
    }

    assert(threw);
}

// Phase 4: Reduction tests

void test_reductions()
{
    Tensor a({2, 3});

    double values[] = {1, 2, 3, 4, 5, 6};

    for (std::size_t i = 0; i < 6; ++i)
    {
        a.data()[i] = values[i];
    }

    assert(sum(a) == 21.0);
    assert(mean(a) == 3.5);
    assert(min(a) == 1.0);
    assert(max(a) == 6.0);
}

void test_reductions_negative_values()
{
    Tensor a({4});

    a.data()[0] = -5;
    a.data()[1] = -2;
    a.data()[2] = -10;
    a.data()[3] = -3;

    assert(sum(a) == -20.0);
    assert(mean(a) == -5.0);
    assert(min(a) == -10.0);
    assert(max(a) == -2.0);
}

void test_scalar_reductions()
{
    Tensor a(std::vector<std::size_t>{});

    a.data()[0] = 42.0;

    assert(sum(a) == 42.0);
    assert(mean(a) == 42.0);
    assert(min(a) == 42.0);
    assert(max(a) == 42.0);
}

void test_empty_reductions()
{
    Tensor a({0, 3});

    assert(sum(a) == 0.0);

    bool mean_threw = false;
    bool min_threw = false;
    bool max_threw = false;

    try
    {
        mean(a);
    }
    catch (const std::invalid_argument&)
    {
        mean_threw = true;
    }

    try
    {
        min(a);
    }
    catch (const std::invalid_argument&)
    {
        min_threw = true;
    }

    try
    {
        max(a);
    }
    catch (const std::invalid_argument&)
    {
        max_threw = true;
    }

    assert(mean_threw);
    assert(min_threw);
    assert(max_threw);
}

int main()
{
    test_transpose();
    test_square_transpose();
    test_invalid_transpose();

    test_row_broadcasting();
    test_column_broadcasting();
    test_broadcast_subtraction();
    test_broadcast_multiplication();
    test_scalar_broadcasting();
    test_incompatible_broadcasting();

    test_reductions();
    test_reductions_negative_values();
    test_scalar_reductions();
    test_empty_reductions();

    std::cout << "Phase 4 tests passed!" << std::endl;

    return 0;
}
