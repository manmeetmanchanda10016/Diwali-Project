#include "minitensor/Tensor.h"
#include "minitensor/Ops.h"

#include <iostream>
#include <vector>
#include <string>
#include <limits>
#include <stdexcept>
#include <algorithm>

using namespace std;
using namespace minitensor;

// ---------------- INPUT HANDLING ----------------

template <typename T>
bool read_input(const string& prompt, T& value)
{
    while (true)
    {
        cout << prompt;

        if (cin >> value)
        {
            return true;
        }

        if (cin.eof())
        {
            cout << "\nInput ended. Exiting application.\n";
            return false;
        }

        cout << "Invalid input. Please enter a valid value.\n";

        cin.clear();
        cin.ignore(numeric_limits<streamsize>::max(), '\n');
    }
}

// ---------------- TENSOR DISPLAY ----------------

void print_tensor(const Tensor& tensor)
{
    cout << "\nShape: ";

    cout << "(";

    for (size_t i = 0; i < tensor.rank(); i++)
    {
        cout << tensor.shape().dimensions()[i];

        if (i + 1 < tensor.rank())
        {
            cout << ", ";
        }
    }

    cout << ")\n";

    cout << "Elements: ";

    for (size_t i = 0; i < tensor.size(); i++)
    {
        cout << tensor.flat_at(i) << " ";
    }

    cout << "\n";
}

// ---------------- CREATE TENSOR ----------------

bool create_tensor(Tensor& tensor)
{
    size_t rank;

    if (!read_input("Enter tensor rank (1-10): ", rank))
        return false;

    if (rank < 1 || rank > 10)
    {
        cout << "Rank must be between 1 and 10.\n";
        return false;
    }

    vector<size_t> dimensions(rank);

    for (size_t i = 0; i < rank; i++)
    {
        if (!read_input(
            "Enter dimension " + to_string(i + 1) + ": ",
            dimensions[i]))
        {
            return false;
        }
    }

    try
    {
        Tensor temp(dimensions);

        cout << "Enter " << temp.size() << " elements:\n";

        for (size_t i = 0; i < temp.size(); i++)
        {
            double value;

            if (!read_input(
                "Element " + to_string(i) + ": ",
                value))
            {
                return false;
            }

            temp.flat_at(i) = value;
        }

        tensor = std::move(temp);

        cout << "\nTensor created successfully!\n";
        print_tensor(tensor);

        return true;
    }
    catch (const exception& e)
    {
        cout << "Error: " << e.what() << "\n";
        return false;
    }
}

// ---------------- INDEX INPUT ----------------

bool read_indices(
    const Tensor& tensor,
    vector<size_t>& indices)
{
    indices.resize(tensor.rank());

    for (size_t i = 0; i < tensor.rank(); i++)
    {
        if (!read_input(
            "Enter index for dimension " + to_string(i + 1) + ": ",
            indices[i]))
        {
            return false;
        }
    }

    return true;
}

// ---------------- ACCESS ELEMENT ----------------

void access_element(const Tensor& tensor)
{
    vector<size_t> indices;

    if (!read_indices(tensor, indices))
        return;

    try
    {
        cout << "Element value: " << tensor.at(indices) << "\n";
    }
    catch (const exception& e)
    {
        cout << "Error: " << e.what() << "\n";
    }
}

// ---------------- MODIFY ELEMENT ----------------

void modify_element(Tensor& tensor)
{
    vector<size_t> indices;

    if (!read_indices(tensor, indices))
        return;

    double value;

    if (!read_input("Enter new value: ", value))
        return;

    try
    {
        tensor.at(indices) = value;

        cout << "Element updated successfully!\n";
        print_tensor(tensor);
    }
    catch (const exception& e)
    {
        cout << "Error: " << e.what() << "\n";
    }
}

// ---------------- TENSOR MANAGEMENT ----------------

void tensor_management()
{
    Tensor tensor({1});

    if (!create_tensor(tensor))
        return;

    while (true)
    {
        cout << "\n========== TENSOR MANAGEMENT ==========\n";
        cout << "1. Display tensor\n";
        cout << "2. Access element\n";
        cout << "3. Modify element\n";
        cout << "4. Fill tensor\n";
        cout << "5. Reshape tensor\n";
        cout << "0. Back to main menu\n";

        int choice;

        if (!read_input("Enter your choice: ", choice))
            return;

        try
        {
            switch (choice)
            {
                case 1:
                    print_tensor(tensor);
                    break;

                case 2:
                    access_element(tensor);
                    break;

                case 3:
                    modify_element(tensor);
                    break;

                case 4:
                {
                    double value;

                    if (!read_input("Enter fill value: ", value))
                        return;

                    tensor.fill(value);

                    cout << "Tensor filled successfully!\n";
                    print_tensor(tensor);
                    break;
                }

                case 5:
                {
                    size_t rank;

                    if (!read_input("Enter new rank (1-10): ", rank))
                        return;

                    if (rank < 1 || rank > 10)
                    {
                        cout << "Rank must be between 1 and 10.\n";
                        break;
                    }

                    vector<size_t> dimensions(rank);

                    for (size_t i = 0; i < rank; i++)
                    {
                        if (!read_input(
                            "Enter dimension " + to_string(i + 1) + ": ",
                            dimensions[i]))
                        {
                            return;
                        }
                    }

                    tensor.reshape(dimensions);

                    cout << "Tensor reshaped successfully!\n";
                    print_tensor(tensor);
                    break;
                }

                case 0:
                    return;

                default:
                    cout << "Invalid choice.\n";
            }
        }
        catch (const exception& e)
        {
            cout << "Operation failed: " << e.what() << "\n";
        }
    }
}

// ---------------- ARITHMETIC OPERATIONS ----------------

void arithmetic_operations()
{
    Tensor a({1});
    Tensor b({1});

    cout << "\nCreate first tensor:\n";

    if (!create_tensor(a))
        return;

    cout << "\nCreate second tensor:\n";

    if (!create_tensor(b))
        return;

    cout << "\n========== ARITHMETIC OPERATIONS ==========\n";
    cout << "1. Addition\n";
    cout << "2. Subtraction\n";
    cout << "3. Elementwise multiplication\n";

    int choice;

    if (!read_input("Enter choice: ", choice))
        return;

    try
    {
        Tensor result({1});

        switch (choice)
        {
            case 1:
                result = add(a, b);
                break;

            case 2:
                result = subtract(a, b);
                break;

            case 3:
                result = multiply(a, b);
                break;

            default:
                cout << "Invalid choice.\n";
                return;
        }

        cout << "\nResult:\n";
        print_tensor(result);
    }
    catch (const exception& e)
    {
        cout << "Operation failed: " << e.what() << "\n";
    }
}

// ---------------- BROADCASTING ----------------

void broadcasting_operations()
{
    Tensor a({1});
    Tensor b({1});

    cout << "\nCreate first tensor:\n";

    if (!create_tensor(a))
        return;

    cout << "\nCreate second tensor:\n";

    if (!create_tensor(b))
        return;

    cout << "\n========== BROADCASTING ==========\n";
    cout << "1. Broadcasting addition\n";
    cout << "2. Broadcasting multiplication\n";

    int choice;

    if (!read_input("Enter choice: ", choice))
        return;

    try
    {
        Tensor result({1});

        switch (choice)
        {
            case 1:
                result = add_broadcast(a, b);
                break;

            case 2:
                result = multiply_broadcast(a, b);
                break;

            default:
                cout << "Invalid choice.\n";
                return;
        }

        cout << "\nBroadcasting result:\n";
        print_tensor(result);
    }
    catch (const exception& e)
    {
        cout << "Broadcasting failed: " << e.what() << "\n";
    }
}

// ---------------- MATRIX MULTIPLICATION ----------------

void matrix_multiplication()
{
    Tensor a({1});
    Tensor b({1});

    cout << "\nCreate first matrix:\n";

    if (!create_tensor(a))
        return;

    cout << "\nCreate second matrix:\n";

    if (!create_tensor(b))
        return;

    cout << "\n========== MATRIX MULTIPLICATION ==========\n";
    cout << "1. Automatic multiplication\n";
    cout << "2. Parallel multiplication\n";
    cout << "3. Blocked multiplication\n";
    cout << "4. Parallel blocked multiplication\n";

    int choice;

    if (!read_input("Enter choice: ", choice))
        return;

    try
    {
        Tensor result({1});

        switch (choice)
        {
            case 1:
                result = matmul(a, b);
                break;

            case 2:
                result = matmul_parallel(a, b);
                break;

            case 3:
            {
                size_t block_size;

                if (!read_input("Enter block size: ", block_size))
                    return;

                result = matmul_blocked(a, b, block_size);
                break;
            }

            case 4:
            {
                size_t block_size;

                if (!read_input("Enter block size: ", block_size))
                    return;

                result = matmul_parallel_blocked(a, b, block_size);
                break;
            }

            default:
                cout << "Invalid choice.\n";
                return;
        }

        cout << "\nMatrix multiplication result:\n";
        print_tensor(result);
    }
    catch (const exception& e)
    {
        cout << "Multiplication failed: " << e.what() << "\n";
    }
}

// ---------------- TRANSPOSE ----------------

void transpose_operation()
{
    Tensor tensor({1});

    if (!create_tensor(tensor))
        return;

    try
    {
        Tensor result = transpose(tensor);

        cout << "\nTransposed tensor:\n";
        print_tensor(result);
    }
    catch (const exception& e)
    {
        cout << "Transpose failed: " << e.what() << "\n";
    }
}

// ---------------- REDUCTIONS ----------------

void reduction_operations()
{
    Tensor tensor({1});

    if (!create_tensor(tensor))
        return;

    cout << "\n========== REDUCTIONS ==========\n";
    cout << "1. Sum\n";
    cout << "2. Mean\n";
    cout << "3. Minimum\n";
    cout << "4. Maximum\n";

    int choice;

    if (!read_input("Enter choice: ", choice))
        return;

    try
    {
        switch (choice)
        {
            case 1:
                cout << "Sum: " << sum(tensor) << "\n";
                break;

            case 2:
                cout << "Mean: " << mean(tensor) << "\n";
                break;

            case 3:
                cout << "Minimum: " << min(tensor) << "\n";
                break;

            case 4:
                cout << "Maximum: " << max(tensor) << "\n";
                break;

            default:
                cout << "Invalid choice.\n";
        }
    }
    catch (const exception& e)
    {
        cout << "Reduction failed: " << e.what() << "\n";
    }
}

// ---------------- MAIN MENU ----------------

int main()
{
    while (true)
    {
        cout << "\n";
        cout << "========================================\n";
        cout << "       MINITENSOR PLAYGROUND\n";
        cout << "========================================\n";

        cout << "1. Tensor Management\n";
        cout << "2. Arithmetic Operations\n";
        cout << "3. Broadcasting Operations\n";
        cout << "4. Matrix Multiplication\n";
        cout << "5. Transpose\n";
        cout << "6. Reduction Operations\n";
        cout << "0. Exit\n";

        cout << "========================================\n";

        int choice;

        if (!read_input("Enter your choice: ", choice))
            break;

        switch (choice)
        {
            case 1:
                tensor_management();
                break;

            case 2:
                arithmetic_operations();
                break;

            case 3:
                broadcasting_operations();
                break;

            case 4:
                matrix_multiplication();
                break;

            case 5:
                transpose_operation();
                break;

            case 6:
                reduction_operations();
                break;

            case 0:
                cout << "\nThank you for using MiniTensor!\n";
                return 0;

            default:
                cout << "Invalid choice. Try again.\n";
        }
    }

    return 0;
}