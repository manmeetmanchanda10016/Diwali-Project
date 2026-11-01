# MiniTensor — Phase 1

A small C++17 tensor library foundation for the Diwali 2026 Mini Tensor Library challenge.

## Phase 1 features

- Tensor creation with arbitrary-rank shapes
- Shape and element-count management
- Contiguous `double` storage using `std::vector`
- Initialization and fill operations
- Safe dimension-product overflow detection
- Deep-copy and move support through standard C++ value semantics
- Unit tests and a small example program

## Project structure

```text
MiniTensor_Phase1/
├── include/minitensor/
│   ├── Shape.h
│   ├── Storage.h
│   └── Tensor.h
├── src/
│   ├── Shape.cpp
│   ├── Storage.cpp
│   └── Tensor.cpp
├── tests/test_phase1.cpp
├── examples/main.cpp
├── CMakeLists.txt
└── README.md
```

## Build on Ubuntu

Install the tools if needed:

```bash
sudo apt update
sudo apt install build-essential cmake
```

Build and run tests:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Run the example:

```bash
./build/basic_example
```

## Open in VS Code

1. Extract the ZIP file.
2. Open VS Code.
3. Select **File → Open Folder**.
4. Select the extracted `MiniTensor_Phase1` folder.
5. Open the integrated terminal and run the build commands above.

You can also open the folder from a terminal with:

```bash
code .
```

## Basic usage

```cpp
#include "minitensor/Tensor.h"

int main() {
    minitensor::Tensor tensor({2, 3}, 5.0);
    tensor.data()[0] = 42.0;
    tensor.fill(1.0);
}
```

## Design notes

- An empty shape `{}` represents a scalar with one element.
- A shape containing a zero dimension, such as `{2, 0, 3}`, has zero elements.
- Tensor elements are stored in one contiguous `std::vector<double>`.
- `data()` exposes a pointer to the first element. Do not use it to access elements when `size() == 0`.
- Copying a tensor copies its data; moving transfers the vector storage.
- This is a Phase 1 foundation. Multidimensional indexing, strides, arithmetic, views, custom allocators and parallel execution are future work.
