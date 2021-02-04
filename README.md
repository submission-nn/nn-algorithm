About 
-----
This repository contains a header only implementation of the paper "A faster algorithm for finding closet pairs in hamming metric". 

Setup
-----
Arch Linux
```bash
sudo pacman -S hdf5 gtest
```

Ubuntu:
You need to follow [this](https://gist.github.com/Cartexius/4c437c084d6e388288201aadf9c8cdd5) guide to install the googletest suite.
```bash
sudo apt install hdf5lib gtest
```

Build:
------
arch linux
```bash
git clone --recurse-submodules https://github.com/submission-nn/nn-algorithm
cd optimalnn
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make 
```

If you use Ubuntu, you have to replace the two last lines with the following to ensure that you have a C++17 ready compiler.
```bash
cmake -DCMAKE_BUILD_TYPE=Release -D CMAKE_C_COMPILER=clang-11 -D CMAKE_CXX_COMPILER=clang++-11  ..
make 
```
Informations:
-----
Currently we have 2 implementations:
- __QUADRATIC__     naive quadratic search
- __WINDOWED__      our new algorithm

Compiler Flags 
------
You cann pass additional preprocessor flags to the cmake process:
```cpp
#define LOGGING             // upon definition A LOT of debug information will be printed
#define CSV_LOGGING         // Same as 'LOGGING' but will print everything as a CSV file. DO NOT use TOGETHER WITH 'LOGGING'
#define SOLUTION_LOGGING    // print additionally to the normal debug log every solution found. 
#define PERFOMANCE_LOGGING  // print some information about the max three depth,.... Extends the `LOGGING`
#define COMPRESS_OUTPUT     // compresses some logging output
#define ALL_DELTA           // Instead of filtering all elements which do not have weight wt(x+z) = \delta k, we allow each element to have weight wt(x+z) <= \delta k
```

Run Our Benchmarks
-----

For $n=64$ run:
```bash
cd build
./test/nn/nn_golden_64 -gtest_filter=GoldenNearestNeighbor_64_10.Windowed:GoldenNearestNeighbor_64_10/*.Windowed:*/GoldenNearestNeighbor_64_10
./test/nn/nn_golden_64 -gtest_filter=GoldenNearestNeighbor_64_15.Windowed:GoldenNearestNeighbor_64_15/*.Windowed:*/GoldenNearestNeighbor_64_15
./test/nn/nn_golden_64 -gtest_filter=GoldenNearestNeighbor_64_20.Windowed:GoldenNearestNeighbor_64_20/*.Windowed:*/GoldenNearestNeighbor_64_20
./test/nn/nn_golden_64 --gtest_filter=GoldenNearestNeighbor_64_10.WindowedWithEpsilon:GoldenNearestNeighbor_64_10/*.WindowedWithEpsilon:*/GoldenNearestNeighbor_64_10
./test/nn/nn_golden_64 --gtest_filter=GoldenNearestNeighbor_64_15.WindowedWithEpsilon:GoldenNearestNeighbor_64_15/*.WindowedWithEpsilon:*/GoldenNearestNeighbor_64_15
./test/nn/nn_golden_64 --gtest_filter=GoldenNearestNeighbor_64_20.WindowedWithEpsilon:GoldenNearestNeighbor_64_20/*.WindowedWithEpsilon:*/GoldenNearestNeighbor_64_20
```

For $n=128$ run:
```bash
cd build
./test/nn/nn_golden_64 -gtest_filter=GoldenNearestNeighbor_128_10.Windowed:GoldenNearestNeighbor_128_10/*.Windowed:*/GoldenNearestNeighbor_128_10
./test/nn/nn_golden_64 -gtest_filter=GoldenNearestNeighbor_128_15.Windowed:GoldenNearestNeighbor_128_15/*.Windowed:*/GoldenNearestNeighbor_128_15
./test/nn/nn_golden_64 -gtest_filter=GoldenNearestNeighbor_128_20.Windowed:GoldenNearestNeighbor_128_20/*.Windowed:*/GoldenNearestNeighbor_128_20
./test/nn/nn_golden_64 --gtest_filter=GoldenNearestNeighbor_128_10.WindowedWithEpsilon:GoldenNearestNeighbor_128_10/*.WindowedWithEpsilon:*/GoldenNearestNeighbor_128_10
./test/nn/nn_golden_64 --gtest_filter=GoldenNearestNeighbor_128_15.WindowedWithEpsilon:GoldenNearestNeighbor_128_15/*.WindowedWithEpsilon:*/GoldenNearestNeighbor_128_15
./test/nn/nn_golden_64 --gtest_filter=GoldenNearestNeighbor_128_20.WindowedWithEpsilon:GoldenNearestNeighbor_128_20/*.WindowedWithEpsilon:*/GoldenNearestNeighbor_128_20
```
For $n=256$ run:
```bash
cd build
./test/nn/nn_golden_64 -gtest_filter=GoldenNearestNeighbor_256_10.Windowed:GoldenNearestNeighbor_256_10/*.Windowed:*/GoldenNearestNeighbor_256_10
./test/nn/nn_golden_64 -gtest_filter=GoldenNearestNeighbor_256_15.Windowed:GoldenNearestNeighbor_256_15/*.Windowed:*/GoldenNearestNeighbor_256_15
./test/nn/nn_golden_64 -gtest_filter=GoldenNearestNeighbor_256_20.Windowed:GoldenNearestNeighbor_256_20/*.Windowed:*/GoldenNearestNeighbor_256_20
./test/nn/nn_golden_64 --gtest_filter=GoldenNearestNeighbor_256_10.WindowedWithEpsilon:GoldenNearestNeighbor_256_10/*.WindowedWithEpsilon:*/GoldenNearestNeighbor_256_10
./test/nn/nn_golden_64 --gtest_filter=GoldenNearestNeighbor_256_15.WindowedWithEpsilon:GoldenNearestNeighbor_256_15/*.WindowedWithEpsilon:*/GoldenNearestNeighbor_256_15
./test/nn/nn_golden_64 --gtest_filter=GoldenNearestNeighbor_256_20.WindowedWithEpsilon:GoldenNearestNeighbor_256_20/*.WindowedWithEpsilon:*/GoldenNearestNeighbor_256_20
```

Project Hierarchy
-----
The code of our algorithm is in the directory `src/*`. 
All the fundemantal data functionality, e.g. compare, xor, ... is implemented in `container.h` In `nn.h` is an implementation of the naive quadratic search and some helper functions. Our Algorithm is implemented in `windowed_nn_v2.h` in the class `WindowedNearestNeighbor2` which inherits from `NearestNeighbor` which implements the quadratic search. 
