# GMatElastoPlasticQPot

[![Travis](https://travis-ci.com/tdegeus/GMatElastoPlasticQPot.svg?branch=master)](https://travis-ci.com/tdegeus/GMatElastoPlasticQPot)

Elasto-plastic material model based on a manifold of quadratic potentials. An overview of the theory can be found in `docs/` in particular in this [PDF](docs/readme.pdf).

# Contents

<!-- MarkdownTOC -->

- [Disclaimer](#disclaimer)
- [Implementation](#implementation)
    - [Overview](#overview)
    - [Conventions](#conventions)
    - [Example](#example)
    - [Debugging](#debugging)
- [Installation](#installation)
    - [C++ headers](#c-headers)
        - [Using conda](#using-conda)
        - [From source](#from-source)
    - [Python module](#python-module)
        - [Using conda](#using-conda-1)
        - [From source](#from-source-1)
- [Compiling](#compiling)
    - [Using CMake](#using-cmake)
        - [Example](#example-1)
        - [Targets](#targets)
    - [By hand](#by-hand)
    - [Using pkg-config](#using-pkg-config)
- [References / Credits](#references--credits)

<!-- /MarkdownTOC -->

# Disclaimer

This library is free to use under the [MIT license](https://github.com/tdegeus/GMatElastoPlasticQPot/blob/master/LICENSE). Any additions are very much appreciated, in terms of suggested functionality, code, documentation, testimonials, word-of-mouth advertisement, etc. Bug reports or feature requests can be filed on [GitHub](https://github.com/tdegeus/GMatElastoPlasticQPot). As always, the code comes with no guarantee. None of the developers can be held responsible for possible mistakes.

Download: [.zip file](https://github.com/tdegeus/GMatElastoPlasticQPot/zipball/master) | [.tar.gz file](https://github.com/tdegeus/GMatElastoPlasticQPot/tarball/master).

(c - [MIT](https://github.com/tdegeus/GMatElastoPlasticQPot/blob/master/LICENSE)) T.W.J. de Geus (Tom) | tom@geus.me | www.geus.me | [github.com/tdegeus/GMatElastoPlasticQPot](https://github.com/tdegeus/GMatElastoPlasticQPot)

# Implementation

## Overview

The headers are meant to be self-explanatory, please inspect them:

+   [Cartesian2d.h](include/GMatElastoPlasticQPot/Cartesian2d.h)

## Conventions

Naming conventions

+   Functions with a capital first letter (e.g. "Stress") return their result.
+   Functions with a small first letter (e.g. "stress") write to allocated final input argument.

Storage conventions

+   Scalar
    ```cpp
    double
    ```

+   2nd-order tensor
    ```cpp
    Tensor2 = xt::xtensor_fixed<double, xt::xshape<2,2>>
    ```

+   List *(a)* of second order tensors *(i,j)* : *A(a,i,j)*
    ```cpp
    xt::xtensor<double,3>
    ```

+   Matrix *(a,b)* of second order tensors *(i,j)* : *A(a,b,i,j)*
    ```cpp
    xt::xtensor<double,4>
    ```

## Example

Only a tiny example is presented here, that is meant to understand the code's structure:

```cpp
#include <GMatElastoPlasticQPot/Cartesian2d.h>

int main()
{
    // a single material point
    // - create class
    GMatElastoPlasticQPot::Cartesian2d::Elastic elastic(K, G);
    GMatElastoPlasticQPot::Cartesian2d::Cusp plastic(K, G, epsy);
    // - compute stress [allocate result]
    Sig = elastic.Stress(Eps);
    ...
    // - compute stress [no allocation]
    elastic.stress(Eps, Sig); 
    ...

    // a "matrix" of material points
    // - create class
    GMatElastoPlasticQPot::Cartesian2d::Elastic matrix(nelem, nip);
    // - set material
    matrix.setElastic(I, K, G);
    matrix.setCusp(I, K, G, epsy);
    // - compute stress [allocate result]
    Sig = matrix.Stress(Eps);
    ...
    // - compute stress [no allocation]
    matrix.stress(Eps, Sig); 
    ...
}
```

## Debugging

Assertions are enabled:

+   If `NDEBUG` is **not** defined. 
+   If `GMATELASTOPLASTICQPOT_ENABLE_ASSERT` is defined **before** including *GMatElastoPlasticQPot* for the first time.

Otherwise assertions are not enabled. To explicitly disable them define `GMATELASTOPLASTICQPOT_DISABLE_ASSERT` **before** including *GMatElastoPlasticQPot* for the first time.

# Installation

## C++ headers

### Using conda

```bash
conda install -c conda-forge gmatelastoplasticqpot
```

### From source

```bash
# Download GMatElastoPlasticQPot
git checkout https://github.com/tdegeus/GMatElastoPlasticQPot.git
cd GMatElastoPlasticQPot

# Install headers, CMake and pkg-config support
cmake .
make install
```

## Python module

### Using conda

```bash
conda install -c conda-forge python-gmatelastoplasticqpot
```

### From source

>   To get the prerequisites you *can* use conda
> 
>   ```bash
>   conda install -c conda-forge pyxtensor
>   conda install -c conda-forge xsimd
>   ```

```bash
# Download GMatElastoPlasticQPot
git checkout https://github.com/tdegeus/GMatElastoPlasticQPot.git
cd GMatElastoPlasticQPot

# Compile and install the Python module
python setup.py build
python setup.py install
```

# Compiling

## Using CMake

### Example

Using *GMatElastoPlasticQPot* your `CMakeLists.txt` can be as follows

```cmake
cmake_minimum_required(VERSION 3.1)
project(example)
find_package(GMatElastoPlasticQPot REQUIRED)
add_executable(example example.cpp)
target_link_libraries(example PRIVATE GMatElastoPlasticQPot)
```

### Targets

The following targets are available:

*   `GMatElastoPlasticQPot`
    Includes the *xtensor* dependency.

*   `GMatElastoPlasticQPot::assert`
    Enables assertions by defining `GMATELASTOPLASTICQPOT_ENABLE_ASSERT`.

*   `GMatElastoPlasticQPot::compiler_warings`
    Enables compiler warnings (generic).

## By hand

Presuming that the compiler is `c++`, compile using:

```
c++ -I/path/to/GMatElastoPlasticQPot/include ...
```

## Using pkg-config

Presuming that the compiler is `c++`, compile using:

```
c++ `pkg-config --cflags GMatElastoPlasticQPot` ...
```

# References / Credits

*   The model is described in *T.W.J. de Geus, M. Popović, W. Ji, A. Rosso, M. Wyart. How collective asperity detachments nucleate slip at frictional interfaces. Proceedings of the National Academy of Sciences, 2019, 201906551. [doi: 10.1073/pnas.1906551116](https://doi.org/10.1073/pnas.1906551116), [arXiv: 1904.07635](http://arxiv.org/abs/1904.07635)*.

*   [xtensor](https://github.com/QuantStack/xtensor) is used under the hood.
