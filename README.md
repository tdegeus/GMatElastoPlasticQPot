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
        - [Optimization](#optimization)
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

+   Functions whose name starts with a capital (e.g. `Stress`) return their result (allocating it internally).
+   Functions whose name starts with a small (e.g. `stress`) write to the, fully allocated, (last) input argument (avoiding re-allocation, but making the user responsible to do it properly).

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

Only a partial example is presented here, that is meant to understand the code's structure:

```cpp
#include <GMatElastoPlasticQPot/Cartesian2d.h>

int main()
{
    // a single material point
    // - construct
    GMatElastoPlasticQPot::Cartesian2d::Elastic elastic(K, G);
    GMatElastoPlasticQPot::Cartesian2d::Cusp plastic(K, G, epsy);
    // - set strain (follows e.g. from FEM discretisation)
    GMatElastoPlasticQPot::Tensor2 Eps;
    ...
    // - compute stress [allocate result]
    GMatElastoPlasticQPot::Tensor2 Sig = elastic.Stress(Eps);
    ...
    // - compute stress [no allocation]
    elastic.stress(Eps, Sig); 
    ...

    // a matrix, of shape [nelem, nip]. of material points
    // - construct
    GMatElastoPlasticQPot::Cartesian2d::Elastic matrix(nelem, nip);
    // - set material
    matrix.setElastic(I, K, G);
    matrix.setCusp(I, K, G, epsy);
    // - set strain (follows e.g. from FEM discretisation)
    xt::xtensor<double,4> Eps = xt::empty<double>({nelem, nip, 2ul, 2ul});
    ... 
    // - compute stress [allocate result]
    xt::xtensor<double,4> Sig = matrix.Stress(Eps);
    ...
    // - compute stress [no allocation]
    matrix.stress(Eps, Sig); 
    ...
}
```

>   See [Cartesian2d.h](include/GMatElastoPlasticQPot/Cartesian2d.h) for more details.

## Debugging

To enable assertions define `GMATELASTOPLASTICQPOT_ENABLE_ASSERT` **before** including *GMatElastoPlasticQPot* for the first time. 

Using *CMake* this can be done using the `GMatElastoPlasticQPot::assert` target (see [below](#using-cmake)).

>   To also enable assertions of *xtensor* also define `XTENSOR_ENABLE_ASSERT` **before** including *xtensor* (and *GMatElastoPlasticQPot*) for the first time. 
>   
>   Using *CMake* all assertions are enabled using the `GMatElastoPlasticQPot::debug` target (see [below](#using-cmake)).

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

Note that *xsimd* and hardware optimisations are not enabled. To enable them you have to compile on your system, as is discussed next.

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

>   You can also use a single call:
>   
>   ```
>   python -m pip install .
>   ```
>   
>   However potentially leading to less readable output.

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
    Includes *GMatElastoPlasticQPot* and the *xtensor* dependency.

*   `GMatElastoPlasticQPot::assert`
    Enables assertions by defining `GMATELASTOPLASTICQPOT_ENABLE_ASSERT`.

*   `GMatElastoPlasticQPot::debug`
    Enables all assertions by defining `GMATELASTOPLASTICQPOT_ENABLE_ASSERT` and `XTENSOR_ENABLE_ASSERT`.

*   `GMatElastoPlasticQPot::compiler_warings`
    Enables compiler warnings (generic).

### Optimization

It is advised to think about compiler optimization and about enabling *xsimd. In *CMake* this can be done using the `xtensor::optimize` and `xtensor::use_xsimd` targets. The above example then becomes:

```cmake
cmake_minimum_required(VERSION 3.1)
project(example)
find_package(GMatElastoPlasticQPot REQUIRED)
add_executable(example example.cpp)
target_link_libraries(example PRIVATE 
    GMatElastoPlasticQPot 
    xtensor::optimize 
    xtensor::use_xsimd)
```

See the [documentation of xtensor](https://xtensor.readthedocs.io/en/latest/) concerning optimization.

## By hand

Presuming that the compiler is `c++`, compile using:

```
c++ -I/path/to/GMatElastoPlasticQPot/include ...
```

Note that you have to take care of the *xtensor* dependency, the C++ version, optimization, enabling *xsimd*, ...

## Using pkg-config

Presuming that the compiler is `c++`, compile using:

```
c++ `pkg-config --cflags GMatElastoPlasticQPot` ...
```

Note that you have to take care of the *xtensor* dependency, the C++ version, optimization, enabling *xsimd*, ...

# References / Credits

*   The model is described in *T.W.J. de Geus, M. Popović, W. Ji, A. Rosso, M. Wyart. How collective asperity detachments nucleate slip at frictional interfaces. Proceedings of the National Academy of Sciences, 2019, 201906551. [doi: 10.1073/pnas.1906551116](https://doi.org/10.1073/pnas.1906551116), [arXiv: 1904.07635](http://arxiv.org/abs/1904.07635)*.

*   [xtensor](https://github.com/QuantStack/xtensor) is used under the hood.
