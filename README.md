# GMatElastoPlasticQPot

[![Travis](https://travis-ci.com/tdegeus/GMatElastoPlasticQPot.svg?branch=master)](https://travis-ci.com/tdegeus/GMatElastoPlasticQPot)

Elasto-plastic material model based on a manifold of quadratic potentials. 
An overview of the theory can be found in `docs/readme.tex` 
conveniently compiled to this [PDF](docs/readme.pdf).

# Contents

<!-- MarkdownTOC levels="1,2,3" -->

- [Disclaimer](#disclaimer)
- [Implementation](#implementation)
    - [C++ and Python](#c-and-python)
    - [Cartesian2d](#cartesian2d)
        - [Overview](#overview)
        - [Function names](#function-names)
        - [Storage](#storage)
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
        - [Optimisation](#optimisation)
    - [By hand](#by-hand)
    - [Using pkg-config](#using-pkg-config)
- [References / Credits](#references--credits)

<!-- /MarkdownTOC -->

# Disclaimer

This library is free to use under the [MIT license](https://github.com/tdegeus/GMatElastoPlasticQPot/blob/master/LICENSE).
Any additions are very much appreciated, in terms of suggested functionality, code, 
documentation, testimonials, word-of-mouth advertisement, etc.
Bug reports or feature requests can be filed on [GitHub](https://github.com/tdegeus/GMatElastoPlasticQPot).
As always, the code comes with no guarantee.
None of the developers can be held responsible for possible mistakes.

Download: 
[.zip file](https://github.com/tdegeus/GMatElastoPlasticQPot/zipball/master) | 
[.tar.gz file](https://github.com/tdegeus/GMatElastoPlasticQPot/tarball/master).

(c - [MIT](https://github.com/tdegeus/GMatElastoPlasticQPot/blob/master/LICENSE)) 
T.W.J. de Geus (Tom) | tom@geus.me | www.geus.me | 
[github.com/tdegeus/GMatElastoPlasticQPot](https://github.com/tdegeus/GMatElastoPlasticQPot)

# Implementation

## C++ and Python

The code is a C++ header-only library (see [installation notes](#c-headers)), 
but a Python module is also provided (see [installation notes](#python-module)).
The interfaces are identical except:

+   All *xtensor* objects (`xt::xtensor<...>`) are *NumPy* arrays in Python. 
    Overloading based on rank is also available in Python.
+   The Python module cannot change output objects in-place: 
    only functions whose name starts with a capital letter are included, see below.
+   All `::` in C++ are `.` in Python.

## Cartesian2d

[Cartesian2d.h](include/GMatElastoPlasticQPot/Cartesian2d.h)

### Overview

At the material point level different models are implemented with different classes:

+   `Elastic`: linear elastic material model that corresponds to 
    the elastic part of the elasto-plastic material model.
+   `Cusp`: the elasto-plastic material model defined by cusp potentials.
+   `Smooth`: the elasto-plastic material model defined by smoothed potentials. 

There is a `Matrix` class that allows you to combine all these material models 
and have a single API for a matrix of material points. 

### Function names

+   Functions whose name starts with a capital letter (e.g. `Stress`) 
    return their result (allocating it internally).
+   Functions whose name starts with a small letter (e.g. `stress`) 
    write to the, fully allocated, last input argument(s) 
    (avoiding re-allocation, but making the user responsible to do it properly).

### Storage

+   Scalar
    ```cpp
    double
    ```

+   2nd-order tensor
    ```cpp
    xt::xtensor_fixed<double, xt::xshape<2,2>> = 
    GMatElastoPlasticQPot::Cartesian2d::Tensor2
    ```

+   List *(i)* of second order tensors *(x,y)* : *A(i,x,y)*
    ```cpp
    xt::xtensor<double,3>
    ```
    Note that the shape is `[I, 2, 2]`.

+   Matrix *(i,j)* of second order tensors *(x,y)* : *A(i,j,x,y)*
    ```cpp
    xt::xtensor<double,4>
    ```
    Note that the shape is `[I, J, 2, 2]`.

### Example

Only a partial examples are presented here, meant to understand the code's structure.

#### Individual material points

```cpp
#include <GMatElastoPlasticQPot/Cartesian2d.h>

namespace GMat = GMatElastoPlasticQPot::Cartesian2d;

int main()
{
    // a single material point
    GMat::Elastic elastic(K, G);
    GMat::Cusp plastic(K, G, epsy);
    ...
    
    // set strain (follows e.g. from FEM discretisation)
    GMat::Tensor2 Eps;
    ...
    
    // compute stress (including allocation of the result)
    GMat::Tensor2 Sig = elastic.Stress(Eps);
    // OR compute stress without (re)allocating the results
    // in this case "Sig" has to be of the correct type and shape
    elastic.stress(Eps, Sig); 
    ...

    return 0;
}
```

#### Matrix of material points

```cpp
#include <GMatElastoPlasticQPot/Cartesian2d.h>

namespace GMat = GMatElastoPlasticQPot::Cartesian2d;

int main()
{
    // a matrix, of shape [nelem, nip], of material points
    GMatElastoPlasticQPot::Cartesian2d::Elastic matrix(nelem, nip);

    // set materials:
    // points where I(x,y) == 1 are assigned, points where I(x,y) == 0 are skipped
    // all points can only be assigned once
    matrix.setElastic(I, K, G);
    matrix.setCusp(I, K, G, epsy);

    // compute stress (including allocation of the result)
    xt::xtensor<double,4> eps = xt::empty<double>({nelem, nip, 2ul, 2ul});
    ... 

    // compute stress [allocate result]
    xt::xtensor<double,4> sig = matrix.Stress(eps);
    // OR compute stress without (re)allocating the results
    // in this case "sig" has to be of the correct type and shape
    matrix.stress(eps, sig); 
    ...

    return 0;
}
```

## Debugging

To enable assertions define `GMATELASTOPLASTICQPOT_ENABLE_ASSERT` 
**before** including *GMatElastoPlasticQPot* for the first time. 

Using *CMake* this can be done using the `GMatElastoPlasticQPot::assert` target 
(see [below](#using-cmake)).

>   To also enable assertions of *xtensor* also define `XTENSOR_ENABLE_ASSERT` 
>   **before** including *xtensor* (and *GMatElastoPlasticQPot*) for the first time. 
>   
>   Using *CMake* all assertions are enabled using the `GMatElastoPlasticQPot::debug` target 
>   (see [below](#using-cmake)).

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

Note that *xsimd* and hardware optimisations are **not enabled**. 
To enable them you have to compile on your system, as is discussed next.

### From source

>   You need *xtensor*, *pyxtensor* and optionally *xsimd* as prerequisites. 
>   Additionally, Python needs to know how to find them. 
>   The easiest is to use *conda* to get the prerequisites:
> 
>   ```bash
>   conda install -c conda-forge pyxtensor
>   conda install -c conda-forge xsimd
>   ```
>   
>   If you then compile and install with the same environment 
>   you should be good to go. 
>   Otherwise, a bit of manual labour might be needed to
>   treat the dependencies.

```bash
# Download GMatElastoPlasticQPot
git checkout https://github.com/tdegeus/GMatElastoPlasticQPot.git
cd GMatElastoPlasticQPot

# Compile and install the Python module
python setup.py build
python setup.py install
# OR you can use one command (but with less readable output)
python -m pip install .
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
    Includes *GMatElastoPlasticQPot* and the *xtensor* dependency.

*   `GMatElastoPlasticQPot::assert`
    Enables assertions by defining `GMATELASTOPLASTICQPOT_ENABLE_ASSERT`.

*   `GMatElastoPlasticQPot::debug`
    Enables all assertions by defining `GMATELASTOPLASTICQPOT_ENABLE_ASSERT` and `XTENSOR_ENABLE_ASSERT`.

*   `GMatElastoPlasticQPot::compiler_warings`
    Enables compiler warnings (generic).

### Optimisation

It is advised to think about compiler optimization and enabling *xsimd*. 
Using *CMake* this can be done using the `xtensor::optimize` and `xtensor::use_xsimd` targets. 
The above example then becomes:

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

*   The model is described in 
    *T.W.J. de Geus, M. Popović, W. Ji, A. Rosso, M. Wyart. 
    How collective asperity detachments nucleate slip at frictional interfaces. 
    Proceedings of the National Academy of Sciences, 2019, 201906551. 
    [doi: 10.1073/pnas.1906551116](https://doi.org/10.1073/pnas.1906551116), 
    [arXiv: 1904.07635](http://arxiv.org/abs/1904.07635)*.

*   [xtensor](https://github.com/QuantStack/xtensor) is used under the hood.
