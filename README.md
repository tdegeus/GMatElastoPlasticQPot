# GMatElastoPlasticQPot

[![Travis](https://travis-ci.com/tdegeus/GMatElastoPlasticQPot.svg?branch=master)](https://travis-ci.com/tdegeus/GMatElastoPlasticQPot)
[![Build status](https://ci.appveyor.com/api/projects/status/6qb7cgh3b7ant3a7?svg=true)](https://ci.appveyor.com/project/tdegeus/gmatelastoplasticqpot)

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
- [Testing & Benchmarking](#testing--benchmarking)
    - [Basic testing](#basic-testing)
    - [Basic benchmarking](#basic-benchmarking)
    - [Extensive testing](#extensive-testing)
- [Upgrading instructions](#upgrading-instructions)
    - [Upgrading to v0.6.*](#upgrading-to-v06)
- [Change-log](#change-log)
    - [v0.7.0](#v070)
    - [v0.6.4](#v064)
    - [v0.6.3](#v063)
    - [v0.6.2](#v062)
    - [v0.6.1](#v061)
    - [v0.6.0](#v060)
    - [v0.5.0](#v050)
    - [v0.4.0](#v040)
    - [v0.3.0](#v030)
    - [v0.2.2](#v022)
- [v0.2.1](#v021)

<!-- /MarkdownTOC -->

# Disclaimer

This library is free to use under the
[MIT license](https://github.com/tdegeus/GMatElastoPlasticQPot/blob/master/LICENSE).
Any additions are very much appreciated, in terms of suggested functionality, code,
documentation, testimonials, word-of-mouth advertisement, etc.
Bug reports or feature requests can be filed on
[GitHub](https://github.com/tdegeus/GMatElastoPlasticQPot).
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

There is a `Matrix` class that allows you to combine all these material models and 
have a single API for a matrix of material points. 

>   Note that all strain tensors are presumed symmetric. 
>   No checks are made to ensure this.

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
    or
    ```cpp
    xt::xtensor<double, 0>
    ```

+   2nd-order tensor
    ```cpp
    xt::xtensor_fixed<double, xt::xshape<2, 2>> = 
    GMatElastoPlasticQPot::Cartesian2d::Tensor2
    ```
    or 
    ```cpp
    xt:xtensor<double, 2>
    ```

+   List *(i)* of second order tensors *(x,y)* : *A(i,x,y)*
    ```cpp
    xt::xtensor<double, 3>
    ```
    Note that the shape is `[I, 2, 2]`.

+   Matrix *(i,j)* of second order tensors *(x,y)* : *A(i,j,x,y)*
    ```cpp
    xt::xtensor<double, 4>
    ```
    Note that the shape is `[I, J, 2, 2]`.

+   Etc.

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
    elastic.setStrain(Eps);
    ...
    
    // compute stress (including allocation of the result)
    GMat::Tensor2 Sig = elastic.Stress();
    // OR compute stress without (re)allocating the results
    // in this case "Sig" has to be of the correct type and shape
    elastic.stress(Sig); 
    ...

    return 0;
}
```

#### Array of material points

```cpp
#include <GMatElastoPlasticQPot/Cartesian2d.h>

namespace GMat = GMatElastoPlasticQPot::Cartesian2d;

int main()
{
    // a array, of shape [nelem, nip], of material points
    GMat::Array<2> array({nelem, nip});

    // set materials:
    // points where I(x,y) == 1 are assigned, points where I(x,y) == 0 are skipped
    // all points can only be assigned once
    array.setElastic(I, K, G);
    array.setCusp(I, K, G, epsy);
    ...

    // set strain tensor (follows e.g. from FEM discretisation)
    xt::xtensor<double,4> eps = xt::empty<double>({nelem, nip, 2ul, 2ul});
    ... 
    array.setStrain(eps);

    // compute stress (allocate result)
    xt::xtensor<double,4> sig = array.Stress();
    // OR compute stress without (re)allocating the results
    // in this case "sig" has to be of the correct type and shape
    array.stress(sig); 
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

>   The library's assertions are enabled in the Python interface, 
>   but debugging with *xtensor* is disabled.

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
    Enables all assertions by defining 
    `GMATELASTOPLASTICQPOT_ENABLE_ASSERT` and `XTENSOR_ENABLE_ASSERT`.

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

Note that you have to take care of the *xtensor* dependency, the C++ version, optimization, 
enabling *xsimd*, ...

## Using pkg-config

Presuming that the compiler is `c++`, compile using:

```
c++ `pkg-config --cflags GMatElastoPlasticQPot` ...
```

Note that you have to take care of the *xtensor* dependency, the C++ version, optimization, 
enabling *xsimd*, ...

# References / Credits

*   The model is described in 
    *T.W.J. de Geus, M. Popović, W. Ji, A. Rosso, M. Wyart. 
    How collective asperity detachments nucleate slip at frictional interfaces. 
    Proceedings of the National Academy of Sciences, 2019, 201906551. 
    [doi: 10.1073/pnas.1906551116](https://doi.org/10.1073/pnas.1906551116), 
    [arXiv: 1904.07635](http://arxiv.org/abs/1904.07635)*.

*   [xtensor](https://github.com/QuantStack/xtensor) is used under the hood.

# Testing & Benchmarking

## Basic testing

>   Run by the continuous integration

```
cd build
cmake .. -DBUILD_TESTS=1
make
./test/main
```

## Basic benchmarking

>   Compiled by the continuous integration

```
cd build
cmake .. -DBUILD_EXAMPLES=1
make
./examples/benchmark-epsd
./examples/benchmark-cusp
./examples/benchmark-matrix
```

## Extensive testing

>   Run by the continuous integration

To make sure that the current version in up-to-date with old versions,
one starts by generating a set or random states using the current version:

```
cd test/compare_versions
python Cartesian2d_generate.py
```

And then checks that the generated states are also found with previous
versions:

```
git checkout tags/v0.6.2
python setup.py build
python setup.py install
python Cartesian2d_check_v0.6.2.py
```

and likewise for

```
python Cartesian2d_check_v0.5.0.py
```

If no assertions are found each time the code should be behaving as supposed to. 
Please feel free to contribute additional tests. 

# Upgrading instructions

## Upgrading to v0.6.*

Compared to v0.5.0, v0.6.1 has some generalisations and efficiency updates. 
This requires the following changes:

*   `Matrix` has been generalised to `Array<rank>`. Practically this requires changing:
    -   `Matrix` to `Array<2>` in C++.
    -   `Matrix` to `Array2d` in Python. 
        Note that `Array1d`, `Array3d`, are also available.

*   Strain is now stored as a member. 
    Functions like `stress` now return the state based on the last specified strain, 
    specified using `setStrain(Esp)`. This leads to the following changes:
    - `stress`: no argument.
    - `tangent`: no argument, single return value (no longer returns stress).
    - `find`: no argument, renamed to `currentIndex`.
    - `epsy`: replaced by `currentYieldLeft` and `currentYieldRight`.

*   By storing strain as a member, 
    efficiency upgrades have been made to find the position in the potential energy landscape. 
    The library therefore now depends on [QPot](https://www.github.com/tdegeus/QPot).

# Change-log

## v0.7.0

*   Exposing "checkYieldBoundLeft" and "checkYieldBoundRight" from QPot v0.2.0.
*   Better float testing using Catch2.
*   Adding dependencies in `CMakeLists.txt` of the examples.
*   Minor style and comments fix.

## v0.6.4

*   Increasing robustness using the new xtensor's `has_fixed_rank` and `get_rank`.
*   Using the new xtensor's `has_shape`
*   Added examples for friction.
*   Added examples for old versions (helps benchmarking and facilitates upgrading).

## v0.6.3

*   Bugfix: Removing alias from material points incompatible with the new storage of stress etc inside the material point.

## v0.6.2

*   Minor bugfix in Python API: calling auto-allocation function.
*   Adding additional test mechanism, comparing to old releases.

## v0.6.1

*   Adding `Array1d` and `Array3d` to Python module.
*   Code-style updates

## v0.6.0

Compared to v0.5.0, v0.6.0 has some generalisations and efficiency updates. 
This requires the following changes:

*   `Matrix` has been generalised to `Array<rank>`. Practically this requires changing:
    -   `Matrix` to `Array<2>` in C++.
    -   `Matrix` to `Array2d` in Python. 
        Note that `Array1d`, `Array3d`, etc. are for the moment not compiled, 
        but can be made available upon request.

*   Strain is now stored as a member. 
    Functions like `stress` now return the state based on the last specified strain, 
    specified using `setStrain(Esp)`. This leads to the following changes:
    - `stress`: no argument.
    - `tangent`: no argument, single return value (no longer returns stress).
    - `find`: no argument, renamed to `currentIndex`.
    - `epsy`: replaced by `currentYieldLeft` and `currentYieldRight`.

*   By storing strain as a member, 
    efficiency upgrades have been made to find the position in the potential energy landscape. 
    The library therefore now depends on [QPot](https://www.github.com/tdegeus/QPot).

## v0.5.0

*   Added 'tangent' which gives the linear response using a fourth order tangent.

## v0.4.0

*   Cleanup all code, applying rules from `.clang-format` (and pep8 for Python).
*   Rewriting readme.
*   Improving syntax of `epsy` for material classes.
*   Adding `isElastic()`, `isCusp()`, and `isSmooth`

## v0.3.0

*   Enabling assertion only when explicitly defining `GMATELASTOPLASTICQPOT_ENABLE_ASSERT`
*   Introducing extra CMake targets
*   Reformatting code
*   Reformatting CMake
*   Reformatting tests
*   Updating readme
*   Updating CI
*   Reformatting `docs/readme.tex`

## v0.2.2

*   Updating CMake
*   Updating Python build
*   Making xsimd 'optional' for Python build: runs in xsimd is found
*   Minor code-style updates

# v0.2.1

*   Added minimal documentation.

