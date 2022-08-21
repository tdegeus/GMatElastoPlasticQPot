# GMatElastoPlasticQPot

[![CI](https://github.com/tdegeus/GMatElastoPlasticQPot/workflows/CI/badge.svg)](https://github.com/tdegeus/GMatElastoPlasticQPot/actions)
[![Doxygen -> gh-pages](https://github.com/tdegeus/GMatElastoPlasticQPot/workflows/gh-pages/badge.svg)](https://tdegeus.github.io/GMatElastoPlasticQPot)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/gmatelastoplasticqpot.svg)](https://anaconda.org/conda-forge/gmatelastoplasticqpot)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/python-gmatelastoplasticqpot.svg)](https://anaconda.org/conda-forge/python-gmatelastoplasticqpot)

Elasto-plastic material model based on a manifold of quadratic potentials.
An overview of the theory can be found in `docs/readme.tex`
conveniently compiled to this [PDF](docs/notes/readme.pdf).

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

# Python implementation

## Partial example

```python
import GMatLinearElastic.Cartesian3d as GMat

shape = [...]
K = np.empty(shape)
G = np.empty(shape)
epsy = np.empty(shape + [1])
...

GMat.CuspXd model(K, G, epsy);
...

Eps = np.empty(shape + [3, 3])
...

model.Eps = Eps
print(model.Sig)
```

## Installation

### Using conda

```bash
conda install -c conda-forge python-gmatelastoplasticqpot
```

Note that *xsimd* and hardware optimisations are **not enabled**.
To enable them you have to compile on your system, as is discussed next.

### From source

>   You need *xtensor*, *xtensor-python* and optionally *xsimd* as prerequisites.
>   The easiest is to use *conda* to get the prerequisites:
>
>   ```bash
>   conda install -c conda-forge xtensor xsimd xtensor-python
>   ```
>
>   If you then compile and install with the same environment you should be good to go.
>   Otherwise, a bit of manual labour might be needed to treat the dependencies.

```bash
git checkout https://github.com/tdegeus/GMatElastoPlasticQPot.git
cd GMatElastoPlasticQPot

# Only if you want to use hardware optimisation:
export SKBUILD_CONFIGURE_OPTIONS="-DUSE_SIMD=1"

python -m pip install . -v
```

# C++ implementation

## Partial example

```cpp
#include <GMatElastoPlasticQPot/Cartesian2d.h>

namespace GMat = GMatElastoPlasticQPot::Cartesian2d;

int main()
{
    static const size_t rank = ...;

    xt::xtensor<double, rank> K = ...;
    xt::xtensor<double, rank> G = ...;
    xt::xtensor<double, rank + 1> epsy = ...;

    GMat::CuspXd model(K, G, epsy);
    ...

    xt::xtensor<double, rank + 2> Eps;
    ...

    // all necessary computation are done at this point
    model.set_Eps(Eps);
    ...

    // get reference to stress
    auto Sig = elastic.Sig();

    return 0;
}
```

## Debugging

To enable assertions define `GMATELASTOPLASTICQPOT_ENABLE_ASSERT`
**before** including *GMatElastoPlasticQPot* for the first time.

Using *CMake* this can be done using the `GMatElastoPlasticQPot::assert` target.

>   To also enable assertions of *xtensor* also define `XTENSOR_ENABLE_ASSERT`
>   **before** including *xtensor* (and *GMatElastoPlasticQPot*) for the first time.
>
>   Using *CMake* all assertions are enabled using the `GMatElastoPlasticQPot::debug` target.

## Installation

### Using conda

```bash
conda install -c conda-forge gmatelastoplasticqpot
```

### From source

```bash
git checkout https://github.com/tdegeus/GMatElastoPlasticQPot.git
cd GMatElastoPlasticQPot

cmake -Bbuild
cd build
cmake --install .
```

## Compiling

## Using CMake

### Example

Your `CMakeLists.txt` can be as follows

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

It is advised to think about compiler optimisation and enabling *xsimd*.
Using *CMake* this can be done using the `xtensor::optimize` and `xtensor::use_xsimd` targets.
The above example then becomes:

```cmake
cmake_minimum_required(VERSION 3.1)
project(example)
find_package(GMatElastoPlasticQPot REQUIRED)
find_package(xtensor REQUIRED)
find_package(xsimd REQUIRED)
add_executable(example example.cpp)
target_link_libraries(example PRIVATE
    GMatElastoPlasticQPot
    xtensor::optimize
    xtensor::use_xsimd)
```

See the [documentation of xtensor](https://xtensor.readthedocs.io/en/latest/) concerning optimisation.

## By hand

Presuming that the compiler is `c++`, compile using:

```
c++ -I/path/to/GMatElastoPlasticQPot/include ...
```

Note that you have to take care of the *xtensor* dependency, the C++ version, optimisation,
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

# Upgrading instructions

## Upgrading to >v0.17.*

The individual material point and the array of material points was fully integrated.
In addition, the number of copies was reduced.

**Important** the sequence of yield strains is now taken as is.
It will not prepended with `- epsy[:, 0]` as was default with `init_elastic`.

### C++

There is only a single class `Elastic`. It's functions where renamed:

*   `.setStrain(...)` -> `.set_Eps(...)`
*   `.Stress()` -> `.Sig()` (now returns a reference).
*   `.stress(...)`: deprecated.
*   `.Tangent()` -> `.C()` (now returns a reference).
*   `.tangent(...)`: deprecated.

### Python

There is only a single class `Elastic`. It's functions are converted to properties:

*   `.setStrain(...)` -> `.Eps = ...`
*   `.Stress()` -> `.Sig` (now returns a reference).
*   `.stress(...)`: deprecated.
*   `.Tangent()` -> `.C` (now returns a reference).
*   `.tangent(...)`: deprecated.

## Upgrading to v0.10.*

`Array<rank>.check` should be replaced by something like:
```cpp
if (xt::any(xt::equal(array.type(), Type::Unset))) {
    throw std::runtime_error("Please set all points");
}
```
Note however that it is no longer required to set all points,
unset points are filled-up with zeros.

## Upgrading to >v0.8.*

`xtensor_fixed` was completely deprecated in v0.8.0, as were the type aliases
`Tensor2` and `Tensor4`.
Please update your code as follows:

*   `Tensor2` -> `xt::xtensor<double, 2>`.
*   `Tensor4` -> `xt::xtensor<double, 4>`.

**Tip:** Used `auto` as return type as much as possible.
This simplifies implementation, and renders is less subjective to library
return type changes.

## Upgrading to >v0.6.*

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

## v0.17.0

Complete API overhaul.

## v0.16.1

*   Adding constant reference functions to array (#95).

## v0.16.0

*   Adding "reset_epsy" (#94)
*   Style updates (#93)

## v0.15.6

*   [Python] Adding convenience import.
*   [CI] Updating Python installation.
*   [Python] Adding version (#91).

## v0.15.5

*   [Examples] Updating GooseFEM::Iterate (#89)
*   [Python] Switching to scikit-build (#89)
*   [CMake] Clean-up (#89)
*   [Tests] Renaming "test" -> "tests" (#89)

## v0.15.4

*   [CMake] Avoid setuptools_scm dependency if SETUPTOOLS_SCM_PRETEND_VERSION is defined

## v0.15.3

*   [Python] passing `CMAKE_ARGS` environment variable.

## v0.15.2

*   [Python] code-style update, removing work-around (#86)
*   [CI] Minor update gh-pages
*   Adding `checkYieldRedraw`
*   Adding missing header
*   [docs] Explaining rationale behind "sigd"

## v0.15.1

*   Exposing non-allocating overloads in Python API
*   Switching to xtensor-python
*   CMake & setup.py modernization
*   Templating all around
*   Removing deprecated functions
*   Internal change: using `flat` instead of `data`
*   [CI] Using micromamba (#80)
*   Using QPot::Chunked instead of QPot::Static (minor API change) (#78)
*   Conforming with GooseFEM syntax (#79)
*   Fixing paths docs

## v0.15.0

*   Type identification: use `bool` instead of `size_t`.
*   Switching-off xsimd for Python testing:
    if used also the Python API of QPot should be compiled with xsimd.
*   Updating references to QPot::Static / underlying model.
*   Adding references to QPot::Static / underlying model to Python API.

## v0.14.0

*   Updating to latest QPot
*   [CMake] Using setuptools_scm for versioning
*   [docs] Simplifying CMakeLists
*   [docs] Introducing Doxygen docs
*   [docs] Work-around Doxygen bug
*   [docs] Doxygen: dark style
*   [docs] Publish docs to GH Pages
*   [Python API] Reducing dependencies

## v0.13.0

*   Adding `nextYield` from latests QPot extension.

## v0.12.0

*   Added overloads `currentYieldRight` and `currentYieldLeft` that allow for a shift.
    Uses latests QPot extension.

## v0.11.0

Create pure elastic Array from elasto-plastic Array by:
```cpp
GM::Array<2> elas(mat.shape());
elas.setElastic(mat.K(), mat.G());
```

## v0.10.0

*   `Array` now sets zeros for all `Type::Unset` points.
    The function `check` is deprecated accordingly.
*   The methods `setStrainIterator`, `strainIterator`, and `stressIterator` are replaced
    by `setStrainPtr`, `strainPtr`, and `stressPtr`, while `tangentPtr` is added.
    These methods now require a pointer input.
*   `Array` now sets zeros for all `Type::Unset` points.
    The function `check` is deprecated accordingly.
*   Updated to latest GMatTensor.

## v0.9.0

*   Returning underlying models & stored strain tensors (#62)
*   Simplifying tests (#61)

## v0.8.0

*   Using *GMatTensor* under the hood.
    This significantly shortens the implementation here, without loosing any functionality
    (while allowing exposing future additions to *GMatTensor*).
*   Switching to GitHub CI.
*   Shortening Python API generator.
*   Using Python's `unittest`.
*   Stopping completely the use of `xtensor_fixed`.

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

## v0.2.1

*   Added minimal documentation.
