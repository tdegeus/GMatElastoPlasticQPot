# GMatElastoPlasticQPot

[![Travis](https://travis-ci.com/tdegeus/GMatElastoPlasticQPot.svg?branch=master)](https://travis-ci.com/tdegeus/GMatElastoPlasticQPot)

Elasto-plastic material model based on a manifold of quadratic potentials. An overview of the theory can be found in `docs/` in particular in this [PDF](docs/readme.pdf).

# Contents

<!-- MarkdownTOC -->

- [Implementation](#implementation)
- [Installation](#installation)
    - [Linux / macOS](#linux--macos)
        - [Install system-wide \(depends on your privileges\)](#install-system-wide-depends-on-your-privileges)
        - [Install in custom location \(user\)](#install-in-custom-location-user)
- [References / Credits](#references--credits)

<!-- /MarkdownTOC -->

# Implementation

The headers are meant to be self-explanatory, please check them out:

* [Cartesian2d.h](include/GMatElastoPlasticQPot/Cartesian2d.h)

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

# Installation

## Linux / macOS

### Install system-wide (depends on your privileges)

1.  Proceed to a (temporary) build directory. For example:

    ```bash
    cd /path/to/GMatElastoPlasticQPot
    mkdir build
    cd build
    ```

2.  'Install' `GMatElastoPlasticQPot`. For the path in **1.**:

    ```bash
    cmake .. 
    make install
    ```

> One usually does not need any compiler arguments after following this protocol.

### Install in custom location (user)

1.  Proceed to a (temporary) build directory. For example:

    ```bash
    cd /path/to/GMatElastoPlasticQPot
    mkdir build
    cd build
    ```

2.  'Install' `GMatElastoPlasticQPot`, to install it in a custom location. For the path in **1.**:

    ```bash
    mkdir /custom/install/path
    cmake .. -DCMAKE_INSTALL_PREFIX:PATH=/custom/install/path
    make install
    ```

3.  Add the appropriate paths to for example your ``~/.bashrc`` (or ``~/.zshrc``). For the path in **2.**: 

    ```bash
    export PKG_CONFIG_PATH=/custom/install/path/share/pkgconfig:$PKG_CONFIG_PATH
    export CPLUS_INCLUDE_PATH=$HOME/custom/install/path/include:$CPLUS_INCLUDE_PATH
    ```

> One usually has to inform the CMake or the compiler about `${CPLUS_INCLUDE_PATH}`.

# References / Credits

*   The model is described in *T.W.J. de Geus, M. Popović, W. Ji, A. Rosso, M. Wyart (2019). How collective asperity detachments nucleate slip at frictional interfaces. [arXiv: 1904.07635](http://arxiv.org/abs/1904.07635)*.

*   [xtensor](https://github.com/QuantStack/xtensor) is used under the hood.
