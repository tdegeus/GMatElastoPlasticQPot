/**
\file
\copyright Copyright. Tom de Geus. All rights reserved.
\license This project is released under the MIT License.
*/

#ifndef GMATELASTOPLASTICQPOT_CONFIG_H
#define GMATELASTOPLASTICQPOT_CONFIG_H

/**
All assertions are implementation as:

    GMATELASTOPLASTICQPOT_ASSERT(...)

They can be enabled by:

    #define GMATELASTOPLASTICQPOT_ENABLE_ASSERT

(before including GMatElastoPlasticQPot).
The advantage is that:

-   File and line-number are displayed if the assertion fails.
-   Assertions can be enabled/disabled independently from those of other libraries.

\throw std::runtime_error
*/
#ifdef GMATELASTOPLASTICQPOT_ENABLE_ASSERT
#define GMATELASTOPLASTICQPOT_ASSERT(expr) GMATTENSOR_ASSERT_IMPL(expr, __FILE__, __LINE__)
#else
#define GMATELASTOPLASTICQPOT_ASSERT(expr)
#endif

/**
Material model based on a sequence of parabolic potentials.
*/
namespace GMatElastoPlasticQPot {

/**
Define container type.
The default `xt::xtensor` can be changed using:

-   `#define GMATELASTOPLASTICQPOT_USE_XTENSOR_PYTHON` -> `xt::pytensor`
*/
namespace array_type {

#ifdef GMATELASTOPLASTICQPOT_USE_XTENSOR_PYTHON

/**
Fixed (static) rank array.
*/
template <typename T, size_t N>
using tensor = xt::pytensor<T, N>;

#else

/**
Fixed (static) rank array.
*/
template <typename T, size_t N>
using tensor = xt::xtensor<T, N>;

#endif

} // namespace array_type

} // namespace GMatElastoPlasticQPot

#endif
