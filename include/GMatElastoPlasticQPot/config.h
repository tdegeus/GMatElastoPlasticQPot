/**
Macros used in the library.

\file
\copyright Copyright 2018. Tom de Geus. All rights reserved.
\license This project is released under the MIT License.
*/

#ifndef GMATELASTOPLASTICQPOT_CONFIG_H
#define GMATELASTOPLASTICQPOT_CONFIG_H

/**
\cond
*/
#define Q(x) #x
#define QUOTE(x) Q(x)

#define GMATELASTOPLASTICQPOT_WARNING_IMPL(message, file, line) \
    std::cout << \
        std::string(file) + ':' + std::to_string(line) + \
        ": " message ") \n\t"; \

#define GMATELASTOPLASTICQPOT_ASSERT_IMPL(expr, file, line) \
    if (!(expr)) { \
        throw std::runtime_error( \
            std::string(file) + ':' + std::to_string(line) + \
            ": assertion failed (" #expr ") \n\t"); \
    }
/**
\endcond
*/

/**
All assertions are implementation as::

    GMATELASTOPLASTICQPOT_ASSERT(...)

They can be enabled by::

    #define GMATELASTOPLASTICQPOT_ENABLE_ASSERT

(before including GMatElastoPlasticQPot).
The advantage is that:

-   File and line-number are displayed if the assertion fails.
-   GMatElastoPlasticQPot's assertions can be enabled/disabled
    independently from those of other libraries.

\throw std::runtime_error
*/
#ifdef GMATELASTOPLASTICQPOT_ENABLE_ASSERT
#define GMATELASTOPLASTICQPOT_ASSERT(expr) GMATELASTOPLASTICQPOT_ASSERT_IMPL(expr, __FILE__, __LINE__)
#else
#define GMATELASTOPLASTICQPOT_ASSERT(expr)
#endif

/**
All warnings are implemented as::

    GMATELASTOPLASTICQPOT_WARNING(...)

They can be disabled by::

    #define GMATELASTOPLASTICQPOT_DISABLE_WARNING
*/
#ifdef GMATELASTOPLASTICQPOT_DISABLE_WARNING
#define GMATELASTOPLASTICQPOT_WARNING(message)
#else
#define GMATELASTOPLASTICQPOT_WARNING(message) GMATELASTOPLASTICQPOT_WARNING_IMPL(message, __FILE__, __LINE__)
#endif

/**
All warnings specific to the Python API are implemented as::

    GMATELASTOPLASTICQPOT_WARNING_PYTHON(...)

They can be enabled by::

    #define GMATELASTOPLASTICQPOT_ENABLE_WARNING_PYTHON
*/
#ifdef GMATELASTOPLASTICQPOT_ENABLE_WARNING_PYTHON
#define GMATELASTOPLASTICQPOT_WARNING_PYTHON(message) GMATELASTOPLASTICQPOT_WARNING_IMPL(message, __FILE__, __LINE__)
#else
#define GMATELASTOPLASTICQPOT_WARNING_PYTHON(message)
#endif

/**
When using `init_elestic` in Cartesian2d::Cusp and Cartesian2d::Smooth,
from v0.18.0 onwards the yield strain index of the added negative yield strain is set to `-1`.
The 'classic' behaviour can be recoverd by:

    #define GMATELASTOPLASTICQPOT_INDEX_ELASTICOFFSET -1

before

    #include <GMatElastoPlasticQPot/Cartesian2d.h>
*/
#ifndef GMATELASTOPLASTICQPOT_INDEX_ELASTICOFFSET
#define GMATELASTOPLASTICQPOT_INDEX_ELASTICOFFSET -1
#endif

/**
Material model based on a sequence of parabolic potentials.
*/
namespace GMatElastoPlasticQPot { }

#endif
