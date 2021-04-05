/**
Basic configuration.

\file GMatElastoPlasticQPot/config.h
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
Material model based on a sequence of parabolic potentials.
*/
namespace GMatElastoPlasticQPot { }

#endif
