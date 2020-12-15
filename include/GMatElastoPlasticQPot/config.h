/*

(c - MIT) T.W.J. de Geus (Tom) | www.geus.me | github.com/tdegeus/GMatElastoPlasticQPot

*/

#ifndef GMATELASTOPLASTICQPOT_CONFIG_H
#define GMATELASTOPLASTICQPOT_CONFIG_H

#ifdef GMATELASTOPLASTICQPOT_ENABLE_ASSERT

    #define GMATELASTOPLASTICQPOT_ASSERT(expr) \
        GMATELASTOPLASTICQPOT_ASSERT_IMPL(expr, __FILE__, __LINE__)

    #define GMATELASTOPLASTICQPOT_ASSERT_IMPL(expr, file, line) \
        if (!(expr)) { \
            throw std::runtime_error( \
                std::string(file) + ':' + std::to_string(line) + \
                ": assertion failed (" #expr ") \n\t"); \
        }

#else

    #define GMATELASTOPLASTICQPOT_ASSERT(expr)

#endif

#define GMATELASTOPLASTICQPOT_VERSION_MAJOR 0
#define GMATELASTOPLASTICQPOT_VERSION_MINOR 12
#define GMATELASTOPLASTICQPOT_VERSION_PATCH 0

#define GMATELASTOPLASTICQPOT_VERSION_AT_LEAST(x,y,z) \
    (GMATELASTOPLASTICQPOT_VERSION_MAJOR > x || (GMATELASTOPLASTICQPOT_VERSION_MAJOR >= x && \
    (GMATELASTOPLASTICQPOT_VERSION_MINOR > y || (GMATELASTOPLASTICQPOT_VERSION_MINOR >= y && \
                                                 GMATELASTOPLASTICQPOT_VERSION_PATCH >= z))))

#define GMATELASTOPLASTICQPOT_VERSION(x,y,z) \
    (GMATELASTOPLASTICQPOT_VERSION_MAJOR == x && \
     GMATELASTOPLASTICQPOT_VERSION_MINOR == y && \
     GMATELASTOPLASTICQPOT_VERSION_PATCH == z)

#endif
