/* =================================================================================================

(c - MIT) T.W.J. de Geus (Tom) | www.geus.me | github.com/tdegeus/GMatElastoPlasticQPot

================================================================================================= */

#ifndef GMATELASTOPLASTICQPOT_CONFIG_H
#define GMATELASTOPLASTICQPOT_CONFIG_H

// -------------------------------------------------------------------------------------------------

// use "M_PI" from "math.h"
#define _USE_MATH_DEFINES

#include <tuple>
#include <stdexcept>
#include <limits>
#include <math.h>
#include <iostream>
#include <vector>
#include <xtensor/xarray.hpp>
#include <xtensor/xnoalias.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xoperation.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xmath.hpp>

// -------------------------------------------------------------------------------------------------

// dummy operation that can be use to suppress the "unused parameter" warnings
#define UNUSED(p) ( (void)(p) )

// -------------------------------------------------------------------------------------------------

#ifndef NDEBUG
#define GMATELASTOPLASTICQPOT_ENABLE_ASSERT
#endif

#ifdef GMATELASTOPLASTICQPOT_ENABLE_ASSERT
#define GMATELASTOPLASTICQPOT_ASSERT(expr) GMATELASTOPLASTICQPOT_ASSERT_IMPL(expr, __FILE__, __LINE__)
#define GMATELASTOPLASTICQPOT_ASSERT_IMPL(expr, file, line)                                                               \
    if (!(expr))                                                                                                          \
    {                                                                                                                     \
        throw std::runtime_error(std::string(file) + ':' + std::to_string(line) + ": assertion failed (" #expr ") \n\t"); \
    }
#else
#define GMATELASTOPLASTICQPOT_ASSERT(expr)
#endif

// -------------------------------------------------------------------------------------------------

#define GMATELASTOPLASTICQPOT_WORLD_VERSION 0
#define GMATELASTOPLASTICQPOT_MAJOR_VERSION 2
#define GMATELASTOPLASTICQPOT_MINOR_VERSION 0

#define GMATELASTOPLASTICQPOT_VERSION_AT_LEAST(x,y,z) \
  (GMATELASTOPLASTICQPOT_WORLD_VERSION>x || (GMATELASTOPLASTICQPOT_WORLD_VERSION>=x && \
  (GMATELASTOPLASTICQPOT_MAJOR_VERSION>y || (GMATELASTOPLASTICQPOT_MAJOR_VERSION>=y && \
                                             GMATELASTOPLASTICQPOT_MINOR_VERSION>=z))))

#define GMATELASTOPLASTICQPOT_VERSION(x,y,z) \
  (GMATELASTOPLASTICQPOT_WORLD_VERSION==x && \
   GMATELASTOPLASTICQPOT_MAJOR_VERSION==y && \
   GMATELASTOPLASTICQPOT_MINOR_VERSION==z)

// -------------------------------------------------------------------------------------------------

#endif
