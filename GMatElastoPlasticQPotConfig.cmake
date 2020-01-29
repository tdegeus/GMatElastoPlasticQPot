# GMatElastoPlasticQPot cmake module
#
# This module sets the target:
#
#   GMatElastoPlasticQPot
#
# In addition, it sets the following variables:
#
#   GMatElastoPlasticQPot_FOUND - true if GMatElastoPlasticQPot found
#   GMatElastoPlasticQPot_VERSION - GMatElastoPlasticQPot's version
#   GMatElastoPlasticQPot_INCLUDE_DIRS - the directory containing GMatElastoPlasticQPot headers
#
# The following support targets are defined to simplify things:
#
#   GMatElastoPlasticQPot::compiler_warnings - enable compiler warnings
#   GMatElastoPlasticQPot::assert - enable GMatElastoPlasticQPot assertions
#   GMatElastoPlasticQPot::debug - enable all assertions (slow)

include(CMakeFindDependencyMacro)

# Define target "GMatElastoPlasticQPot"

if(NOT TARGET GMatElastoPlasticQPot)
    include("${CMAKE_CURRENT_LIST_DIR}/GMatElastoPlasticQPotTargets.cmake")
    get_target_property(GMatElastoPlasticQPot_INCLUDE_DIRS GMatElastoPlasticQPot INTERFACE_INCLUDE_DIRECTORIES)
endif()

# Find dependencies

find_dependency(xtensor)

# Define support target "GMatElastoPlasticQPot::compiler_warnings"

if(NOT TARGET GMatElastoPlasticQPot::compiler_warnings)
    add_library(GMatElastoPlasticQPot::compiler_warnings INTERFACE IMPORTED)
    if(MSVC)
        set_property(
            TARGET GMatElastoPlasticQPot::compiler_warnings
            PROPERTY INTERFACE_COMPILE_OPTIONS
            /W4)
    else()
        set_property(
            TARGET GMatElastoPlasticQPot::compiler_warnings
            PROPERTY INTERFACE_COMPILE_OPTIONS
            -Wall -Wextra -pedantic -Wno-unknown-pragmas)
    endif()
endif()

# Define support target "GMatElastoPlasticQPot::assert"

if(NOT TARGET GMatElastoPlasticQPot::assert)
    add_library(GMatElastoPlasticQPot::assert INTERFACE IMPORTED)
    set_property(
        TARGET GMatElastoPlasticQPot::assert
        PROPERTY INTERFACE_COMPILE_DEFINITIONS
        GMATELASTOPLASTICQPOT_ENABLE_ASSERT)
endif()
