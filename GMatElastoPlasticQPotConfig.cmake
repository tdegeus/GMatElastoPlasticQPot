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

if (${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} VERSION_GREATER_EQUAL 3.11)
    if(NOT TARGET GMatElastoPlasticQPot::compiler_warnings)
        add_library(GMatElastoPlasticQPot::compiler_warnings INTERFACE IMPORTED)
        if(MSVC)
            target_compile_options(GMatElastoPlasticQPot::compiler_warnings INTERFACE
                /W4)
        else()
            target_compile_options(GMatElastoPlasticQPot::compiler_warnings INTERFACE
                -Wall
                -Wextra
                -pedantic
                -Wno-unknown-pragmas)
        endif()
    endif()
endif()

# Define support target "GMatElastoPlasticQPot::assert"

if (${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} VERSION_GREATER_EQUAL 3.11)
    if(NOT TARGET GMatElastoPlasticQPot::assert)
        add_library(GMatElastoPlasticQPot::assert INTERFACE IMPORTED)
        target_compile_definitions(GMatElastoPlasticQPot::assert INTERFACE GMATELASTOPLASTICQPOT_ENABLE_ASSERT)
    endif()
endif()
