# GMatElastoPlasticQPot cmake module
#
# This module sets the target:
#
#     GMatElastoPlasticQPot
#
# In addition, it sets the following variables:
#
#     GMatElastoPlasticQPot_FOUND - true if the library is found
#     GMatElastoPlasticQPot_VERSION - the library's version
#     GMatElastoPlasticQPot_INCLUDE_DIRS - directory containing the library's headers
#
# The following support targets are defined to simplify things:
#
#     GMatElastoPlasticQPot::compiler_warnings - enable compiler warnings
#     GMatElastoPlasticQPot::assert - enable library assertions
#     GMatElastoPlasticQPot::debug - enable all assertions (slow)

include(CMakeFindDependencyMacro)

# Define target "GMatElastoPlasticQPot"

if(NOT TARGET GMatElastoPlasticQPot)
    include("${CMAKE_CURRENT_LIST_DIR}/GMatElastoPlasticQPotTargets.cmake")
endif()

# Define "GMatElastoPlasticQPot_INCLUDE_DIRS"

get_target_property(
    GMatElastoPlasticQPot_INCLUDE_DIRS
    GMatElastoPlasticQPot
    INTERFACE_INCLUDE_DIRECTORIES)

# Find dependencies

find_dependency(GMatElastic)
find_dependency(GMatTensor)
find_dependency(QPot)
find_dependency(xtensor)

# Define support target "GMatElastoPlasticQPot::compiler_warnings"

if(NOT TARGET GMatElastoPlasticQPot::compiler_warnings)
    add_library(GMatElastoPlasticQPot::compiler_warnings INTERFACE IMPORTED)
    target_link_libraries(GMatElastoPlasticQPot::compiler_warnings INTERFACE
        GMatTensor::compiler_warnings)
endif()

# Define support target "GMatElastoPlasticQPot::assert"

if(NOT TARGET GMatElastoPlasticQPot::assert)
    add_library(GMatElastoPlasticQPot::assert INTERFACE IMPORTED)
    set_property(
        TARGET GMatElastoPlasticQPot::assert
        PROPERTY INTERFACE_COMPILE_DEFINITIONS
        GMATELASTOPLASTICQPOT_ENABLE_ASSERT
        GMATELASTIC_ENABLE_ASSERT)
endif()

# Define support target "GMatElastoPlasticQPot::debug"

if(NOT TARGET GMatElastoPlasticQPot::debug)
    add_library(GMatElastoPlasticQPot::debug INTERFACE IMPORTED)
    set_property(
        TARGET GMatElastoPlasticQPot::debug
        PROPERTY INTERFACE_COMPILE_DEFINITIONS
        GMATELASTIC_ENABLE_ASSERT
        GMATELASTOPLASTICQPOT_ENABLE_ASSERT
        GMATTENSOR_ENABLE_ASSERT
        QPOT_ENABLE_ASSERT
        XTENSOR_ENABLE_ASSERT)
endif()
