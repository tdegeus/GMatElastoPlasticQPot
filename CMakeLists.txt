
cmake_minimum_required(VERSION 3.0)

# Basic settings
# ==============

project(GMatElastoPlasticQPot)

option(BUILD_TESTS "${PROJECT_NAME} Build tests" OFF)

# Version
# =======

file(
    STRINGS
        "${CMAKE_CURRENT_SOURCE_DIR}/include/GMatElastoPlasticQPot/config.h"
        GMatElastoPlasticQPot_version_defines
    REGEX
        "#define GMATELASTOPLASTICQPOT_VERSION_(MAJOR|MINOR|PATCH)")

foreach(ver ${GMatElastoPlasticQPot_version_defines})
    if(ver MATCHES "#define GMATELASTOPLASTICQPOT_VERSION_(MAJOR|MINOR|PATCH) +([^ ]+)$")
        set(GMATELASTOPLASTICQPOT_VERSION_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}" CACHE INTERNAL "")
    endif()
endforeach()

set(GMATELASTOPLASTICQPOT_VERSION
    ${GMATELASTOPLASTICQPOT_VERSION_MAJOR}.${GMATELASTOPLASTICQPOT_VERSION_MINOR}.${GMATELASTOPLASTICQPOT_VERSION_PATCH})

message(STATUS "Building GMatElastoPlasticQPot v${GMATELASTOPLASTICQPOT_VERSION}")

# Set target
# ==========

find_package(xtensor REQUIRED)

add_library(GMatElastoPlasticQPot INTERFACE)

target_include_directories(GMatElastoPlasticQPot INTERFACE
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(GMatElastoPlasticQPot INTERFACE xtensor)

# Installation
# ============

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

install(
    DIRECTORY
        "${CMAKE_CURRENT_SOURCE_DIR}/include/"
    DESTINATION
        include)

install(
    TARGETS
        GMatElastoPlasticQPot
    EXPORT
        GMatElastoPlasticQPot-targets)

install(
    EXPORT
        GMatElastoPlasticQPot-targets
    FILE
        GMatElastoPlasticQPotTargets.cmake
    DESTINATION
        "${CMAKE_INSTALL_LIBDIR}/cmake/GMatElastoPlasticQPot")

set(_GMATELASTOPLASTICQPOT ${CMAKE_SIZEOF_VOID_P})
unset(CMAKE_SIZEOF_VOID_P)

write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/GMatElastoPlasticQPotConfigVersion.cmake"
    VERSION
        ${GMATELASTOPLASTICQPOT_VERSION}
    COMPATIBILITY
        AnyNewerVersion)

set(CMAKE_SIZEOF_VOID_P ${_GMATELASTOPLASTICQPOT})

install(
    FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/GMatElastoPlasticQPotConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/GMatElastoPlasticQPotConfigVersion.cmake"
    DESTINATION
        "${CMAKE_INSTALL_LIBDIR}/cmake/GMatElastoPlasticQPot")

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/GMatElastoPlasticQPot.pc.in"
    "${CMAKE_CURRENT_BINARY_DIR}/GMatElastoPlasticQPot.pc"
     @ONLY)

install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/GMatElastoPlasticQPot.pc"
    DESTINATION
        "${CMAKE_INSTALL_LIBDIR}/pkgconfig/")

# Add builds
# ==========

include("GMatElastoPlasticQPotConfig.cmake")

if(BUILD_TESTS)
    add_subdirectory(test)
endif()
