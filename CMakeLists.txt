
cmake_minimum_required(VERSION 3.0)

# Basic settings
# ==============

project(GMatElastoPlasticQPot)

option(BUILD_TESTS "${PROJECT_NAME} Build tests" OFF)
option(BUILD_EXAMPLES "${PROJECT_NAME} Build examples" OFF)

# Version
# =======

file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/include/GMatElastoPlasticQPot/config.h"
             _GMAT_version_defines
     REGEX "#define GMATELASTOPLASTICQPOT_VERSION_(MAJOR|MINOR|PATCH)")

foreach(ver ${_GMAT_version_defines})
    if(ver MATCHES "#define GMATELASTOPLASTICQPOT_VERSION_(MAJOR|MINOR|PATCH) +([^ ]+)$")
        set(_GMAT_VERSION_${CMAKE_MATCH_1}
            "${CMAKE_MATCH_2}"
            CACHE INTERNAL "")
    endif()
endforeach()

set(_GMAT_VERSION ${_GMAT_VERSION_MAJOR}.${_GMAT_VERSION_MINOR}.${_GMAT_VERSION_PATCH})

message(STATUS "Building GMatElastoPlasticQPot v${_GMAT_VERSION}")

# Set target
# ==========

find_package(xtensor REQUIRED)
find_package(QPot REQUIRED)

add_library(GMatElastoPlasticQPot INTERFACE)

target_include_directories(GMatElastoPlasticQPot INTERFACE
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(GMatElastoPlasticQPot INTERFACE xtensor QPot)

# Installation
# ============

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/" DESTINATION include)

install(TARGETS GMatElastoPlasticQPot EXPORT GMatElastoPlasticQPot-targets)

install(
    EXPORT GMatElastoPlasticQPot-targets
    FILE GMatElastoPlasticQPotTargets.cmake
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/GMatElastoPlasticQPot")

set(_GMAT ${CMAKE_SIZEOF_VOID_P})
unset(CMAKE_SIZEOF_VOID_P)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/GMatElastoPlasticQPotConfigVersion.cmake"
    VERSION ${_GMAT_VERSION}
    COMPATIBILITY AnyNewerVersion)

set(CMAKE_SIZEOF_VOID_P ${_GMAT})

install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/GMatElastoPlasticQPotConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/GMatElastoPlasticQPotConfigVersion.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/GMatElastoPlasticQPot")

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/GMatElastoPlasticQPot.pc.in"
               "${CMAKE_CURRENT_BINARY_DIR}/GMatElastoPlasticQPot.pc" @ONLY)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/GMatElastoPlasticQPot.pc"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig/")

# Add builds
# ==========

include("GMatElastoPlasticQPotConfig.cmake")

set(CTEST_TEST_TARGET_ALIAS RUN_TESTS_AND_EXAMPLES)
include(CTest)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

if(BUILD_EXAMPLES)
    enable_testing()
    add_subdirectory(examples)
endif()
