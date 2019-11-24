
cmake_minimum_required(VERSION 3.0)

# Options
# =======

project(GMatElastoPlasticQPot)

option(BUILD_TESTS "${PROJECT_NAME} Build tests" OFF)

# Internals
# =========

set(GMATELASTOPLASTICQPOT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Version
# =======

file(
    STRINGS "${GMATELASTOPLASTICQPOT_INCLUDE_DIR}/GMatElastoPlasticQPot/config.h" GMatElastoPlasticQPot_version_defines
    REGEX "#define GMATELASTOPLASTICQPOT_VERSION_(MAJOR|MINOR|PATCH)")

foreach(ver ${GMatElastoPlasticQPot_version_defines})
    if(ver MATCHES "#define GMATELASTOPLASTICQPOT_VERSION_(MAJOR|MINOR|PATCH) +([^ ]+)$")
        set(GMATELASTOPLASTICQPOT_VERSION_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}" CACHE INTERNAL "")
    endif()
endforeach()

set(GMATELASTOPLASTICQPOT_VERSION
    ${GMATELASTOPLASTICQPOT_VERSION_MAJOR}.${GMATELASTOPLASTICQPOT_VERSION_MINOR}.${GMATELASTOPLASTICQPOT_VERSION_PATCH})

message(STATUS "Building GMatElastoPlasticQPot v${GMATELASTOPLASTICQPOT_VERSION}")

# Build
# =====

set(GMATELASTOPLASTICQPOT_HEADERS
    ${GMATELASTOPLASTICQPOT_INCLUDE_DIR}/GMatElastoPlasticQPot/Cartesian2d.h
    ${GMATELASTOPLASTICQPOT_INCLUDE_DIR}/GMatElastoPlasticQPot/Cartesian2d.hpp
    ${GMATELASTOPLASTICQPOT_INCLUDE_DIR}/GMatElastoPlasticQPot/Cartesian2d_Cusp.hpp
    ${GMATELASTOPLASTICQPOT_INCLUDE_DIR}/GMatElastoPlasticQPot/Cartesian2d_Elastic.hpp
    ${GMATELASTOPLASTICQPOT_INCLUDE_DIR}/GMatElastoPlasticQPot/Cartesian2d_Matrix.hpp
    ${GMATELASTOPLASTICQPOT_INCLUDE_DIR}/GMatElastoPlasticQPot/Cartesian2d_Smooth.hpp
    ${GMATELASTOPLASTICQPOT_INCLUDE_DIR}/GMatElastoPlasticQPot/config.h)

add_library(${PROJECT_NAME} INTERFACE)

target_include_directories(
    ${PROJECT_NAME} INTERFACE
    $<BUILD_INTERFACE:${GMATELASTOPLASTICQPOT_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>)

# Installation
# ============

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

install(
    TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}-targets)

export(
    EXPORT ${PROJECT_NAME}-targets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake")

install(
    FILES ${GMATELASTOPLASTICQPOT_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})

set(GMATELASTOPLASTICQPOT_CMAKECONFIG_INSTALL_DIR
    "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}" CACHE
    STRING "install path for ${PROJECT_NAME}Config.cmake")

configure_package_config_file(
    "GMatElastoPlasticQPotConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    INSTALL_DESTINATION ${GMATELASTOPLASTICQPOT_CMAKECONFIG_INSTALL_DIR})

set(_GMATELASTOPLASTICQPOT_CMAKE_SIZEOF_VOID_P ${CMAKE_SIZEOF_VOID_P})

unset(CMAKE_SIZEOF_VOID_P)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    VERSION ${GMATELASTOPLASTICQPOT_VERSION}
    COMPATIBILITY AnyNewerVersion)

set(CMAKE_SIZEOF_VOID_P ${_GMATELASTOPLASTICQPOT_CMAKE_SIZEOF_VOID_P})

install(
    FILES
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    DESTINATION ${GMATELASTOPLASTICQPOT_CMAKECONFIG_INSTALL_DIR})

install(
    EXPORT ${PROJECT_NAME}-targets
    FILE "${PROJECT_NAME}Targets.cmake"
    DESTINATION ${GMATELASTOPLASTICQPOT_CMAKECONFIG_INSTALL_DIR})

configure_file(
    "GMatElastoPlasticQPot.pc.in"
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc"
     @ONLY)

install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig/")

# Add builds
# ==========

if(BUILD_TESTS)
    add_subdirectory(test)
endif()
