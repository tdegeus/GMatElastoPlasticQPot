
cmake_minimum_required(VERSION 3.0)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    project(GMatElastoPlasticQPot-examples)
    find_package(GMatElastoPlasticQPot REQUIRED CONFIG)
endif()

set(CMAKE_BUILD_TYPE Release)

set(ASSERT OFF)
set(DEBUG OFF)
option(SIMD "Enable xsimd" OFF)
option(WARNINGS "Enable warnings" ON)

file(GLOB APP_SOURCES *.cpp)

foreach(testsourcefile ${APP_SOURCES})

    string(REPLACE ".cpp" "" test_name ${testsourcefile})
    get_filename_component(test_name ${test_name} NAME)

    add_executable(${test_name} ${testsourcefile})

    find_package(GooseFEM REQUIRED)
    find_package(xtensor REQUIRED)

    target_link_libraries(${test_name} PUBLIC GMatElastoPlasticQPot GooseFEM)

    if(SIMD)
        find_package(xsimd REQUIRED)
        target_link_libraries(${test_name} PRIVATE xtensor::optimize xtensor::use_xsimd)
    endif()

    if(WARNINGS)
        target_link_libraries(${test_name} PRIVATE GMatElastoPlasticQPot::compiler_warnings)
    endif()

    if(ASSERT)
        target_link_libraries(${test_name} PRIVATE GooseFEM::assert GMatElastoPlasticQPot::assert)
    endif()

    if(DEBUG)
        target_link_libraries(${test_name} PRIVATE GooseFEM::debug GMatElastoPlasticQPot::debug)
    endif()

endforeach()

