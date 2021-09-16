
cmake_minimum_required(VERSION 3.0)

project(GMatElastoPlasticQPot-examples)

option(WARNINGS "Show build warnings" ON)
option(ASSERT "Enable assertions" OFF)

set(CMAKE_BUILD_TYPE Release)

file(GLOB APP_SOURCES *.cpp)

foreach(testsourcefile ${APP_SOURCES})

    string(REPLACE ".cpp" "" testname ${testsourcefile})
    get_filename_component(testname ${testname} NAME)

    add_executable(${testname} ${testsourcefile})

    find_package(GMatElastoPlasticQPot REQUIRED)
    find_package(GooseFEM REQUIRED)
    find_package(xtensor REQUIRED)

    target_link_libraries(${testname} PUBLIC
        GMatElastoPlasticQPot
        xtensor::optimize
        xtensor::use_xsimd
        GooseFEM)

    if(WARNINGS)
        target_link_libraries(${testname} PRIVATE GMatElastoPlasticQPot::compiler_warnings)
    endif()

    if(ASSERT)
        target_link_libraries(${testname} PRIVATE GooseFEM::debug GMatElastoPlasticQPot::debug)
    endif()

endforeach()
