
cmake_minimum_required(VERSION 3.0)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    project(GMatElastoPlasticQPot-test)
    find_package(GMatElastoPlasticQPot REQUIRED CONFIG)
    set(GMATELASTOPLASTICQPOT_INCLUDE_DIR ${GMatElastoPlasticQPot_INCLUDE_DIRS})
endif()

option(WARNINGS "Show build warnings" ON)

set(CMAKE_BUILD_TYPE Release)
if(MSVC)
  set(CMAKE_CXX_STANDARD 17)
else()
  set(CMAKE_CXX_STANDARD 14)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DXTENSOR_USE_XSIMD=ON")

if(NOT MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

find_package(Catch2 REQUIRED)
find_package(xtensor REQUIRED)
find_package(xtl REQUIRED)
find_package(xsimd REQUIRED)

add_executable(main main.cpp Cartesian2d.cpp)
target_link_libraries(main PUBLIC Catch2::Catch2 xtensor xtl xsimd)
target_include_directories(main PRIVATE "${GMATELASTOPLASTICQPOT_INCLUDE_DIR}")
if(WARNINGS)
    if(MSVC)
        target_compile_options(main PRIVATE /W4)
    else()
        target_compile_options(main PRIVATE -Wall -Wextra -pedantic -Wno-unknown-pragmas)
    endif()
endif()
