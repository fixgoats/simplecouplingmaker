cmake_minimum_required(VERSION 3.24)

project(stuff)

set(CMAKE_C_STANDARD 99)


set(PROJECT_SOURCES
        main.cpp
)
set(MAIN_TARGET hamilton)


# Adding Raylib
include(FetchContent)
set(FETCHCONTENT_QUIET FALSE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE) # don't build the supplied examples
set(BUILD_GAMES    OFF CACHE BOOL "" FORCE) # don't build the supplied example games

FetchContent_Declare(
    raylib
    GIT_REPOSITORY "https://github.com/raysan5/raylib.git"
    GIT_TAG "5.5"
    GIT_PROGRESS TRUE
)

FetchContent_MakeAvailable(raylib)

add_executable(${MAIN_TARGET})
target_sources(${MAIN_TARGET} PRIVATE ${PROJECT_SOURCES})
add_compile_options(
       -Wall -Wextra -Wpedantic -Wno-error=maybe-uninitialized
       $<$<CONFIG:RELEASE>:-Ofast -march=native>
       $<$<CONFIG:DEBUG>:-O0>
       $<$<CONFIG:DEBUG>:-ggdb3 -fsanitize=address>
)
add_compile_definitions(
        $<$<CONFIG:RELEASE>:NDEBUG>
        $<$<CONFIG:RELEASE>:BOOST_DISABLE_ASSERTS>
)

set_property(TARGET ${MAIN_TARGET} PROPERTY CXX_STANDARD 23)
include(CheckIPOSupported)
check_ipo_supported(RESULT ipoSupported OUTPUT output)
target_compile_options(${MAIN_TARGET} PRIVATE -Wall )

target_include_directories(${MAIN_TARGET} PRIVATE deps/eigen-5.0.1 deps/raylib-5.5/include)
target_link_libraries(${MAIN_TARGET} raylib)

if(ipoSupported AND CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Using LTO")
    set_property(TARGET ${MAIN_TARGET} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
elseif(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
    message(VERBOSE "Build type is not release, LTO not enabled.")
else()
    message(WARNING "LTO is not supported: ${output}")
endif()

