cmake_minimum_required(VERSION 3.30)
project(Cppied)

option(CXX "enable C++ compilation" ON)
if(CXX)
    enable_language(CXX)
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#Compilers
message(STATUS "C compiler: ${CMAKE_C_COMPILER}")
message(STATUS "CXX compiler: ${CMAKE_CXX_COMPILER}")

#Packages
message("CMAKE SOURCE : ${CMAKE_SOURCE_DIR}")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")
message("CMAKE MODULE PATH : ${CMAKE_MODULE_PATH}")
find_package(Eigen3 REQUIRED)
find_package(GUROBI REQUIRED)
find_package(OpenCV REQUIRED)

#Includes messages
message(STATUS "Include dirs: ${GUROBI_INCLUDE_DIRS}")

#Sources
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "src/*.cpp")
file(GLOB_RECURSE TESTS_SOURCES CONFIGURE_DEPENDS "tests/*.cpp")

add_library(cppied_lib ${SOURCES})
target_include_directories(cppied_lib
        PUBLIC
        ${GUROBI_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS})
target_link_libraries(cppied_lib
        PUBLIC
        Eigen3::Eigen
        ${GUROBI_CXX_LIBRARY}
        ${GUROBI_LIBRARY}
        ${OpenCV_LIBS}
)

#Main executable - O3 optimization
add_executable(Cppied main.cpp)
target_link_libraries(Cppied PRIVATE cppied_lib)
target_compile_options(Cppied PRIVATE -O0 -g)

#Google Tests
include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)
enable_testing()

add_executable(test_cppied ${TESTS_SOURCES})
target_link_libraries(test_cppied
        PRIVATE
        cppied_lib
        GTest::gtest
        GTest::gtest_main
)
# Test compile options: no optimization, debug info, sanitizers
target_compile_options(test_cppied PRIVATE -O0 -g -fsanitize=address -fsanitize=undefined)
target_link_options(test_cppied PRIVATE -fsanitize=address -fsanitize=undefined)

include(GoogleTest)
gtest_discover_tests(test_cppied)