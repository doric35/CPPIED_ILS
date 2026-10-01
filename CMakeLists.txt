cmake_minimum_required(VERSION 3.27)
project(CppiedEJOR)

#-----Set options-----
option(CXX "enable C++ compilation" ON)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#-----Packages-----
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(Eigen3 REQUIRED)
find_package(GTest REQUIRED)
find_package(GUROBI REQUIRED)
find_package(OpenCV REQUIRED)

#-----Source files-----
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "src/*.cpp")
file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS "tests/*.cpp")

#-----Optimization flags-----
set(OPT_FLAGS -O3 -march=native)
#set(OPT_FLAGS -g -O0 -Wall -Wextra -Werror -fsanitize=address -fno-omit-frame-pointer -g)
#set(OPT_FLAGS -fsanitize=address -fno-omit-frame-pointer -g)

#-----External libraries-----
add_library(cppied_lib ${SOURCES})
target_link_libraries(
        cppied_lib
        PUBLIC
        Eigen3::Eigen
        ${OpenCV_LIBS}
        )
include_directories(${GUROBI_INCLUDE_DIRS})
target_link_libraries(cppied_lib
        PUBLIC
        optimized ${GUROBI_CXX_LIBRARY}
        debug ${GUROBI_CXX_DEBUG_LIBRARY})
target_link_libraries(cppied_lib
        PUBLIC
        ${GUROBI_LIBRARY})
target_compile_options(cppied_lib PRIVATE ${OPT_FLAGS})

#-----Build executable-----
add_executable(CppiedEJOR main.cpp)
target_link_libraries(CppiedEJOR PRIVATE cppied_lib)
target_compile_options(CppiedEJOR PRIVATE ${OPT_FLAGS})
target_link_options(CppiedEJOR PRIVATE ${OPT_FLAGS})

#####################
# GTests
#####################
enable_testing()

add_executable(CppiedTests ${TEST_SOURCES})
target_link_libraries(
        CppiedTests
        PRIVATE
        cppied_lib
        GTest::gtest
        GTest::gtest_main)
target_compile_options(CppiedTests PRIVATE ${OPT_FLAGS})
target_link_options(CppiedTests PRIVATE ${OPT_FLAGS})

include(GoogleTest)
gtest_discover_tests(CppiedTests)

#-----Generate test files-----

# Generate test config files into the build tree, substituting the project root
configure_file(tests/configurations/test_config.txt.in
               ${CMAKE_SOURCE_DIR}/tests/configurations/test_config.txt @ONLY)
configure_file(tests/configurations/vnd_test_config.in
               ${CMAKE_SOURCE_DIR}/tests/configurations/vnd_test_config.txt @ONLY)
configure_file(tests/configurations/ils_test_config.in
               ${CMAKE_SOURCE_DIR}/tests/configurations/ils_test_config.txt @ONLY)
configure_file(tests/configurations/ils_local_search_config.in
               ${CMAKE_SOURCE_DIR}/tests/configurations/ils_local_search_config.txt @ONLY)
configure_file(tests/configurations/ilp_test_config.in
        ${CMAKE_SOURCE_DIR}/tests/configurations/ilp_test_config.txt @ONLY)
configure_file(tests/configurations/ilp_large_test_config.in
        ${CMAKE_SOURCE_DIR}/tests/configurations/ilp_large_test_config.txt @ONLY)

file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/tests/scratch)

target_compile_definitions(CppiedTests PRIVATE
        PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
        PROJECT_BINARY_DIR="${CMAKE_BINARY_DIR}")
