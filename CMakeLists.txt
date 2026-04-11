cmake_minimum_required(VERSION 3.27) 
project(Cppied) 

option(CXX "enable C++ compilation" ON) 
if(CXX) 
    enable_language(CXX) 
endif() 
set(CMAKE_CXX_STANDARD 23) 
set(CMAKE_CXX_STANDARD_REQUIRED ON) 

#Packages 
set(GUROBI_HOME $ENV{GUROBI_HOME}) 
list(APPEND CMAKE_MODULE_PATH cmake "${GUROBI_HOME}/cmake") 
include_directories(${GUROBI_HOME}/include) 
find_library(GUROBI_LIB gurobi130 PATHS ${GUROBI_HOME}/lib) 
find_library(GUROBI_CXX_LIB gurobi_c++ PATHS ${GUROBI_HOME}/lib) 

#Eigen 3 & OpenCv 
find_package(Eigen3 REQUIRED) 
find_package(OpenCV REQUIRED) 

#Sources 
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "src/*.cpp") 
file(GLOB_RECURSE TESTS_SOURCES CONFIGURE_DEPENDS "tests/*.cpp") 
add_library(cppied_lib ${SOURCES}) 
target_link_libraries(cppied_lib PUBLIC Eigen3::Eigen ${GUROBI_CXX_LIB} ${GUROBI_LIB} ${OpenCV_LIBS} ) 

#Main executable - O3 optimization 
add_executable(Cppied main.cpp) 
target_link_libraries(Cppied PRIVATE cppied_lib) 
target_compile_options(Cppied PRIVATE -O3) 

#Google Tests 
include(FetchContent) 
FetchContent_Declare( googletest URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip ) 
FetchContent_MakeAvailable(googletest) 
#enable_testing() 

add_executable(test_cppied ${TESTS_SOURCES}) 
target_link_libraries(test_cppied PRIVATE cppied_lib GTest::gtest GTest::gtest_main ) 
# Test compile options: no optimization, debug info, sanitizers 
target_compile_options(test_cppied PRIVATE -O0 -g -fsanitize=address,undefined -fno-omit-frame-pointer -g) 
target_link_options(test_cppied PRIVATE -fsanitize=address,undefined) 
include(GoogleTest) 
#gtest_discover_tests(test_cppied)
