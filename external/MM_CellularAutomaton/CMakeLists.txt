cmake_minimum_required(VERSION 3.27)
project(SeabedGen)

set(CMAKE_CXX_STANDARD 23)

add_executable(SeabedGen
    main.cpp
    cellular_automaton.cpp)