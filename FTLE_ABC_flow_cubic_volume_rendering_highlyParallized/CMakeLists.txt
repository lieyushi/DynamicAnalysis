# CMake entry point
cmake_minimum_required (VERSION 2.6)
project (abcFlow)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

include_directories(
    /usr/include/
    /usr/include/eigen3
    /usr/local/include/
    .
)

set(ALL_LIBS
)

add_definitions(
#	-DTW_STATIC
#	-DTW_NO_LIB_PRAGMA
#	-DTW_NO_DIRECT3D
#	-DGLEW_STATIC
#	-D_CRT_SECURE_NO_WARNINGS
)

set(SRC
    src/Block.h
    src/Block.cpp
	src/ABC_flow.h
    src/ABC_flow.cpp
    src/FTLE_Calculator.h
    src/FTLE_Calculator.cpp
    src/main.cpp
)

add_executable(abcFlow ${SRC})
target_link_libraries(abcFlow ${ALL_LIBS})

