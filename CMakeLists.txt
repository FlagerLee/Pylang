cmake_minimum_required(VERSION 3.11)

#-------------------------------------------------------------------------------
# Project setup and globals
#-------------------------------------------------------------------------------

project(pylang VERSION 0.0.2.1)

if (POLICY CMP0077)
    cmake_policy(SET CMP0077 NEW)
endif ()

if (POLICY CMP0116)
    cmake_policy(SET CMP0116 OLD)
endif ()

if (POLICY CMP0148)
    cmake_policy(SET CMP0148 NEW)
endif()

set(CMAKE_CXX_STANDARD 17)

#-------------------------------------------------------------------------------
# LLVM and MLIR setup
#-------------------------------------------------------------------------------

set(MLIR_DIR ${PROJECT_SOURCE_DIR}/third-party/llvm-project/build/lib/cmake/mlir)

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_MLIR_BINARY_DIR ${MLIR_DIR}/../../../bin)
set(LLVM_MLIR_SOURCE_DIR ${MLIR_DIR}/../../../../mlir)
set(LLVM_PROJECT_SOURCE_DIR ${MLIR_DIR}/../../../../)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

set(LLVM_LINK_COMPONENTS Support)

set(PYTHON_MLIR_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(PYTHON_MLIR_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

#-------------------------------------------------------------------------------
# fmt library setup
#-------------------------------------------------------------------------------

add_subdirectory(third-party/fmt)
include_directories(third-party/fmt/include)

#-------------------------------------------------------------------------------
# gtest setup
#-------------------------------------------------------------------------------

add_subdirectory(third-party/googletest)
include_directories(${gtest_SOURCE_DIR}/include)

#-------------------------------------------------------------------------------
# PyBind setup
#-------------------------------------------------------------------------------

add_subdirectory(third-party/pybind11)

#-------------------------------------------------------------------------------
# Python setup
#-------------------------------------------------------------------------------

include(MLIRDetectPythonEnv)
find_package (Python3 3.10 EXACT COMPONENTS Interpreter)

set(PYLANG_PYTHON_PACKAGES_DIR "${CMAKE_CURRENT_BINARY_DIR}/python_packages")
set(MLIR_ENABLE_BINDINGS_PYTHON=ON)

#-------------------------------------------------------------------------------
# Config detect
#-------------------------------------------------------------------------------

INCLUDE (CheckCXXSourceCompiles)
INCLUDE (CheckCXXSourceRuns)
# pointer bit width
cmake_host_system_information(RESULT temp QUERY IS_64BIT)
if(temp)
    set(PTR_BIT_WIDTH 8)
else()
    set(PTR_BIT_WIDTH 4)
endif()

# have le64toh function
CHECK_CXX_SOURCE_COMPILES ( "
#include <endian.h>
int main() { le64toh(1); }
" HAVE_LE64TOH_CHECK
        FAIL_REGEX "")
if(HAVE_LE64TOH_CHECK)
    set(HAVE_LE64TOH ON)
endif()

#-------------------------------------------------------------------------------
# Project setup
#-------------------------------------------------------------------------------

set(PYLANG_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(PYLANG_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/lib)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PYLANG_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PYLANG_LIBRARY_DIR})

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)

# add tests
enable_testing()
add_subdirectory(test)
