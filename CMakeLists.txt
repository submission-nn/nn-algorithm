cmake_minimum_required(VERSION 3.10)
project(nn_code)

set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_VERBOSE_MAKEFILE ON)

include_directories("deps/b63/src")
include_directories("deps")
include_directories("src")
include_directories(test)

# two possible options. 'tcmalloc' is the standard google malloc implementation
#set(MALLOC_LIBRARY tcmalloc)
set(MALLOC_LIBRARY "")

# sadly we need '-march=native' even in 'DEBUG' mode. Because otherwise the code wouldn't compile.
# for fsized-deallocation see: https://github.com/google/tcmalloc/blob/master/docs/tuning.md
set(DEBUG_FLAGS "-g -DDEBUG -march=native ${VALUE_FLAG}")
set(OPTIMIZE_FLAGS "-march=native -O3 -flto -DNDEBUG")
set(SANITIZER_FLAGS -fsanitize=address -fno-omit-frame-pointer)
set(GCC_HELPER_FLAGS -fvar-tracking-assignments)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # if using GCC
    set(DEBUG_FLAGS "${DEBUG_FLAGS} ${GCC_HELPER_FLAGS} -O0 ")
endif()

# Set additionally our own debug/optimisation compiler/linkage flags. Maybe not he nicest way, but by far the easiest.
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${DEBUG_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${DEBUG_FLAGS}")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${OPTIMIZE_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OPTIMIZE_FLAGS}")

add_subdirectory(test)
add_subdirectory(bench)
