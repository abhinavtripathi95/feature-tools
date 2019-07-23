project (sfop)
cmake_minimum_required (VERSION 3.0)
set (CMAKE_CXX_STANDARD 11)

if(NOT CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra -fPIC")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(X11 REQUIRED)
link_directories(${X11_LIBRARIES})
include_directories(${X11_INCLUDE_DIR}
			include)

add_subdirectory(src)
