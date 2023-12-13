cmake_minimum_required(VERSION 3.10)

project(FMI)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(Boost_USE_STATIC_LIBS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

get_directory_property(hasParent PARENT_DIRECTORY)

if (hasParent)
    set(FMI_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include/" PARENT_SCOPE)
endif()

find_package(Boost COMPONENTS log REQUIRED)

add_subdirectory(extern/TCPunch/client/)

add_library(FMI STATIC src/Communicator.cpp src/utils/Configuration.cpp src/comm/Channel.cpp src/comm/ClientServer.cpp
        src/utils/ChannelPolicy.cpp src/comm/PeerToPeer.cpp src/comm/Direct.cpp)

target_include_directories(FMI PRIVATE ${Boost_INCLUDE_DIRS} ${TCPUNCH_INCLUDE_DIRS})

target_link_libraries(FMI ${Boost_Libraries} Boost::log tcpunch)
