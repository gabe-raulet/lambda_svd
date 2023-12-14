cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
project(lambda_svd LANGUAGES CXX)

add_compile_options(-fsanitize=address -O0 -g -fno-omit-frame-pointer)
add_link_options(-fsanitize=address)

find_package(MPI REQUIRED)

set(LAPACK_INCLUDE_DIRS "/opt/homebrew/Cellar/openblas/0.3.24/include")
set(LAPACK_LIBRARIES "-L/opt/homebrew/Cellar/lapack/3.12.0/lib -L/opt/homebrew/Cellar/openblas/0.3.24/lib -llapacke -lopenblas")

add_library(lambda_svd
            src/kiss.cpp
            src/mmiodense.cpp
            src/svdalgs.cpp
            src/serial_tree.cpp
            src/dist_mpi_tree.cpp
            src/dist_fmi_tree.cpp
            src/mpitimer.cpp
            src/fmi_wrapper.cpp)

include_directories(./inc)

link_libraries(${LAPACK_LIBRARIES})

target_compile_options(lambda_svd PUBLIC ${MPI_CXX_COMPILE_FLAGS})
target_link_libraries(lambda_svd PUBLIC ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINKFLAGS})
target_include_directories(lambda_svd PUBLIC ${MPI_CXX_INCLUDE_PATH})

add_subdirectory(./fmi)
target_link_libraries(lambda_svd PUBLIC FMI)
target_include_directories(lambda_svd PUBLIC ${FMI_INCLUDE_DIRS} ${LAPACK_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR})

add_subdirectory(./svd_serial)
add_subdirectory(./svd_mpi)
add_subdirectory(./svd_fmi)
