cmake_minimum_required(VERSION 3.10)
project(lambda_svd LANGUAGES CXX)

find_package(MPI REQUIRED)

set(LAPACK_INCLUDE_DIRS "/opt/homebrew/Cellar/openblas/0.3.24/include")
set(LAPACK_LIBRARIES "-L/opt/homebrew/Cellar/lapack/3.12.0/lib -L/opt/homebrew/Cellar/openblas/0.3.24/lib -llapacke -lopenblas")

add_library(lambda_svd kiss.cpp mmiodense.cpp svdalgs.cpp serial_tree.cpp dist_mpi_tree.cpp mpitimer.cpp)

include_directories(PUBLIC ${LAPACK_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR})
link_libraries(${LAPACK_LIBRARIES})

if (TARGET MPI::MPI_CXX)
    target_link_libraries(lambda_svd PUBLIC MPI::MPI_CXX)
else()
    target_compile_options(lambda_svd PUBLIC ${MPI_CXX_COMPILE_FLAGS})
    target_link_libraries(lambda_svd PUBLIC ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINKFLAGS})
    target_include_directories(lambda_svd PUBLIC ${MPI_CXX_INCLUDE_PATH})
endif()

add_subdirectory(svd_serial svd_serial)
add_subdirectory(svd_mpi svd_mpi)

