add_executable(svd_mpi svd_mpi.cpp)
add_executable(svd_mpi_bench svd_mpi_bench.cpp)

target_link_libraries(svd_mpi lambda_svd)
target_link_libraries(svd_mpi_bench lambda_svd)
