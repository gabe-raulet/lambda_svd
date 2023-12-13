cmake_minimum_required(VERSION 3.10)
project(lambda_svd LANGUAGES CXX)

set(LAPACK_INCLUDE_DIRS "/opt/homebrew/Cellar/openblas/0.3.24/include")
set(LAPACK_LIBRARIES "-L/opt/homebrew/Cellar/lapack/3.12.0/lib -L/opt/homebrew/Cellar/openblas/0.3.24/lib -llapacke -lopenblas")

add_library(lambda_svd kiss.cpp mmiodense.cpp svdalgs.cpp)

include_directories(PUBLIC ${LAPACK_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR})
link_libraries(${LAPACK_LIBRARIES})

add_subdirectory(svd_serial)

