cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
project(lambda_svd LANGUAGES CXX)

add_subdirectory(./fmi)

#set(CMAKE_PREFIX_PATH "/opt/software/lib64/aws-lambda-runtime/cmake/;/opt/aws-sdk-cpp/cmake")
find_package(aws-lambda-runtime REQUIRED)
find_package(AWSSDK COMPONENTS s3)
include(FindBLAS)

target_include_directories(lambda_svd PUBLIC ${FMI_INCLUDE_DIRS} ${LAPACK_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR})

add_library(lambda_svd kiss.cpp mmiodense.cpp svdalgs.cpp serial_tree.cpp dist_fmi_tree.cpp fmi_wrapper.cpp)

target_link_libraries(lambda_svd PUBLIC FMI AWS::aws-lambda-runtime ${AWSSDK_LINK_LIBRARIES})

add_subdirectory(./svd_serial)
add_subdirectory(./svd_fmi)
