add_executable(svd_aws_bench svd_aws_bench.cpp)
target_link_libraries(svd_aws_bench lambda_svd)
aws_lambda_package_target(svd_aws_bench)

#add_executable(matrix_multiply matrix_multiply.cpp)
#target_link_libraries(matrix_multiply lambda_svd)
#aws_lambda_package_target(matrix_multiply)

#add_executable(svd_aws_test svd_aws_test.cpp)
#target_link_libraries(svd_aws_test lambda_svd)
#aws_lambda_package_target(svd_aws_test)
