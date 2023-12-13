add_executable(svd_aws svd_aws.cpp)
target_link_libraries(svd_aws lambda_svd)

aws_lambda_package_target(svd_aws)
