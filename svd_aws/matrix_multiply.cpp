#include <aws/core/Aws.h>
#include <aws/core/utils/json/JsonSerializer.h>
#include <aws/lambda-runtime/runtime.h>
#include <tcpunch.h>
#include <fcntl.h>
#include <Communicator.h>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <string>
#include "kiss.h"

/*
 * Start with just the random benchmark.
 */

using namespace aws::lambda_runtime;

static invocation_response my_handler(invocation_request const& req)
{
    using namespace Aws::Utils::Json;

    JsonValue json(req.payload);

    if (!json.WasParseSuccessful())
    {
        return invocation_response::failure("Failed to parse input JSON", "InvalidJSON");
    }

    auto v = json.View();

    int m = v.GetInteger("m");
    int n = v.GetInteger("n");
    kiss_init();

    double *A = (double*)malloc(m*n*sizeof(double));
    double *B = (double*)malloc(m*n*sizeof(double));
    double *C = (double*)malloc(m*n*sizeof(double));

    for (int i = 0; i < m*n; ++i)
        A[i] = kiss_unirandf();

    for (int i = 0; i < m*n; ++i)
        B[i] = kiss_unirandf();

    for (int i = 0; i < m*n; ++i)
        C[i] = A[i] + B[i];

    std::ostringstream res;
    for (int i = 0; i < m*n; ++i)
        res << C[i] << "\n";

    return invocation_response::success(res.str(), "application/txt");
}

int main(int argc, char *argv[])
{
    using namespace Aws;

    auto handler_fn = [](aws::lambda_runtime::invocation_request const& req)
    {
        return my_handler(req);
    };

    run_handler(handler_fn);
    return 0;
}
