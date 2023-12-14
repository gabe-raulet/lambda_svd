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
#include <fmi.h>
#include <string>
#include "kiss.h"
#include "mmiodense.h"
#include "fmi_wrapper.h"
#include "svdalgs.h"
#include "dist_fmi_tree.h"
#include "cblas.h"
#include "lapacke.h"

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


    int myrank, nprocs;
    std::string mattype;

    int m, n, p, r, s, a, seed;
    double *Aloc, *Utest, *Stest, *Vttest;

    auto v = json.View();
    std::string timestamp = v.GetString("timestamp");
    int num = v.GetInteger("number");
    nprocs = v.GetInteger("nprocs");
    myrank = v.GetInteger("myrank");
    //r = v.GetInteger("matrank");
    //p = v.GetInteger("trunc");
    //seed = v.GetInteger("seed");
    //mattype = v.GetString("mattype");


    auto comm = FMI::Communicator(myrank, nprocs, "fmi.json", timestamp);

    std::ostringstream res;

    auto t1 = std::chrono::high_resolution_clock::now();

    //kiss_seed(seed);


    int recv;

    if (myrank == 0)
    {
        fmi_send((void*)&num, sizeof(int), 1, comm);
    }
    else
    {
        fmi_recv((void*)&recv, sizeof(int), 0, comm);
    }

    res << "myrank=" << myrank << " sent " << num << " and received " << recv << "\n";
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
