#include <aws/core/Aws.h>
#include <aws/core/utils/json/JsonSerializer.h>
#include <aws/lambda-runtime/runtime.h>
#include <tcpunch.h>
#include <fcntl.h>
#include <Communicator.h>
#include <sstream>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <fmi.h>
#include <string>
#include "kiss.h"
#include "mmiodense.h"
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
    nprocs = v.GetInteger("nprocs");
    myrank = v.GetInteger("myrank");
    r = v.GetInteger("matrank");
    p = v.GetInteger("trunc");
    seed = v.GetInteger("seed");
    mattype = v.GetString("mattype");

    a = 1;

    if (!strcmp(mattype.c_str(), "tall"))
    {
        n = r;
        m = a*r;
    }
    else if (!strcmp(mattype.c_str(), "wide"))
    {
        n = a*r;
        m = r;
    }
    else
    {
        m = n = r;
    }

    s = n / nprocs;

    std::ostringstream res;
    res << "myrank=" << myrank << ",nprocs=" << nprocs << ",m=" << m << ",n=" << n <<",r=" << r << ",p=" << p << ",seed=" << seed << ",mattype=" << mattype << ",a=" << a << "\n";
    auto comm = FMI::Communicator(myrank, nprocs, "fmi.json", timestamp);

    res << "hello from " << myrank << "\n";

    //auto t1 = std::chrono::high_resolution_clock::now();
    clock_t t1 = clock();

    kiss_seed(myrank*seed);

    Aloc = (double*)malloc(m*s*sizeof(double));

    for (int i = 0; i < m*s; ++i)
        Aloc[i] = kiss_unirandf();

    if (!myrank)
    {
        Utest = (double*)malloc(m*p*sizeof(double));
        Stest = (double*)malloc(p*sizeof(double));
        Vttest = (double*)malloc(p*n*sizeof(double));
    }
    else
    {
        Utest = Stest = Vttest = NULL;
    }

    //auto t2 = std::chrono::high_resolution_clock::now();
    clock_t t2 = clock();
    //auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1);
    double elapsed = ((double)(t2-t1))/CLOCKS_PER_SEC;
    res << myrank << " took " << elapsed << " seconds to initialize\n";
    //res << myrank << " took " << elapsed.count() << " seconds to initialize\n";

    t1 = clock();
    //t1 = std::chrono::high_resolution_clock::now();

    if (dist_fmi_tree(Aloc, Utest, Stest, Vttest, m, n, p, 0, myrank, nprocs, comm) != 0)
    {
        return invocation_response::failure("dist_fmi_tree failed", "dist_fmi_tree_error");
    }

    //t2 = std::chrono::high_resolution_clock::now();
    t2 = clock();
    //elapsed = std::chrono::duration_cast<std::chrono::seconds>(t2-t1);
    elapsed = ((double)(t2-t1))/CLOCKS_PER_SEC;

    res << myrank << " took " << elapsed << " seconds to run dist_fmi_tree\n";

    if (!myrank)
    {
	for (int i = 0; i < p; ++i) res << Stest[i] << "\n";
        free(Utest);
        free(Stest);
        free(Vttest);
    }

    free(Aloc);

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
