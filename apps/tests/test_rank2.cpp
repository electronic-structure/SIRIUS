#include <sirius.h>

using namespace sirius;

void f1(int num_gvec, int lmmax, int num_atoms)
{
    mdarray<double_complex, 3> alm(num_gvec, lmmax, num_atoms);
    mdarray<double_complex, 2> o(num_gvec, num_gvec);
    alm.randomize();
    o.zero();

    Timer t("rank2_update");

    for (int ia = 0; ia < num_atoms; ia++)
    {
        for (int lm = 0; lm < lmmax; lm++)
        {
            #pragma omp parallel for schedule(static)
            for (int ig2 = 0; ig2 < num_gvec; ig2++)
            {
                for (int ig1 = 0; ig1 < num_gvec; ig1++)
                {
                    o(ig1, ig2) += conj(alm(ig1, lm, ia)) * alm(ig2, lm, ia);
                }
            }
        }
    }
    double tval = t.stop();
    printf("\n");
    printf("execution time (sec) : %12.6f\n", tval);
    printf("performance (GFlops) : %12.6f\n", 8e-9 * num_gvec * num_gvec * lmmax * num_atoms / tval);
}

void f2(int num_gvec, int lmmax, int num_atoms)
{
    mdarray<double_complex, 3> alm(num_gvec, lmmax, num_atoms);
    mdarray<double_complex, 2> o(num_gvec, num_gvec);
    alm.randomize();
    o.zero();

    Timer t("zgemm");

    blas<cpu>::gemm(0, 2, num_gvec, num_gvec, lmmax * num_atoms, alm.ptr(), alm.ld(), alm.ptr(), alm.ld(), o.ptr(), o.ld());

    double tval = t.stop();
    printf("\n");
    printf("execution time (sec) : %12.6f\n", tval);
    printf("performance (GFlops) : %12.6f\n", 8e-9 * num_gvec * num_gvec * lmmax * num_atoms / tval);
}

int main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--NG=", "{int} number of G-vectors");
    args.register_key("--NL=", "{int} number of lm components");
    args.register_key("--NA=", "{int} number of atoms");

    args.parse_args(argn, argv);
    if (argn == 1)
    {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        exit(0);
    }

    Platform::initialize(1);

    int num_gvec = args.value<int>("NG");
    int lmmax = args.value<int>("NL");
    int num_atoms = args.value<int>("NA");

    f1(num_gvec, lmmax, num_atoms);

    f2(num_gvec, lmmax, num_atoms);

    Timer::print();

    Platform::finalize();
}
