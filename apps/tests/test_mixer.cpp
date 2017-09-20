#include <sirius.h>

using namespace sirius;

void test_mixer(Mixer<double>& mixer)
{
    int N = 10;
    std::vector<double> a(N), b(N);
    for (int i = 0; i < N; i++)
    {
        a[i] = i + 1;
        b[i] = i + 2;
    }

    double beta = mixer.beta();
    printf("beta: %f\n", beta);

    for (int i = 0; i < N; i++) mixer.input_shared(i, a[i]);
    mixer.initialize();

    for (int i = 0; i < N; i++) mixer.input_shared(i, b[i]);

    mixer.mix(1e-16);
    std::vector<double> c(N);

    for (int i = 0; i < N; i++) c[i] = mixer.output_shared(i);

    for (int i = 0; i < N; i++) printf("diff: %18.12f\n", std::abs(c[i] - (beta * b[i] + (1 - beta) * a[i])));
}

int main(int argn, char** argv)
{
    cmd_args args;

    args.parse_args(argn, argv);
    if (args.exist("help"))
    {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }

    sirius::initialize(1);

    int N = 10;
    double beta = 0.05;
    
    Linear_mixer<double> mixer1(N, 0, beta, mpi_comm_world());
    test_mixer(mixer1);
    
    Broyden1<double> mixer2(N, 0, 8, beta, mpi_comm_world());
    test_mixer(mixer2);

    Broyden2<double> mixer3(N, 0, 8, beta, 0.15, 100.0, mpi_comm_world());
    test_mixer(mixer3);

    sirius::finalize();
}
