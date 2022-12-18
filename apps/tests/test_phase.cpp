#include <sirius.hpp>

using namespace sirius;

void test1()
{
    int N = 30000000;
    std::vector<r3::vector<double>> a(N);

    for (int i = 0; i < N; i++) {
        double r = utils::random<double>();
        a[i] = {r, r, r};
    }
    std::vector<double_complex> phase(N, 0);
    double t1{0};
    double t2{0};

    for (int i = 0; i < 10; i++) {
        double t = -omp_get_wtime();
        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            phase[i] = std::exp(double_complex(0, dot(a[i], a[i])));
        }
        t1 += (t + omp_get_wtime());

        t = -omp_get_wtime();
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++) {
            phase[i] = std::exp(double_complex(0, dot(a[i], a[i])));
        }
        t2 += (t + omp_get_wtime());
    }
    printf("(default schedule) speed: %f million phase-factors / sec.\n", N * 10 / t1 / 1000000);
    printf("(static schedule) speed: %f million phase-factors / sec.\n", N * 10 / t2 / 1000000);

}

int main(int argn, char** argv)
{
    cmd_args args(argn, argv, {{}});

    test1();
}
