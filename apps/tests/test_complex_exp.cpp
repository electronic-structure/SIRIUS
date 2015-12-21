#include <sirius.h>

using namespace sirius;

void test_complex_exp()
{
    int N = 100000000;
    mdarray<double_complex, 1> phase(N);
    phase.zero();

    for (int i = 0; i < N; i++)
    {
        phase(i) = type_wrapper<double_complex>::random();
    }

    Timer t("random_plus_exp");
    #pragma omp parallel for
    for (int i = 0; i < N; i++)
    {
        phase(i) = std::exp(phase(i));
    }
    double tval = t.stop();

    printf("number of evaluations: %i\n", N);

    printf("%8.2f M double complex exponents / sec.\n", N / tval / 1000000);

}

int main(int argn, char** argv)
{
    Platform::initialize(1);
    test_complex_exp();
    Platform::finalize();
}
