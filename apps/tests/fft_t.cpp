#include <sirius.h>

using namespace sirius;

void test1()
{
    double a1[] = {24, 0, 0};
    double a2[] = {0, 24, 0};
    double a3[] = {0, 0, 24};
    Global global;
    
    global.unit_cell()->set_lattice_vectors(a1, a2, a3);
    global.set_pw_cutoff(17.0);
    global.initialize();

    Timer t("fft::transform");
    
    int num_fft_threads = Platform::num_fft_threads();
    #pragma omp parallel default(shared) num_threads(num_fft_threads)
    {        
        int thread_id = omp_get_thread_num();
    
        #pragma omp for
        for (int i = 0; i < 716; i++)
        {
            global.reciprocal_lattice()->fft()->transform(1, thread_id);
            global.reciprocal_lattice()->fft()->transform(-1, thread_id);
        }
    }
    t.stop();
    
    Timer::print();
    Timer::clear();
}

int main(int argn, char **argv)
{
    Platform::initialize(1);
    test1();
}
