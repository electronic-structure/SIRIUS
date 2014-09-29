#include <sirius.h>

using namespace sirius;

void test1()
{
    double a1[] = {24, 0, 0};
    double a2[] = {0, 24, 0};
    double a3[] = {0, 0, 24};
    Global global(Platform::comm_world());
    
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

void test2()
{
#ifdef _GPU_
    printf("\n");
    printf("test2(): cuFFT transform\n");
    
    double a1[] = {20, 0, 0};
    double a2[] = {0, 22, 0};
    double a3[] = {0, 0, 24};
    Global global(Platform::comm_world());
    
    global.unit_cell()->set_lattice_vectors(a1, a2, a3);
    global.set_pw_cutoff(12.0);
    global.initialize();

    auto fft = global.reciprocal_lattice()->fft();

    int nfft_max = 4;
    FFT3D<gpu> fft_gpu(fft->grid_size(), nfft_max);

    /* allocate work area array */
    mdarray<char, 1> work_area(nullptr, fft_gpu.work_area_size());
    work_area.allocate_on_device();
    
    /* allocate space for spinor components */
    mdarray<double_complex, 2> buff(fft_gpu.size(), nfft_max);
    buff.allocate_on_device();
    
    std::vector<double_complex> fft1(fft->size());

    // loop over lowest harmonics in reciprocal space
    for (int i0 = -2; i0 <= 2; i0++)
    {
        for (int i1 = -2; i1 <= 2; i1++)
        {
            for (int i2 = -2; i2 <= 2; i2++)
            {
                memset(&fft1[0], 0, fft1.size() * sizeof(double_complex));
                // load a single harmonic
                fft1[fft->index(i0, i1, i2)] = double_complex(1.0, 0.0);
                fft->input(&fft1[0]);
                fft->transform(1);
                fft->output(&fft1[0]);
                
                buff.zero();
                for (int k = 0; k < nfft_max; k++)
                {
                    buff(fft->index(i0, i1, i2), k) = double_complex(1.0, 0.0);
                }
                buff.copy_to_device();
                fft_gpu.transform(1, buff.at<gpu>());
                buff.copy_to_host();
                
                double diff = 0.0;
                for (int k = 0; k < nfft_max; k++)
                {
                    for (int i = 0; i < fft->size(); i++) diff += abs(fft1[i] - buff(i, k));
                }
                printf("harmonic : %i %i %i, diff : %f\n", i0, i1, i2, diff);
            }
        }
    }
#endif
}

int main(int argn, char **argv)
{
    Platform::initialize(1);
    test1();
    test2();
    Platform::finalize();
}
