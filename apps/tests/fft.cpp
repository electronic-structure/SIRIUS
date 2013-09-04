#include <sirius.h>

using namespace sirius;

// backward and forward transform of a random array
void test1()
{
    printf("\ntest1: backward and forward transform of a random array\n");
    
    Global global;

    double a1[] = {10, 10, 0};
    double a2[] = {0, 8, 3};
    double a3[] = {1, 7, 15};
    
    global.set_lattice_vectors(a1, a2, a3);
    global.set_pw_cutoff(20.0);
    global.add_atom_type(1, "Si");
    double pos[] = {0, 0, 0};
    
    global.add_atom(1, pos);
    
    global.initialize(1);
    
    global.Reciprocal_lattice::print_info();
    
    std::vector<complex16> fft1(global.fft().size());
    std::vector<complex16> fft2(global.fft().size());
    
    for (int i = 0; i < global.fft().size(); i++)
    {
        fft1[i] = complex16(double(rand()) / RAND_MAX, double(rand()) / RAND_MAX);
        fft2[i] = fft1[i];
    }
    
    Timer t("fft::transform|-1,+1");
    global.fft().input(&fft1[0]);
    global.fft().transform(-1);
    global.fft().output(&fft1[0]);
    
    global.fft().input(&fft1[0]);
    global.fft().transform(1);
    global.fft().output(&fft1[0]);
    t.stop();

    double d = 0.0;
    for (int i = 0; i < global.fft().size(); i++) d += pow(abs(fft1[i] - fft2[i]), 2);
    d = sqrt(d / global.fft().size());
    printf("\nRMS difference : %18.10e", d);
    if (d < 1e-10)
    {
        printf("  OK\n");
    }
    else
    {
        printf("  Fail\n");
    }
    
    Timer::print();
    Timer::clear();
}

void test2()
{
    printf("\ntest2: straightforward transform of a single harmonic and comparison with FFT result\n");
    double a1[] = {10, 0, -7};
    double a2[] = {0, 10, 20};
    double a3[] = {-9, -11, 10};

    Global global;

    global.set_lattice_vectors(a1, a2, a3);
    global.set_pw_cutoff(10.0);
    
    global.add_atom_type(1, "Si");
    double pos[] = {0, 0, 0};
    
    global.add_atom(1, pos);
    
    global.initialize(1);
    
    global.Reciprocal_lattice::print_info();
    printf("\n");

    std::vector<complex16> fft1(global.fft().size());

    // loop over lowest harmonics in reciprocal space
    for (int i0 = -2; i0 <= 2; i0++)
    {
        for (int i1 = -2; i1 <= 2; i1++)
        {
            for (int i2 = -2; i2 <= 2; i2++)
            {
                double d = 0.0;
                
                memset(&fft1[0], 0, fft1.size() * sizeof(complex16));
                // load a single harmonic
                fft1[global.fft().index(i0, i1, i2)] = complex16(1.0, 0.0);
                global.fft().input(&fft1[0]);
                global.fft().transform(1);
                global.fft().output(&fft1[0]);

                double gv[3];
                int fgv[] = {i0, i1, i2};
                global.get_coordinates<cartesian, reciprocal>(fgv, gv);

                // map FFT buffer to a 3D array
                mdarray<complex16, 3> fft2(&fft1[0], global.fft().size(0), global.fft().size(1), global.fft().size(2));

                // loop over 3D array (real space)
                for (int j0 = 0; j0 < global.fft().size(0); j0++)
                {
                    for (int j1 = 0; j1 < global.fft().size(1); j1++)
                    {
                        for (int j2 = 0; j2 < global.fft().size(2); j2++)
                        {
                            // get real space fractional coordinate
                            double frv[] = {double(j0) / global.fft().size(0), 
                                            double(j1) / global.fft().size(1), 
                                            double(j2) / global.fft().size(2)};
                            double rv[3];
                            global.get_coordinates<cartesian, direct>(frv, rv);
                            d += pow(abs(fft2(j0, j1, j2) - exp(complex16(0.0, Utils::scalar_product(rv, gv)))), 2);
                        }
                    }
                }
                d = sqrt(d / global.fft().size());
                printf("harmonic %4i %4i %4i, RMS difference : %18.10e", i0, i1, i2, d);
                if (d < 1e-10)
                {
                    printf("  OK\n");
                }
                else
                {
                    printf("  Fail\n");
                }
            }
        }
    }
    
    Timer::print();
    Timer::clear();
}

void test3()
{
    printf("\ntest3: show timers for large unit cell FFT\n");
    double a1[] = {60, 0, 0};
    double a2[] = {0, 60, 0};
    double a3[] = {0, 0, 60};
    Global global;
    
    global.set_lattice_vectors(a1, a2, a3);
    global.set_pw_cutoff(9.0);
    global.add_atom_type(1, "Si");
    double pos[] = {0, 0, 0};
    
    global.add_atom(1, pos);
    global.initialize(1);
    global.Reciprocal_lattice::print_info();

    Timer t("fft::transform");
    global.fft().transform(1);
    t.stop();
    
    Timer::print();
    Timer::clear();
}

int main(int argn, char **argv)
{
    Platform::set_num_fft_threads(1);
    Platform::initialize(1);
    test1();
    test2();
    test3();
}
