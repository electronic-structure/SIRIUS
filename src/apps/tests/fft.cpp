#include "sirius.h"

// backward and forward transform of a random array
void test1()
{
    std::cout << "test1" << std::endl;
    double a1[] = {10, 10, 0};
    double a2[] = {0, 20, 3};
    double a3[] = {1, 7, 35};
    sirius_global.set_lattice_vectors(a1, a2, a3);
    sirius_global.set_pw_cutoff(20.0);
    sirius_global.init_fft_grid();

    std::cout << "grid size : " << sirius_global.fft().size() << std::endl;
    std::cout << "dimensions : " << sirius_global.fft().size(0) << " " 
                                 << sirius_global.fft().size(1) << " " 
                                 << sirius_global.fft().size(2) << std::endl;

    std::vector<complex16> fft1(sirius_global.fft().size());
    std::vector<complex16> fft2(sirius_global.fft().size());
    
    for (int i = 0; i < sirius_global.fft().size(); i++)
    {
        fft1[i] = complex16(double(rand()) / RAND_MAX, double(rand()) / RAND_MAX);
        fft2[i] = fft1[i];
    }
    
    sirius::Timer* t = new sirius::Timer(std::string("test1"));
    sirius_global.fft().transform(&fft1[0], -1);
    sirius_global.fft().transform(&fft1[0], 1);
    delete t;
    
    double d = 0.0;
    for (int i = 0; i < sirius_global.fft().size(); i++)
        d += abs(fft1[i] - fft2[i]);
    
    std::cout << "total diff : " << d << std::endl;
}

// transform of a single harmonic and comparison with an exact value  
void test2()
{
    std::cout << "test2" << std::endl;
    double a1[] = {10, 0, -7};
    double a2[] = {0, 10, 20};
    double a3[] = {-9, -11, 10};
    sirius_global.set_lattice_vectors(a1, a2, a3);
    sirius_global.set_pw_cutoff(10.0);
    sirius_global.init_fft_grid();

    std::cout << "grid size : " << sirius_global.fft().size() << std::endl;
    
    double d = 0.0;

    std::vector<complex16> fft1(sirius_global.fft().size());
    for (int i0 = -2; i0 <= 2; i0++)
        for (int i1 = -2; i1 <= 2; i1++)
            for (int i2 = -2; i2 <= 2; i2++)
            {
                memset(&fft1[0], 0, fft1.size() * sizeof(complex16));
                fft1[sirius_global.fft().index(i0, i1, i2)] = complex16(1.0, 0.0);
                sirius_global.fft().transform(&fft1[0], 1);

                double gv[3];
                int fgv[] = {i0, i1, i2};
                sirius_global.get_reciprocal_cartesian_coordinates(fgv, gv);

                mdarray<complex16,3> fft2(&fft1[0], sirius_global.fft().size(0), sirius_global.fft().size(1), sirius_global.fft().size(2));
                for (int j0 = 0; j0 < sirius_global.fft().size(0); j0++)
                    for (int j1 = 0; j1 < sirius_global.fft().size(1); j1++)
                        for (int j2 = 0; j2 < sirius_global.fft().size(2); j2++)
                        {
                            double frv[] = {double(j0) / sirius_global.fft().size(0), 
                                            double(j1) / sirius_global.fft().size(1), 
                                            double(j2) / sirius_global.fft().size(2)};
                            double rv[3];
                            sirius_global.get_cartesian_coordinates(frv, rv);
                            d += abs(fft2(j0, j1, j2) - exp(complex16(0.0, vector_scalar_product(rv, gv))));
                        }
            }

    std::cout << "total diff : " << d << std::endl;
}

int main(int argn, char **argv)
{
    test1();
    test2();
    sirius::Timer::print();

    /*std::vector<complex16> zfft(sirius_global.fft().size(), 0.0);
    zfft[sirius_global.fft().index(2, 0, 0)] = std::complex<double>(1.0, 0.0);
    sirius_global.fft().transform((double*)&zfft[0], 1);

    for (int i=0; i<sirius_global.fft().size(0); i++)
         std::cout << i << " " << real(zfft[i]) << " " << imag(zfft[i]) << std::endl;    
    
    */
    
}
