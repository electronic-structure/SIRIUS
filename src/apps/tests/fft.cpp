#include "sirius.h"

using namespace sirius;

// backward and forward transform of a random array
void test1()
{
    std::cout << "test1" << std::endl;
    //double a1[] = {10, 10, 0};
    //double a2[] = {0, 20, 3};
    //double a3[] = {1, 7, 35};
    double a1[] = {42, 42, 0};
    double a2[] = {42, 0, 42};
    double a3[] = {0, 42, 42};
     sirius::global.set_lattice_vectors(a1, a2, a3);
    sirius::global.set_pw_cutoff(12.0);
    sirius::global.reciprocal_lattice::init();
    
    std::cout << "grid size : " << global.fft().size() << std::endl;
    std::cout << "dimensions : " << global.fft().size(0) << " " 
                                 << global.fft().size(1) << " " 
                                 << global.fft().size(2) << std::endl;

    std::vector<complex16> fft1(global.fft().size());
    std::vector<complex16> fft2(global.fft().size());
    
    for (int i = 0; i < global.fft().size(); i++)
    {
        fft1[i] = complex16(double(rand()) / RAND_MAX, double(rand()) / RAND_MAX);
        fft2[i] = fft1[i];
    }
    
    sirius::Timer* t = new sirius::Timer(std::string("test1"));
    global.fft().transform(&fft1[0], -1);
    global.fft().transform(&fft1[0], 1);
    delete t;
    
    double d = 0.0;
    for (int i = 0; i < global.fft().size(); i++)
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
    global.set_lattice_vectors(a1, a2, a3);
    global.set_pw_cutoff(10.0);
    global.reciprocal_lattice::init();

    std::cout << "grid size : " << global.fft().size() << std::endl;
    
    double d = 0.0;

    std::vector<complex16> fft1(global.fft().size());
    for (int i0 = -2; i0 <= 2; i0++)
        for (int i1 = -2; i1 <= 2; i1++)
            for (int i2 = -2; i2 <= 2; i2++)
            {
                memset(&fft1[0], 0, fft1.size() * sizeof(complex16));
                fft1[global.fft().index(i0, i1, i2)] = complex16(1.0, 0.0);
                global.fft().transform(&fft1[0], 1);

                double gv[3];
                int fgv[] = {i0, i1, i2};
                global.get_coordinates<cartesian, reciprocal>(fgv, gv);

                mdarray<complex16,3> fft2(&fft1[0], global.fft().size(0), global.fft().size(1), global.fft().size(2));
                for (int j0 = 0; j0 < global.fft().size(0); j0++)
                    for (int j1 = 0; j1 < global.fft().size(1); j1++)
                        for (int j2 = 0; j2 < global.fft().size(2); j2++)
                        {
                            double frv[] = {double(j0) / global.fft().size(0), 
                                            double(j1) / global.fft().size(1), 
                                            double(j2) / global.fft().size(2)};
                            double rv[3];
                            global.get_coordinates<cartesian, direct>(frv, rv);
                            d += abs(fft2(j0, j1, j2) - exp(complex16(0.0, scalar_product(rv, gv))));
                        }
            }

    std::cout << "total diff : " << d << std::endl;
}

int main(int argn, char **argv)
{
    test1();
    test2();
    sirius::Timer::print();
}
