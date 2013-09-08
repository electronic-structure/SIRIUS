#include <sirius.h>

int main(int argn, char **argv)
{
    Platform::initialize(true);

    mdarray<complex16, 2> a(2000, 2000);
    mdarray<complex16, 2> b(2000, 2000);
    mdarray<complex16, 2> c(2000, 2000);


    a.allocate_on_device();
    b.allocate_on_device();
    c.allocate_on_device();
    
    complex16 zone(1, 0);
    complex16 zzero(0, 0);

    sirius::Timer t("zgemm", false);
    t.start();
    for (int i = 0; i < 10; i++)
    {
        blas<gpu>::gemm(0, 0, 2000, 2000, 2000, &zone, a.get_ptr_device(), a.ld(), b.get_ptr_device(), b.ld(), 
                        &zzero, c.get_ptr_device(), c.ld());
    }
    c.copy_to_host();
    std::cout << "after zgemm" << std::endl;
    t.stop();
    sirius::Timer::print();
}
