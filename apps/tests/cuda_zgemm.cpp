#include <sirius.h>

void test_gemm(int N, mdarray<complex16, 2>& c)
{
    sirius::Timer t("test_gemm"); 
    
    mdarray<complex16, 2> a(N, N);
    mdarray<complex16, 2> b(N, N);

    //#pragma omp parallel for
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            a(j, i) = complex16(double(rand()) / RAND_MAX, double(rand()) / RAND_MAX);
            b(j, i) = complex16(double(rand()) / RAND_MAX, double(rand()) / RAND_MAX);
        }
    }

    sirius::Timer t1("gemm_only"); 
    std::cout << "start of gemm loop" << std::endl;
    for (int i = 0; i < 10; i++)
    {
        std::cout << "i = " << i << std::endl;
        blas<cpu>::gemm(0, 0, N, N, N, a.get_ptr(), a.ld(), b.get_ptr(), b.ld(), c.get_ptr(), c.ld());
    }
    std::cout << "end of gemm loop" << std::endl;
}

int main(int argn, char **argv)
{
    bool init_cublas = (argn == 1) ? true : false;
    Platform::initialize(true, init_cublas);

    //== mdarray<complex16, 2> a(2000, 2000);
    //== mdarray<complex16, 2> b(2000, 2000);
    //== mdarray<complex16, 2> c(2000, 2000);


    //== a.allocate_on_device();
    //== b.allocate_on_device();
    //== c.allocate_on_device();
    //== 
    //== complex16 zone(1, 0);
    //== complex16 zzero(0, 0);

    //== sirius::Timer t("zgemm", false);
    //== t.start();
    //== for (int i = 0; i < 10; i++)
    //== {
    //==     blas<gpu>::gemm(0, 0, 2000, 2000, 2000, &zone, a.get_ptr_device(), a.ld(), b.get_ptr_device(), b.ld(), 
    //==                     &zzero, c.get_ptr_device(), c.ld());
    //== }
    //== c.copy_to_host();
    //== std::cout << "after zgemm" << std::endl;
    //== t.stop();
    
    int N = 3000;
    mdarray<complex16, 2> c(N, N);
    test_gemm(N, c);

    sirius::Timer::print();

    Platform::finalize();
}
