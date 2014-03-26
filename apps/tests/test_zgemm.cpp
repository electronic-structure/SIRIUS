#include <sirius.h>

void test_gemm(int M, int N, int K, mdarray<double_complex, 2>& c)
{
    sirius::Timer t("test_gemm"); 
    
    mdarray<double_complex, 2> a(M, K);
    mdarray<double_complex, 2> b(K, N);

    for (int j = 0; j < K; j++)
    {
        for (int i = 0; i < N; i++) a(i, j) = type_wrapper<double_complex>::random();
    }
    for (int j = 0; j < N; j++)
    {
        for (int i = 0; i < K; i++) b(i, j) = type_wrapper<double_complex>::random();
    }

    printf("testing zgemm with M, N, K = %i, %i, %i\n", M, N, K);
    sirius::Timer t1("gemm_only"); 
    blas<cpu>::gemm(0, 0, M, N, K, a.ptr(), a.ld(), b.ptr(), b.ld(), c.ptr(), c.ld());
    t1.stop();
    printf("execution time (sec) : %12.6f\n", t1.value());
    printf("performance (GFlops) : %12.6f\n", double(8 * M * N * K) / 1e9 / t1.value());
}

int main(int argn, char **argv)
{
    if (argn != 4)
    {
        printf("Usage: %s M N K\n", argv[0]);
        exit(0);
    }
    int M, N, K;
    std::istringstream(argv[1]) >> M;
    std::istringstream(argv[2]) >> N;
    std::istringstream(argv[3]) >> K;
       
    bool init_cublas = (argn == 1) ? true : false;
    Platform::initialize(true, init_cublas);

    //== mdarray<double_complex, 2> a(2000, 2000);
    //== mdarray<double_complex, 2> b(2000, 2000);
    //== mdarray<double_complex, 2> c(2000, 2000);


    //== a.allocate_on_device();
    //== b.allocate_on_device();
    //== c.allocate_on_device();
    //== 
    //== double_complex zone(1, 0);
    //== double_complex zzero(0, 0);

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
    
    mdarray<double_complex, 2> c(M, N);
    for (int j = 0; j < N; j++)
    {
        for (int i = 0; i < M; i++) c(i, j) = complex_zero;
    }
    test_gemm(M, N, K, c);

    //sirius::Timer::print();

    Platform::finalize();
}
