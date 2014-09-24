#include <sirius.h>

using namespace sirius;

extern "C" void randomize_on_gpu(double* ptr, size_t size);

void test()
{
    int N = 500;
    int K = 30000;
    mdarray<double_complex, 2> A(N, K);
    mdarray<double_complex, 2> B(K, N);
    mdarray<double_complex, 2> C(N, N);

    for (int j = 0; j < K; j++)
    {
        for (int i = 0; i < N; i++)
        {
            A(i, j) = 1.0;
            B(j, i) = 1.0;
        }
    }
    blas<cpu>::gemm(0, 0, N, N, K, A.at<cpu>(), A.ld(), B.at<cpu>(), B.ld(), C.at<cpu>(), C.ld());
    Platform::comm_world().allreduce(C.ptr(), (int)C.size());

    std::cout << C(0, 0) << " " << C(N - 1, N - 1) << std::endl;

    A.allocate_on_device();
    A.copy_to_device();
    B.allocate_on_device();
    B.copy_to_device();
    C.allocate_on_device();

    blas<gpu>::gemm(0, 0, N, N, K, A.at<gpu>(), A.ld(), B.at<gpu>(), B.ld(), C.at<gpu>(), C.ld());
    Platform::comm_world().allreduce(C.at<gpu>(), (int)C.size());
    C.copy_to_host();

    std::cout << C(0, 0) << " " << C(N - 1, N - 1) << std::endl;
}

void test1()
{
    mdarray<double_complex, 1> A(1000000);
    A.allocate_on_device();
    randomize_on_gpu((double*)A.at<gpu>(), A.size() * 2);

    A.copy_to_host();
    Platform::comm_world().allreduce(A.at<cpu>(), (int)A.size());
    
    mdarray<double_complex, 1> A_ref(1000000);
    A >> A_ref;

    Platform::comm_world().allreduce(A.at<gpu>(), (int)A.size());
    A.copy_to_host();

    for (int i = 0; i < 1000000; i++)
    {
        double d = std::abs(A(i) - A_ref(i));
        if (d > 1e-8) INFO << "i=" << i << " diff=" << d << std::endl;
    }
}

int main(int argn, char** argv)
{
    Platform::initialize(1);
    test();
    test1();
    Platform::finalize();
}
