#include <sirius.h>

using namespace sirius;

void test()
{
    int N = 200;
    mdarray<double_complex, 2> A(N, N);
    mdarray<double_complex, 2> B(N, N);
    mdarray<double_complex, 2> C(N, N);

    for (int j = 0; j < N; j++)
    {
        for (int i = 0; i < N; i++)
        {
            A(i, j) = 1.0;
            B(i, j) = 1.0;
        }
    }
    blas<cpu>::gemm(2, 0, N, N, N, A.ptr(), A.ld(), B.ptr(), B.ld(), C.ptr(), C.ld());
    Platform::comm_world().allreduce(C.ptr(), (int)C.size());

    std::cout << C(0, 0) << " " << C(N - 1, N - 1) << std::endl;

    A.allocate_on_device();
    A.copy_to_device();
    B.allocate_on_device();
    B.copy_to_device();
    C.allocate_on_device();

    blas<gpu>::gemm(2, 0, N, N, N, A.ptr_device(), A.ld(), B.ptr_device(), B.ld(), C.ptr_device(), C.ld());
    Platform::comm_world().allreduce(C.ptr_device(), (int)C.size());
    C.copy_to_host();

    std::cout << C(0, 0) << " " << C(N - 1, N - 1) << std::endl;
}

int main(int argn, char** argv)
{
    Platform::initialize(1);
    test();
    Platform::finalize();
}
