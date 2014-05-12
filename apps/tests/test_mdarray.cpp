#include <sirius.h>

using namespace sirius;

int main(int argn, char **argv)
{
    Platform::initialize(1);

    int* memleak = new int[100];

    mdarray<int, 1> a1(4);
    for (int i = 0; i < 4; i++) a1(i) = 100 + i;

    mdarray<int, 1> a2(a1);
    for (int i = 0; i < 4; i++)
    {
        std::cout << "a1(" << i << ")=" << a1(i) << std::endl;
        std::cout << "a2(" << i << ")=" << a2(i) << std::endl;
    }

    a1.deallocate();

    std::cout << "Deallocate a1" << std::endl;

    for (int i = 0; i < 4; i++)
    {
        std::cout << "a2(" << i << ")=" << a2(i) << std::endl;
    }


    mdarray<int, 1> a3 = a2;
    mdarray<int, 1> a4;
    a4 = a1;

    

    Platform::finalize();
}
