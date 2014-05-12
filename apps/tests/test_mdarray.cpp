#include <sirius.h>

using namespace sirius;

mdarray<int, 1> f1()
{
    mdarray<int, 1> aa;
    aa.set_dimensions(4);
    aa.allocate();
    for (int i = 0; i < 4; i++) aa(i) = 200 + i;
    return aa;
}

int main(int argn, char **argv)
{
    Platform::initialize(1);

    int* memleak = new int[100];

    mdarray<int, 1> a1(4);
    for (int i = 0; i < 4; i++) a1(i) = 100 + i;

    mdarray<int, 1> a2 = f1();
    for (int i = 0; i < 4; i++)
    {
        std::cout << "a1(" << i << ")=" << a1(i) << std::endl;
        std::cout << "a2(" << i << ")=" << a2(i) << std::endl;
    }
    mdarray<int, 1> a3(std::move(a2));

//    a1.deallocate();
//
//    std::cout << "Deallocate a1" << std::endl;
//
//    for (int i = 0; i < 4; i++)
//    {
//        std::cout << "a2(" << i << ")=" << a2(i) << std::endl;
//    }
//
//
//    mdarray<int, 1> a3 = a2;
//    
    for (int i = 0; i < 4; i++)
    {
        std::cout << "a3(" << i << ")=" << a3(i) << std::endl;
    }

    mdarray<int, 1> a4;
    a4 = std::move(a3);

    

    Platform::finalize();
}
