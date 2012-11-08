#include "sirius.h"

void test1(void)
{
    for (int i = 0; i < 4; i++)
    {
        splindex spi(19, intvec(4), intvec(i));
        std::cout << "begin = " << spi.begin() << " end = " << spi.end() << std::endl; 
    }


}

void test2(void)
{
    for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++)
    {
        splindex spi(19, intvec(2, 2), intvec(i, j));
        std::cout << "begin = " << spi.begin() << " end = " << spi.end() << std::endl; 
    }


}

int main(int argn, char** argv)
{
    std::cout << "1D grid" << std::endl;
    test1();
    std::cout << "2D grid" << std::endl;
    test2();
}
