#include <sirius.h>

void test1(void)
{
    for (int i = 0; i < 4; i++)
    {
        splindex<block> spl(13, 4, i);
        std::cout << " size = " << spl.local_size() << std::endl; 
    }


}

int main(int argn, char** argv)
{
    test1();
}
