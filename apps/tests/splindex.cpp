#include <sirius.h>

void test1(void)
{
    for (int i = 0; i < 4; i++)
    {
        splindex<block> spl(18, 4, i);
        std::cout << " size = " << spl.local_size() << std::endl; 

        for (int k = 0; k < 18; k++)
        {
            int ialoc = spl.location(_splindex_offs_, k);
            int rank = spl.location(_splindex_rank_, k);

            std::cout << "    ialoc, rank : " << ialoc << " " << rank << std::endl;
        }
    }


}

int main(int argn, char** argv)
{
    test1();
}
