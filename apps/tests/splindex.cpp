#include <sirius.h>

void test1(void)
{
    printf("\n");
    printf("split 17 elements between 4 ranks\n");
    printf("\n");
    for (int i = 0; i < 4; i++)
    {
        splindex<block> spl(17, 4, i);
        printf("rank : %i, local size : %i\n", i, spl.local_size());
        
        printf("local index and rank for each element:\n");
        for (int k = 0; k < 17; k++)
        {
            int iloc = spl.location(_splindex_offs_, k);
            int rank = spl.location(_splindex_rank_, k);
            printf(" global index : %i => local index : %i, rank: %i\n", k, iloc, rank);
        }
    }


}

void test2()
{
    printf("\n");
    printf("test2\n");

    for (int i = 0; i < 4; i++)
    {
        printf("rank : %i\n", i);
        splindex<block> spl(17, 4, i);
        
        #pragma omp parallel
        for (auto it = spl.begin(); it.valid(); it++)
        {
            printf("thread_id: %i, local index : %i, global index : %i\n", Platform::thread_id(), it.idx_local(), it.idx());
        }
    }
    

}

int main(int argn, char** argv)
{
    test1();
    test2();
}
