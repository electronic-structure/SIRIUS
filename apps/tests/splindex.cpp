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

void test1a(void)
{
    printf("\n");
    printf("split 17 elements between 4 ranks in block-cyclic distribution\n");
    printf("\n");
    splindex<block_cyclic>::set_cyclic_block_size(2);
    for (int i = 0; i < 4; i++)
    {
        splindex<block_cyclic> spl(17, 4, i);
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
        for (auto it = splindex_iterator<block>(spl); it.valid(); it++)
        {
            #pragma omp flush
            printf("thread_id: %i, local index : %i, global index : %i\n", Platform::thread_id(), it.idx_local(), it.idx());
        }
    }
    

}

void test3()
{
    printf("\n");
    printf("test3\n");
    for (int num_ranks = 1; num_ranks < 17; num_ranks++)
    {
        for (int N = 0; N < 113; N++)
        {
            splindex<block> spl(N, num_ranks, 0);
            int sz = 0;
            for (int i = 0; i < num_ranks; i++) sz += spl.local_size(i);
            if (sz != N) 
            {
                std::cout << "wrong sum of local sizes" << std::endl;
                exit(0);
            }
            for (int i = 0; i < N; i++)
            {
                int rank = spl.location(_splindex_rank_, i);
                int offset = spl.location(_splindex_offs_, i);
                if (i != spl.global_index(offset, rank))
                {
                    std::cout << "wrong index" << std::endl;
                    exit(0);
                }
            }
        }
    }
}

void test4()
{
    printf("\n");
    printf("test4\n");
    for (int bs = 1; bs < 17; bs++)
    {
        splindex<block_cyclic>::set_cyclic_block_size(bs);
        for (int num_ranks = 1; num_ranks < 13; num_ranks++)
        {
            for (int N = 1; N < 1113; N++)
            {
                splindex<block_cyclic> spl(N, num_ranks, 0);
                int sz = 0;
                for (int i = 0; i < num_ranks; i++) sz += spl.local_size(i);
                if (sz != N) 
                {
                    std::cout << "wrong sum of local sizes" << std::endl;
                    exit(0);
                }

                for (int i = 0; i < N; i++)
                {
                    int rank = spl.location(_splindex_rank_, i);
                    int offset = spl.location(_splindex_offs_, i);
                    if (i != spl.global_index(offset, rank))
                    {
                        std::cout << "wrong index" << std::endl;
                        std::cout << "bs = " << bs << std::endl
                                  << "num_ranks =  " << num_ranks << std::endl
                                  << "N = " << N << std::endl
                                  << "idx = " << i << std::endl
                                  << "rank = " << rank << std::endl
                                  << "offset = " << offset << std::endl
                                  << "computed index = " << spl.global_index(offset, rank) << std::endl;
                        exit(0);
                    }
                }
            }
        }
    }
}

int main(int argn, char** argv)
{
    Platform::initialize(1);
    test1();
    test1a();
    test2();
    test3();
    test4();
    Platform::finalize();
}
