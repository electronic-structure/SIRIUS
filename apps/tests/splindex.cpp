#include <sirius.h>

void test1(void)
{
    printf("\n");
    printf("split 17 elements between 4 ranks\n");
    printf("\n");
    for (int i = 0; i < 4; i++)
    {
        splindex<splindex_t::block> spl(17, 4, i);
        printf("rank : %i, local size : %i\n", i, (int)spl.local_size());
        
        printf("local index and rank for each element:\n");
        for (int k = 0; k < 17; k++)
        {
            int iloc = (int)spl.local_index(k);
            int rank = spl.local_rank(k);
            printf(" global index : %i => local index : %i, rank: %i\n", k, iloc, rank);
        }
    }
}

void test1a(void)
{
    printf("\n");
    printf("split 17 elements between 4 ranks in block-cyclic distribution\n");
    printf("\n");
    for (int i = 0; i < 4; i++)
    {
        splindex<splindex_t::block_cyclic> spl(17, 4, i, 2);
        printf("rank : %i, local size : %i\n", i, (int)spl.local_size());
        
        printf("local index and rank for each element:\n");
        for (int k = 0; k < 17; k++)
        {
            int iloc = (int)spl.local_index(k);
            int rank = spl.local_rank(k);
            printf(" global index : %i => local index : %i, rank: %i\n", k, iloc, rank);
        }
    }
}

//void test2()
//{
//    printf("\n");
//    printf("test2\n");
//
//    for (int i = 0; i < 4; i++)
//    {
//        printf("rank : %i\n", i);
//        splindex<block> spl(17, 4, i);
//        
//        #pragma omp parallel
//        for (auto it = splindex_iterator<block>(spl); it.valid(); it++)
//        {
//            #pragma omp flush
//            printf("thread_id: %i, local index : %i, global index : %i\n", Platform::thread_id(), (int)it.idx_local(), (int)it.idx());
//        }
//    }
//}

void test3()
{
    printf("\n");
    printf("test3\n");
    for (int num_ranks = 1; num_ranks < 20; num_ranks++)
    {
        for (int N = 1; N < 1130; N++)
        {
            splindex<splindex_t::block> spl(N, num_ranks, 0);
            int sz = 0;
            for (int i = 0; i < num_ranks; i++) sz += (int)spl.local_size(i);
            if (sz != N) 
            {
                std::cout << "Error: wrong sum of local sizes." << std::endl;

                std::cout << "global index size: " << N << std::endl;
                std::cout << "computed global index size: " << sz << std::endl;
                std::cout << "number of ranks: " << num_ranks << std::endl;
                //std::cout << "block size: " << spl.block_size() << std::endl;
                
                for (int i = 0; i < num_ranks; i++) std::cout << "i, local_size(i): " << i << ", " << spl.local_size(i) << std::endl;

                exit(0);
            }
            for (int i = 0; i < N; i++)
            {
                int rank = spl.local_rank(i);
                int offset = (int)spl.local_index(i);
                if (i != (int)spl.global_index(offset, rank))
                {
                    std::cout << "Error: wrong index." << std::endl;
                    std::cout << "global index size: " << N << std::endl;
                    std::cout << "number of ranks: " << num_ranks << std::endl;
                    std::cout << "global index: " << i << std::endl;
                    std::cout << "rank, offset: " << rank << ", " << offset << std::endl;
                    //std::cout << "block size: " << spl.block_size() << std::endl;
                    std::cout << "computed global index: " << spl.global_index(offset, rank) << std::endl;
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
        for (int num_ranks = 1; num_ranks < 13; num_ranks++)
        {
            for (int N = 1; N < 1113; N++)
            {
                splindex<splindex_t::block_cyclic> spl(N, num_ranks, 0, bs);
                int sz = 0;
                for (int i = 0; i < num_ranks; i++) sz += (int)spl.local_size(i);
                if (sz != N) 
                {
                    std::cout << "wrong sum of local sizes" << std::endl;
                    exit(0);
                }

                for (int i = 0; i < N; i++)
                {
                    int rank = spl.local_rank(i);
                    int offset = (int)spl.local_index(i);
                    if (i != (int)spl.global_index(offset, rank))
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

void test5()
{
    printf("\n");
    printf("test5\n");
    for (int num_ranks = 1; num_ranks < 20; num_ranks++) {
        for (int N = 1; N < 1130; N++) {
            splindex<splindex_t::block> spl_tmp(N, num_ranks, 0);

            splindex<splindex_t::chunk> spl(N, num_ranks, 0, spl_tmp.counts());

            for (int i = 0; i < N; i++) {
                int rank = spl.local_rank(i);
                int offset = spl.local_index(i);
                if (i != spl.global_index(offset, rank)) {
                    std::cout << "wrong index" << std::endl;
                    exit(0);
                }
            }
        }
    }
}

int main(int argn, char** argv)
{
    sirius::initialize(1);
    test1();
    test1a();
    //test2();
    test3();
    test4();
    test5();
    sirius::finalize();
}
