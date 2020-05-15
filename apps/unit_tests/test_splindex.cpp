#include <sirius.hpp>
#include "testing.hpp"

using namespace sirius;

int test1()
{
    for (int num_ranks = 1; num_ranks < 20; num_ranks++) {
        for (int N = 1; N < 1130; N++) {
            splindex<splindex_t::block> spl(N, num_ranks, 0);
            int sz = 0;
            for (int i = 0; i < num_ranks; i++) {
                sz += (int)spl.local_size(i);
            }
            if (sz != N) {
                std::stringstream s;
                s << "test1: wrong sum of local sizes." << std::endl;
                s << "global index size: " << N << std::endl;
                s << "computed global index size: " << sz << std::endl;
                s << "number of ranks: " << num_ranks << std::endl;
                for (int i = 0; i < num_ranks; i++) {
                    s << "i, local_size(i): " << i << ", " << spl.local_size(i) << std::endl;
                }
                throw std::runtime_error(s.str());
            }
            for (int i = 0; i < N; i++) {
                int rank = spl.local_rank(i);
                int offset = (int)spl.local_index(i);
                if (i != (int)spl.global_index(offset, rank)) {
                    std::stringstream s;
                    s << "test1: wrong index." << std::endl;
                    s << "global index size: " << N << std::endl;
                    s << "number of ranks: " << num_ranks << std::endl;
                    s << "global index: " << i << std::endl;
                    s << "rank, offset: " << rank << ", " << offset << std::endl;
                    s << "computed global index: " << spl.global_index(offset, rank) << std::endl;
                    throw std::runtime_error(s.str());
                }
            }
        }
    }
    return 0;
}

int test2()
{
    for (int bs = 1; bs < 17; bs++) {
        for (int num_ranks = 1; num_ranks < 13; num_ranks++) {
            for (int N = 1; N < 1113; N++) {
                splindex<splindex_t::block_cyclic> spl(N, num_ranks, 0, bs);
                int sz = 0;
                for (int i = 0; i < num_ranks; i++) {
                    sz += (int)spl.local_size(i);
                }
                if (sz != N) {
                    std::stringstream s;

                    s << "test2: wrong sum of local sizes" << std::endl;
                    throw std::runtime_error(s.str());
                }

                for (int i = 0; i < N; i++) {
                    int rank = spl.local_rank(i);
                    int offset = (int)spl.local_index(i);
                    if (i != (int)spl.global_index(offset, rank)) {
                        std::stringstream s;
                        s << "test2: wrong index" << std::endl;
                        s << "bs = " << bs << std::endl
                          << "num_ranks =  " << num_ranks << std::endl
                          << "N = " << N << std::endl
                          << "idx = " << i << std::endl
                          << "rank = " << rank << std::endl
                          << "offset = " << offset << std::endl
                          << "computed index = " << spl.global_index(offset, rank) << std::endl;
                        throw std::runtime_error(s.str());
                    }
                }
            }
        }
    }
    return 0;
}

int test3()
{
    for (int num_ranks = 1; num_ranks < 20; num_ranks++) {
        for (int N = 1; N < 1130; N++) {
            splindex<splindex_t::block> spl_tmp(N, num_ranks, 0);

            splindex<splindex_t::chunk> spl(N, num_ranks, 0, spl_tmp.counts());

            for (int i = 0; i < N; i++) {
                int rank = spl.local_rank(i);
                int offset = spl.local_index(i);
                if (i != spl.global_index(offset, rank)) {
                    std::stringstream s;
                    s << "test3: wrong index" << std::endl;
                    throw std::runtime_error(s.str());
                }
            }
        }
    }
    return 0;
}

int main(int argn, char** argv)
{
    int err{0};
    err += call_test("test block index", test1);
    err += call_test("test block-cyclic index", test2);
    err += call_test("test chunk index", test3);
    return std::min(err, 1);
}
