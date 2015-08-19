#include <sirius.h>

void test_alltoall(int num_gkvec, int num_bands)
{
    Communicator comm(MPI_COMM_WORLD);


    splindex<block> spl_gkvec(num_gkvec, comm.size(), comm.rank());
    splindex<block> spl_bands(num_bands, comm.size(), comm.rank());



    
    matrix<double_complex> a(spl_gkvec.local_size(), num_bands);
    matrix<double_complex> b(num_gkvec, spl_bands.local_size());

    for (int i = 0; i < num_bands; i++)
    {
        for (int j = 0; j < (int)spl_gkvec.local_size(); j++) a(j, i) = type_wrapper<double_complex>::random();
    }
    b.zero();

    auto h = a.hash();

    std::vector<int> sendcounts(comm.size());
    std::vector<int> sdispls(comm.size());
    std::vector<int> recvcounts(comm.size());
    std::vector<int> rdispls(comm.size());

    sdispls[0] = 0;
    rdispls[0] = 0;
    for (int rank = 0; rank < comm.size(); rank++)
    {
        sendcounts[rank] = int(spl_gkvec.local_size() * spl_bands.local_size(rank));
        if (rank) sdispls[rank] = sdispls[rank - 1] + sendcounts[rank - 1];

        recvcounts[rank] = int(spl_gkvec.local_size(rank) * spl_bands.local_size());
        if (rank) rdispls[rank] = rdispls[rank - 1] + recvcounts[rank - 1];
    }

    sirius::Timer t("alltoall", comm); 
    comm.alltoall(&a(0, 0), &sendcounts[0], &sdispls[0], &b(0, 0), &recvcounts[0], &rdispls[0]);
    double tval = t.stop();

    sdispls[0] = 0;
    rdispls[0] = 0;
    for (int rank = 0; rank < comm.size(); rank++)
    {
        sendcounts[rank] = int(spl_gkvec.local_size(rank) * spl_bands.local_size());
        if (rank) sdispls[rank] = sdispls[rank - 1] + sendcounts[rank - 1];

        recvcounts[rank] = int(spl_gkvec.local_size() * spl_bands.local_size(rank));
        if (rank) rdispls[rank] = rdispls[rank - 1] + recvcounts[rank - 1];
    }

    t.start();
    comm.alltoall(&b(0, 0), &sendcounts[0], &sdispls[0], &a(0, 0), &recvcounts[0], &rdispls[0]);
    tval = t.stop();

    if (a.hash() != h) printf("wrong hash\n");

    if (Platform::rank() == 0)
    {
        printf("alltoall time (sec) : %12.6f\n", tval);
    }
}

void test_alltoall_v2()
{
    Communicator comm(MPI_COMM_WORLD);

    std::vector<int> counts_in(comm.size(), 16);
    std::vector<double> sbuf(16);

    std::vector<int> counts_out(comm.size(), 0);
    counts_out[0] = 16 * comm.size();

    std::vector<double> rbuf(counts_out[comm.rank()]);

    auto a2a_desc = comm.map_alltoall(counts_in, counts_out);
    comm.alltoall(&sbuf[0], &a2a_desc.sendcounts[0], &a2a_desc.sdispls[0], &rbuf[0], &a2a_desc.recvcounts[0], &a2a_desc.rdispls[0]);

    PRINT("test_alltoall_v2 done");
    comm.barrier();
}

void test_alltoall_v3()
{
    Communicator comm(MPI_COMM_WORLD);

    std::vector<int> counts_in(comm.size(), 16);
    std::vector<double_complex> sbuf(16);

    std::vector<int> counts_out(comm.size(), 0);
    if (comm.size() == 1)
    {
        counts_out[0] = 16 * comm.size();
    }
    else
    {
        counts_out[0] = 8 * comm.size();
        counts_out[1] = 8 * comm.size();
    }

    std::vector<double_complex> rbuf(counts_out[comm.rank()]);

    auto a2a_desc = comm.map_alltoall(counts_in, counts_out);
    comm.alltoall(&sbuf[0], &a2a_desc.sendcounts[0], &a2a_desc.sdispls[0], &rbuf[0], &a2a_desc.recvcounts[0], &a2a_desc.rdispls[0]);

    PRINT("test_alltoall_v3 done");
    comm.barrier();
}

int main(int argn, char **argv)
{
    cmd_args args;
    args.register_key("--num_gkvec=", "{int} number of Gk-vectors");
    args.register_key("--num_bands=", "{int} number of bands");
    args.register_key("--repeat=", "{int} repeat test number of times");

    args.parse_args(argn, argv);
    if (argn == 1)
    {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        exit(0);
    }

    int num_gkvec = args.value<int>("num_gkvec");
    int num_bands = args.value<int>("num_bands");
    int repeat = args.value<int>("repeat", 1);

    Platform::initialize(true);

    for (int i = 0; i < repeat; i++) test_alltoall(num_gkvec, num_bands);

    test_alltoall_v2();
    test_alltoall_v3();

    Platform::finalize();
}
