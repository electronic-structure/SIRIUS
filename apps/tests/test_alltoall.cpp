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

    Platform::finalize();
}
