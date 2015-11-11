#include <sirius.h>
#include <thread>

#include <wave_functions.h>

using namespace sirius;

void slab_to_panel(int n__, mdarray<double_complex, 2>& slab__, mdarray<double_complex, 2>& panel__,
                   Gvec& gvec__, MPI_grid& mpi_grid__)
{
    Timer t("slab_to_panel");

    /* entire communicator */
    auto& comm = mpi_grid__.communicator();
    /* number of column ranks */
    int num_ranks_col = mpi_grid__.communicator(1 << 0).size();
    /* row rank */
    int rank_row = mpi_grid__.communicator(1 << 1).rank();
    
    /* this is how n columns of matrix will be distributed in panel */
    splindex<block> panel_col_distr(n__, num_ranks_col, mpi_grid__.communicator(1 << 0).rank());
    /* local number of columns */
    int n_loc = static_cast<int>(panel_col_distr.local_size());
    
    /* store the number of G-vectors to be received by this rank */
    block_data_descriptor gvec_slab_distr(num_ranks_col);
    for (int i = 0; i < num_ranks_col; i++)
    {
        gvec_slab_distr.counts[i] = gvec__.num_gvec(rank_row * num_ranks_col + i);
    }
    gvec_slab_distr.calc_offsets();

    assert(gvec_slab_distr.offsets[num_ranks_col - 1] + gvec_slab_distr.counts[num_ranks_col - 1] == gvec__.num_gvec_fft());
    
    /* receiving buffer */
    mdarray<double_complex, 1> buf(n_loc * gvec__.num_gvec_fft());

    int rank = comm.rank();
    int num_gvec = gvec__.num_gvec(rank);
    
    /* send parts of slab
     * +---+---+--+
     * |   |   |  |  <- irow = 0
     * +---+---+--+
     * |   |   |  |
     * ............
     * ranks in flat and 2D grid are related as: rank = irow * ncol + icol */
    for (int icol = 0; icol < num_ranks_col; icol++)
    {
        int dest_rank = comm.cart_rank({icol, rank / num_ranks_col});
        comm.isend(&slab__(0, panel_col_distr.global_offset(icol)),
                   static_cast<int>(num_gvec * panel_col_distr.local_size(icol)),
                   dest_rank, rank % num_ranks_col);
    }
    
    /* receive parts of panel
     *                 n_loc
     *                 +---+  
     *                 |   |
     * gvec_slab_distr +---+
     *                 |   | 
     *                 +---+ */
    for (int i = 0; i < num_ranks_col; i++)
    {
        int src_rank = rank_row * num_ranks_col + i;
        comm.recv(&buf(gvec_slab_distr.offsets[i] * n_loc), gvec_slab_distr.counts[i] * n_loc, src_rank, i);
    }
    
    /* reorder received blocks to make G-vector index continuous */
    #pragma omp parallel for
    for (int i = 0; i < n_loc; i++)
    {
        for (int j = 0; j < num_ranks_col; j++)
        {
            std::memcpy(&panel__(gvec_slab_distr.offsets[j], i),
                        &buf(gvec_slab_distr.offsets[j] * n_loc + gvec_slab_distr.counts[j] * i),
                        gvec_slab_distr.counts[j] * sizeof(double_complex));
        }
    }
}

void panel_to_slab(int n__, mdarray<double_complex, 2>& panel__, mdarray<double_complex, 2>& slab__,
                   Gvec& gvec__, MPI_grid& mpi_grid__)
{
    Timer t("panel_to_slab");

    /* entire communicator */
    auto& comm = mpi_grid__.communicator();
    /* number of column ranks */
    int num_ranks_col = mpi_grid__.communicator(1 << 0).size();
    /* row rank */
    int rank_row = mpi_grid__.communicator(1 << 1).rank();
    
    /* this is how n columns of matrix will be distributed in panel */
    splindex<block> panel_col_distr(n__, num_ranks_col, mpi_grid__.communicator(1 << 0).rank());
    /* local number of columns */
    int n_loc = static_cast<int>(panel_col_distr.local_size());
    
    /* store the number of G-vectors to be sent by this rank */
    block_data_descriptor gvec_slab_distr(num_ranks_col);
    for (int i = 0; i < num_ranks_col; i++)
    {
        gvec_slab_distr.counts[i] = gvec__.num_gvec(rank_row * num_ranks_col + i);
    }
    gvec_slab_distr.calc_offsets();

    assert(gvec_slab_distr.offsets[num_ranks_col - 1] + gvec_slab_distr.counts[num_ranks_col - 1] == gvec__.num_gvec_fft());
    
    /* sending buffer */
    mdarray<double_complex, 1> buf(n_loc * gvec__.num_gvec_fft());

    int rank = comm.rank();
    int num_gvec = gvec__.num_gvec(rank);

    /* reorder sending blocks */
    #pragma omp parallel for
    for (int i = 0; i < n_loc; i++)
    {
        for (int j = 0; j < num_ranks_col; j++)
        {
            std::memcpy(&buf(gvec_slab_distr.offsets[j] * n_loc + gvec_slab_distr.counts[j] * i),
                        &panel__(gvec_slab_distr.offsets[j], i),
                        gvec_slab_distr.counts[j] * sizeof(double_complex));
        }
    }

    for (int i = 0; i < num_ranks_col; i++)
    {
        int dest_rank = rank_row * num_ranks_col + i;
        comm.isend(&buf(gvec_slab_distr.offsets[i] * n_loc), gvec_slab_distr.counts[i] * n_loc, dest_rank, i);
    }
    
    for (int icol = 0; icol < num_ranks_col; icol++)
    {
        int src_rank = comm.cart_rank({icol, rank / num_ranks_col});
        comm.recv(&slab__(0, panel_col_distr.global_offset(icol)),
                   static_cast<int>(num_gvec * panel_col_distr.local_size(icol)),
                   src_rank, rank % num_ranks_col);
    }
}

void test_fft(vector3d<int> const& dims__, double cutoff__, int num_bands__, std::vector<int> mpi_grid_dims__)
{
    Communicator comm(MPI_COMM_WORLD);
    MPI_grid mpi_grid(mpi_grid_dims__, comm);

    matrix3d<double> M;
    M(0, 0) = M(1, 1) = M(2, 2) = 1.0;

    FFT3D fft(dims__, Platform::max_num_threads(), mpi_grid.communicator(1 << 1), CPU);
    
    Gvec gvec(vector3d<double>(0, 0, 0), M, cutoff__, fft.fft_grid(), fft.comm(), mpi_grid.communicator(1 << 0).size(), false);

    splindex<block> panel_col_distr(num_bands__, mpi_grid.communicator(1 << 0).size(), mpi_grid.communicator(1 << 0).rank());
    mdarray<double_complex, 2> psi_panel(gvec.num_gvec_fft(), panel_col_distr.local_size());
   
    mdarray<double_complex, 2> psi_slab(gvec.num_gvec(mpi_grid.communicator().rank()), num_bands__);
    mdarray<double_complex, 2> psi_slab_2(gvec.num_gvec(mpi_grid.communicator().rank()), num_bands__);

    Wave_functions psi_in(num_bands__, gvec, mpi_grid);
    Wave_functions psi_out(num_bands__, gvec, mpi_grid);
    
    if (comm.rank() == 0)
    {
        printf("num_gvec: %i\n", gvec.num_gvec());
    }
    printf("num_gvec_fft: %i\n", gvec.num_gvec_fft());
    printf("num_gvec_loc: %i\n", gvec.num_gvec(comm.rank()));
    printf("num_gvec_loc: %i\n", psi_in.num_gvec_loc());
    

    for (int i = 0; i < num_bands__; i++)
    {
        for (int j = 0; j < psi_in.num_gvec_loc(); j++)
        {
            psi_in(j, i) = type_wrapper<double_complex>::random();
        }
    }
    psi_in.slab_to_panel(num_bands__);

    splindex<block> spl_n(num_bands__, mpi_grid.communicator(1 << 0).size(), mpi_grid.communicator(1 << 0).rank());
    for (int i = 0; i < spl_n.local_size(); i++)
    {
        fft.transform<1>(psi_in.gvec(), &psi_in.panel(0, i));
        fft.transform<-1>(psi_out.gvec(), &psi_out.panel(0, i));
    }
    
    psi_out.panel_to_slab(num_bands__);

    double diff = 0;
    for (int i = 0; i < num_bands__; i++)
    {
        for (int j = 0; j < psi_in.num_gvec_loc(); j++)
        {
            diff += std::abs(psi_in(j, i) - psi_out(j, i));
        }
    }
    if (diff > 1e-12) TERMINATE("Fail");

    if (comm.rank() == 0) printf("Ok\n");




    //psi_slab.zero();
    //for (int i = 0; i < num_bands__; i++)
    //{
    //    for (int j = 0; j < gvec.num_gvec(mpi_grid.communicator().rank()); j++)
    //    {
    //        int ig = gvec.offset_gvec(mpi_grid.communicator().rank()) + j;
    //        if (ig == i) psi_slab(j, i) = 1;
    //    }
    //}



    //for (int i = 0; i < (int)panel_col_distr.local_size(); i++)
    //{
    //    for (int j = 0; j < gvec.num_gvec_fft(); j++) psi_panel(j, i) = type_wrapper<double_complex>::random();
    //}



    //slab_to_panel(num_bands__, psi_slab, psi_panel, gvec, mpi_grid);

    ////pstdout pout(comm);
    ////pout.printf("rank: %i\n", comm.rank());
    ////for (int i = 0; i < gvec.num_gvec_loc(); i++)
    ////{
    ////    pout.printf("row: %4i ", i);
    ////    for (int j = 0; j < panel_col_distr.local_size(); j++)
    ////    {
    ////        pout.printf("{%12.6f %12.6f} ", psi_panel(i, j).real(), psi_panel(i, j).imag());
    ////    }
    ////    pout.printf("\n");
    ////}
    ////pout.flush();






    ////Timer::delay(1);
    ////STOP();

    //mdarray<double_complex, 1> psi_tmp(gvec.num_gvec_fft());
    //for (int igloc = 0; igloc < (int)panel_col_distr.local_size(); igloc++)
    ////for (int ig = 0; ig < gvec.num_gvec(); ig++)
    //{
    //    int ig = static_cast<int>(panel_col_distr[igloc]);
    //    auto v = gvec[ig];
    //    printf("ig: %i, gvec: %i %i %i\n", ig, v[0], v[1], v[2]);
    //    //psi.zero();
    //    //psi(ig) = 1.0;
    //    //fft.transform<1>(gvec, &psi(gvec.gvec_offset()));
    //    fft.transform<1>(gvec, &psi_panel(0, igloc));
    //    fft.transform<-1>(gvec, &psi_tmp(0));

    //    double diff = 0;
    //    for (int i = 0; i < gvec.num_gvec_fft(); i++)
    //    {
    //        diff += std::pow(std::abs(psi_panel(i, igloc) - psi_tmp(i)), 2);
    //    }
    //    diff = std::sqrt(diff / gvec.num_gvec_fft());
    //    
    //    //== double diff = 0;
    //    //== /* loop over 3D array (real space) */
    //    //== for (int j0 = 0; j0 < fft.fft_grid().size(0); j0++)
    //    //== {
    //    //==     for (int j1 = 0; j1 < fft.fft_grid().size(1); j1++)
    //    //==     {
    //    //==         for (int j2 = 0; j2 < fft.local_size_z(); j2++)
    //    //==         {
    //    //==             /* get real space fractional coordinate */
    //    //==             auto rl = vector3d<double>(double(j0) / fft.fft_grid().size(0), 
    //    //==                                        double(j1) / fft.fft_grid().size(1), 
    //    //==                                        double(fft.offset_z() + j2) / fft.fft_grid().size(2));
    //    //==             int idx = fft.fft_grid().index_by_coord(j0, j1, j2);

    //    //==             diff += std::pow(std::abs(fft.buffer(idx) - std::exp(double_complex(0.0, twopi * (rl * v)))), 2);
    //    //==         }
    //    //==     }
    //    //== }
    //    //== diff = std::sqrt(diff / fft.size());
    //    printf("RMS difference : %18.10e", diff);
    //    if (diff < 1e-10)
    //    {
    //        printf("  OK\n");
    //    }
    //    else
    //    {
    //        printf("  Fail\n");
    //    }
    //}
}

int main(int argn, char **argv)
{
    cmd_args args;
    args.register_key("--dims=", "{vector3d<int>} FFT dimensions");
    args.register_key("--cutoff=", "{double} cutoff radius in G-space");
    args.register_key("--num_bands=", "{int} number of bands");
    args.register_key("--mpi_grid=", "{vector2d<int>} MPI grid");

    args.parse_args(argn, argv);
    if (argn == 1)
    {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        exit(0);
    }

    vector3d<int> dims = args.value< vector3d<int> >("dims");
    double cutoff = args.value<double>("cutoff", 1);
    int num_bands = args.value<int>("num_bands", 50);
    std::vector<int> mpi_grid = args.value< std::vector<int> >("mpi_grid", {1, 1});

    Platform::initialize(1);

    test_fft(dims, cutoff, num_bands, mpi_grid);
    
    Timer::print();

    Platform::finalize();
}
