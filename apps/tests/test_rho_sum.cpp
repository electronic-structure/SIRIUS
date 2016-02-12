#include <sirius.h>
#include <thread>

using namespace sirius;

void test_rho_sum(double alat, double pw_cutoff, double wf_cutoff, int num_bands, std::vector<int>& mpi_grid)
{
    Communicator comm(MPI_COMM_WORLD);
    BLACS_grid blacs_grid(comm, mpi_grid[0], mpi_grid[1]);

    double a1[] = {alat, 0, 0};
    double a2[] = {0, alat, 0};
    double a3[] = {0, 0, alat};

    Simulation_parameters p;
    Unit_cell uc(p, comm);
    uc.set_lattice_vectors(a1, a2, a3);

    auto& rlv = uc.reciprocal_lattice_vectors();
    auto dims = Utils::find_translation_limits(pw_cutoff, rlv);

    int num_fft_workers = 1;
    int num_fft_threads = omp_get_max_threads();
    
    std::vector<FFT3D*> fft;
    for (int i = 0; i < num_fft_threads; i++)
        fft.push_back(new FFT3D(dims, num_fft_workers, blacs_grid.comm_row(), CPU));

    //Gvec gv_rho(vector3d<double>(0, 0, 0), pw_cutoff, rlv, fft[0], false);
    Gvec gv_wf(vector3d<double>(0, 0, 0), wf_cutoff, rlv, fft[0], false);

    if (comm.rank() == 0)
    {
        printf("MPI grid: %i %i\n", mpi_grid[0], mpi_grid[1]);
        printf("FFT dimensions: %i %i %i\n", fft[0]->size(0), fft[0]->size(1), fft[0]->size(2));
        //printf("num_gvec_rho: %i\n", gv_rho.num_gvec());
        printf("num_gvec_wf: %i\n", gv_wf.num_gvec());
    }

    mdarray<double, 1> rho_tot(fft[0]->local_size());
    rho_tot.zero();

    splindex<block> spl_gv_wf(gv_wf.num_gvec(), blacs_grid.num_ranks_row(), blacs_grid.rank_row());

    dmatrix<double_complex> psi(gv_wf.num_gvec(), num_bands, blacs_grid,
                                (int)splindex_base::block_size(gv_wf.num_gvec(), blacs_grid.num_ranks_row()), 1);
    psi.zero();

    for (int i = 0; i < num_bands; i++) 
    {
        for (int ig = 0; ig < gv_wf.num_gvec(); ig++)
        {
            psi.set(ig, i, double_complex(1, 0) / std::sqrt(double((gv_wf.num_gvec()))));
        }
    }

    if (psi.num_rows_local() != (int)spl_gv_wf.local_size())
    {
        TERMINATE("wrong index splitting");
    }

    mdarray<double_complex, 1> buf(gv_wf.num_gvec_loc());

    auto a2a_desc = blacs_grid.comm_row().map_alltoall(spl_gv_wf.counts(), gv_wf.counts()); 

    //== pstdout pout(blacs_grid.comm_row());

    //== pout.printf("-----------\n");
    //== pout.printf("rank: %i\n", blacs_grid.comm_row().rank());
    //== pout.printf("-----------\n");
    //== pout.printf("sendcounts : ");
    //== for (int j = 0; j < blacs_grid.comm_row().size(); j++)
    //==     pout.printf("%5i ", a2a_desc.sendcounts[j]);
    //== pout.printf("\n");
    //== pout.printf("sdispls    : ");
    //== for (int j = 0; j < blacs_grid.comm_row().size(); j++)
    //==     pout.printf("%5i ", a2a_desc.sdispls[j]);
    //== pout.printf("\n");
    //== pout.printf("recvcounts : ");
    //== for (int j = 0; j < blacs_grid.comm_row().size(); j++)
    //==     pout.printf("%5i ", a2a_desc.recvcounts[j]);
    //== pout.printf("\n");
    //== pout.printf("rdispls    : ");
    //== for (int j = 0; j < blacs_grid.comm_row().size(); j++)
    //==     pout.printf("%5i ", a2a_desc.rdispls[j]);
    //== pout.printf("\n");

    //== pout.flush();
    //== comm.barrier();

    //== PRINT(" ");

    int nested = omp_get_nested();
    omp_set_nested(1);

    Timer t2("sum_rho", comm);
    mdarray<double, 2> rho(fft[0]->local_size(), num_fft_threads);
    rho.zero();

    #pragma omp parallel num_threads(num_fft_threads)
    {
        int thread_id = omp_get_thread_num();
        #pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < psi.num_cols_local(); i++)
        {
            //PRINT("band: %i", i);
            //double_complex* inp = (gv_wf.num_gvec_loc() != 0) ? &buf(0) : NULL;

            //blacs_grid.comm_row().alltoall(&psi(0, i), &a2a_desc.sendcounts[0], &a2a_desc.sdispls[0], inp,
            //                               &a2a_desc.recvcounts[0], &a2a_desc.rdispls[0]);
            //
            ///* this is to make assert statements of mdarray happy */
            //int* map = (gv_wf.num_gvec_loc_ != 0) ? &gv_wf.index_map_local_to_local_(0) : NULL;
            //fft.input_pw(gv_wf.num_gvec_loc_, map, inp);
            //fft.transform(1);
            //for (int j = 0; j < (int)fft.local_size(); j++) rho(j) += (std::pow(std::real(fft.buffer(j)), 2) +
            //                                                           std::pow(std::imag(fft.buffer(j)), 2));
            fft[thread_id]->input(gv_wf.num_gvec(), gv_wf.index_map(), &psi(0, i));
            fft[thread_id]->transform(1, gv_wf.z_sticks_coord());
                        
            #pragma omp parallel for schedule(static) num_threads(fft[thread_id]->num_fft_workers())
            for (int ir = 0; ir < fft[thread_id]->local_size(); ir++)
            {
                auto z = fft[thread_id]->buffer(ir);
                rho(ir, thread_id) += (std::pow(z.real(), 2) + std::pow(z.imag(), 2));
            }
        }
    }
    t2.stop();
    omp_set_nested(nested);

    #pragma omp parallel
    {
        for (int i = 0; i < num_fft_threads; i++)
        {
            #pragma omp for schedule(static)
            for (int ir = 0; ir < fft[0]->local_size(); ir++) rho_tot(ir) += rho(ir, i);
        }
    }

    double nel = 0;
    for (int j = 0; j < (int)fft[0]->local_size(); j++) nel += rho_tot(j);
    comm.allreduce(&nel, 1);
    nel = nel / fft[0]->size();

    PRINT("num_bands: %i, num_electrons: %f", num_bands, nel);

    for (int i = 0; i < num_fft_threads; i++) delete fft[i];
}

int main(int argn, char **argv)
{
    cmd_args args;
    args.register_key("--alat=", "{double} lattice constant");
    args.register_key("--pw_cutoff=", "{double} plane-wave cutoff [a.u.^-1]");
    args.register_key("--wf_cutoff=", "{double} wave-function cutoff [a.u.^-1]");
    args.register_key("--num_bands=", "{int} number of bands");

    args.parse_args(argn, argv);
    if (args.exist("help"))
    {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        exit(0);
    }

    double alat = 7.0;
    double pw_cutoff = 16.0;
    double wf_cutoff = 5.0;
    int num_bands = 10;
    std::vector<int> mpi_grid;

    alat = args.value<double>("alat", alat);
    pw_cutoff = args.value<double>("pw_cutoff", pw_cutoff);
    wf_cutoff = args.value<double>("wf_cutoff", wf_cutoff);
    num_bands = args.value<int>("num_bands", num_bands);
    mpi_grid = args.value< std::vector<int> >("mpi_grid", std::vector<int>({1, 1}));


    Platform::initialize(1);

    test_rho_sum(alat, pw_cutoff, wf_cutoff, num_bands, mpi_grid);

    //#ifdef __PRINT_MEMORY_USAGE
    //MEMORY_USAGE_INFO();
    //#endif
    Timer::print();

    Platform::finalize();
}
