#include <sirius.h>

using namespace sirius;

void test_fft(double cutoff__)
{
    matrix3d<double> M = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

    FFT3D_grid fft_grid(cutoff__, M);

    FFT3D fft(fft_grid, mpi_comm_world(), CPU);

    
    Gvec gvec(vector3d<double>(0, 0, 0), M, cutoff__, fft_grid, mpi_comm_world().size(), false, false);

    if (mpi_comm_world().rank() == 0)
    {
        printf("num_gvec: %i\n", gvec.num_gvec());
    }
    MPI_grid mpi_grid(mpi_comm_world());

    Gvec_FFT_distribution gvec_fft_distr(gvec, mpi_grid);
    printf("num_gvec_fft: %i\n", gvec_fft_distr.num_gvec_fft());
    printf("offset_gvec_fft: %i\n", gvec_fft_distr.offset_gvec_fft());

    fft.prepare(gvec_fft_distr);

    mdarray<double_complex, 1> f(gvec.num_gvec());
    for (int ig = 0; ig < gvec.num_gvec(); ig++)
    {
        auto v = gvec[ig];
        if (mpi_comm_world().rank() == 0) printf("ig: %6i, gvec: %4i %4i %4i   ", ig, v[0], v[1], v[2]);
        f.zero();
        f[ig] = 1.0;
        fft.transform<1>(gvec_fft_distr, &f[gvec_fft_distr.offset_gvec_fft()]);

        double diff = 0;
        /* loop over 3D array (real space) */
        for (int j0 = 0; j0 < fft.grid().size(0); j0++)
        {
            for (int j1 = 0; j1 < fft.grid().size(1); j1++)
            {
                for (int j2 = 0; j2 < fft.local_size_z(); j2++)
                {
                    /* get real space fractional coordinate */
                    auto rl = vector3d<double>(double(j0) / fft.grid().size(0), 
                                               double(j1) / fft.grid().size(1), 
                                               double(fft.offset_z() + j2) / fft.grid().size(2));
                    int idx = fft.grid().index_by_coord(j0, j1, j2);

                    diff += std::pow(std::abs(fft.buffer(idx) - std::exp(double_complex(0.0, twopi * (rl * v)))), 2);
                }
            }
        }
        mpi_comm_world().allreduce(&diff, 1);
        diff = std::sqrt(diff / fft.size());
        if (mpi_comm_world().rank() == 0)
        {
            printf("error : %18.10e", diff);
            if (diff < 1e-10)
            {
                printf("  OK\n");
            }
            else
            {
                printf("  Fail\n");
                exit(1);
            }
        }
    }

    fft.dismiss();
}

//void test2(vector3d<int> const& dims__, double cutoff__, std::vector<int> mpi_grid_dims__)
//{
//    Communicator comm(MPI_COMM_WORLD);
//    MPI_grid mpi_grid(mpi_grid_dims__, comm);
//
//    matrix3d<double> M;
//    M(0, 0) = M(1, 1) = M(2, 2) = 1.0;
//
//    FFT3D fft1(dims__, Platform::max_num_threads(), mpi_grid.communicator(1 << 1), CPU);
//    auto dims2 = dims__;
//    for (int i: {0, 1, 2}) dims2[i] *= 2;
//    FFT3D fft2(dims2, Platform::max_num_threads(), mpi_grid.communicator(1 << 1), CPU);
//
//
//    Gvec gvec1(vector3d<double>(0, 0, 0), M, cutoff__, fft1.grid(), fft1.comm(), mpi_grid.communicator(1 << 0).size(), false, false);
//
//    FFT3D* fft = &fft2;
//
//    mdarray<double_complex, 1> psi_tmp(gvec1.num_gvec());
//    for (int ig = 0; ig < std::min(gvec1.num_gvec(), 100); ig++)
//    {
//        auto v = gvec1[ig];
//        printf("ig: %i, gvec: %i %i %i\n", ig, v[0], v[1], v[2]);
//        psi_tmp.zero();
//        psi_tmp(ig) = 1.0;
//        fft->transform<1>(gvec1, &psi_tmp(gvec1.offset_gvec_fft()));
//
//        double diff = 0;
//        /* loop over 3D array (real space) */
//        for (int j0 = 0; j0 < fft->grid().size(0); j0++)
//        {
//            for (int j1 = 0; j1 < fft->grid().size(1); j1++)
//            {
//                for (int j2 = 0; j2 < fft->local_size_z(); j2++)
//                {
//                    /* get real space fractional coordinate */
//                    auto rl = vector3d<double>(double(j0) / fft->grid().size(0), 
//                                               double(j1) / fft->grid().size(1), 
//                                               double(fft->offset_z() + j2) / fft->grid().size(2));
//                    int idx = fft->grid().index_by_coord(j0, j1, j2);
//
//                    diff += std::pow(std::abs(fft->buffer(idx) - std::exp(double_complex(0.0, twopi * (rl * v)))), 2);
//                }
//            }
//        }
//        diff = std::sqrt(diff / fft->size());
//        printf("RMS difference : %18.10e", diff);
//        if (diff < 1e-10)
//        {
//            printf("  OK\n");
//        }
//        else
//        {
//            printf("  Fail\n");
//        }
//    }
//}

//void test3(vector3d<int> const& dims__, double cutoff__)
//{
//    Communicator comm(MPI_COMM_WORLD);
//
//    matrix3d<double> M;
//    M(0, 0) = M(1, 1) = M(2, 2) = 1.0;
//
//    FFT3D fft(dims__, Platform::max_num_threads(), comm, CPU);
//
//    Gvec gvec(vector3d<double>(0, 0, 0), M, cutoff__, fft.grid(), comm, 1, false, false);
//    Gvec gvec_r(vector3d<double>(0, 0, 0), M, cutoff__, fft.grid(), comm, 1, false, true);
//
//    printf("num_gvec: %i, num_gvec_reduced: %i\n", gvec.num_gvec(), gvec_r.num_gvec());
//    printf("num_gvec_loc: %i %i\n", gvec.num_gvec(comm.rank()), gvec_r.num_gvec(comm.rank()));
//
//
//    //mdarray<double_complex, 1> phi(gvec.num_gvec());
//    //for (int i = 0; i < fft.size(); i++) fft.buffer(i) = type_wrapper<double>::random();
//    //fft.transform<-1>(gvec, &phi(gvec.offset_gvec_fft()));
//
//    //for (size_t i = 0; i < gvec.z_columns().size(); i++)
//    //{
//    //    auto zcol = gvec.z_columns()[i];
//    //    printf("x,y: %3i %3i\n", zcol.x, zcol.y);
//    //    for (size_t j = 0; j < zcol.z.size(); j++)
//    //    {
//    //        printf("z: %3i, val: %12.6f %12.6f\n", zcol.z[j], phi(zcol.offset + j).real(), phi(zcol.offset + j).imag());
//    //    }
//    //}
//
//    mdarray<double_complex, 1> phi(gvec_r.num_gvec());
//    for (int i = 0; i < gvec_r.num_gvec(); i++) phi(i) = type_wrapper<double_complex>::random();
//    phi(0) = 1.0;
//    fft.transform<1>(gvec_r, &phi(gvec.offset_gvec_fft()));
//
//    mdarray<double_complex, 1> phi1(gvec_r.num_gvec());
//    fft.transform<-1>(gvec_r, &phi1(gvec.offset_gvec_fft()));
//
//    
//    double diff = 0;
//    for (int i = 0; i < gvec_r.num_gvec(); i++)
//    {
//        diff += std::abs(phi(i) - phi1(i));
//    }
//    printf("diff: %18.12f\n", diff);
//
//
//}

int main(int argn, char **argv)
{
    cmd_args args;
    args.register_key("--cutoff=", "{double} cutoff radius in G-space");

    args.parse_args(argn, argv);
    if (args.exist("help"))
    {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }

    double cutoff = args.value<double>("cutoff", 5);

    sirius::initialize(1);

    test_fft(cutoff);

    runtime::Timer::print();
    
    sirius::finalize();
    return 0;
}
