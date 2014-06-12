#include <sirius.h>


#ifdef _SCALAPACK_
void test_diag(int num_gkvec, int num_aw, int nrow, int ncol)
{
    int blacs_handler = linalg<scalapack>::create_blacs_handler(MPI_COMM_WORLD);
    int context = blacs_handler;
    Cblacs_gridinit(&context, "C", nrow, ncol);

    int nrow1, ncol1, irow, icol;
    linalg<scalapack>::gridinfo(context, &nrow1, &ncol1, &irow, &icol);

    dmatrix<double_complex> a_n, a_t, ha_n, h, o;
    a_n.set_dimensions(num_gkvec, num_aw, context);
    ha_n.set_dimensions(num_gkvec, num_aw, context);
    a_t.set_dimensions(num_aw, num_gkvec, context);
    h.set_dimensions(num_gkvec, num_gkvec, context);
    o.set_dimensions(num_gkvec, num_gkvec, context);

    #ifdef _GPU_
    a_n.allocate_page_locked();
    ha_n.allocate_page_locked();
    a_t.allocate_page_locked();
    h.allocate_page_locked();
    o.allocate_page_locked();
    #else
    a_n.allocate();
    ha_n.allocate();
    a_t.allocate();
    h.allocate();
    o.allocate();
    #endif
    
    for (int i = 0; i < num_aw; i++)
    {
        for (int igk = 0; igk < num_gkvec; igk++)
        {
            double d = (i == igk) ? 1 : 0;
            a_n.set(igk, i, double_complex(d + std::pow(1.0 + i + igk, -2), 0));
            a_t.set(i, igk, double_complex(d + std::pow(1.0 + i + igk, -2), 0));
            ha_n.set(igk, i, double_complex(d + std::pow(1.0 + i + igk, -1), 0));
        }
    }

    h.zero();
    o.zero();

    //if (Platform::mpi_rank() == 0)
    //{
    //    printf("testing parallel zgemm with M, N, K = %i, %i, %i, opA = %i\n", M, N, K, transa);
    //    printf("nrow, ncol = %i, %i, bs = %i\n", nrow, ncol, linalg<scalapack>::cyclic_block_size());
    //}
    //Platform::barrier();
    
    
    sirius::Timer t1("zgemm", sirius::_global_timer_); 
    blas<cpu>::gemm(0, 0, num_gkvec, num_gkvec, num_aw, complex_one, a_n, a_t, complex_zero, o);
    blas<cpu>::gemm(0, 0, num_gkvec, num_gkvec, num_aw, complex_one, ha_n, a_t, complex_zero, h);
    t1.stop();
    
    std::string name = "scalapack";
    ev_solver_t gen_evp_solver_type = ev_scalapack;
    generalized_evp* gen_evp_solver = nullptr;

    if (name == "lapack") 
    {
        gen_evp_solver_type = ev_lapack;
    }
    else if (name == "scalapack") 
    {
        gen_evp_solver_type = ev_scalapack;
    }
    else if (name == "elpa1") 
    {
        gen_evp_solver_type = ev_elpa1;
    }
    else if (name == "elpa2") 
    {
        gen_evp_solver_type = ev_elpa2;
    }
    else if (name == "magma") 
    {
        gen_evp_solver_type = ev_magma;
    }
    else if (name == "plasma")
    {
        gen_evp_solver_type = ev_plasma;
    }
    else if (name == "rs_gpu")
    {
        gen_evp_solver_type = ev_rs_gpu;
    }
    else if (name == "rs_cpu")
    {
        gen_evp_solver_type = ev_rs_cpu;
    }
    else
    {
        error_local(__FILE__, __LINE__, "wrong eigen value solver");
    }

    /* create generalized eign-value solver */
    switch (gen_evp_solver_type)
    {
        case ev_lapack:
        {
            gen_evp_solver = new generalized_evp_lapack(1e-15);
            break;
        }
        case ev_scalapack:
        {
            gen_evp_solver = new generalized_evp_scalapack(nrow, ncol, context, -1.0);
            break;
        }
        //== case ev_elpa1:
        //== {
        //==     gen_evp_solver = new generalized_evp_elpa1(nrow, irow, ncol, icol, blacs_context_, 
        //==                                                mpi_grid().communicator(1 << _dim_row_),
        //==                                                mpi_grid().communicator(1 << _dim_col_));
        //==     break;
        //== }
        //== case ev_elpa2:
        //== {
        //==     gen_evp_solver = new generalized_evp_elpa2(nrow, irow, ncol, icol, blacs_context_, 
        //==                                                mpi_grid().communicator(1 << _dim_row_),
        //==                                                mpi_grid().communicator(1 << _dim_col_),
        //==                                                mpi_grid().communicator(1 << _dim_col_ | 1 << _dim_row_));
        //==     break;
        //== }
        case ev_magma:
        {
            gen_evp_solver = new generalized_evp_magma();
            break;
        }
        case ev_rs_gpu:
        {
            gen_evp_solver = new generalized_evp_rs_gpu(nrow, irow, ncol, icol, context);
            break;
        }
        case ev_rs_cpu:
        {
            gen_evp_solver = new generalized_evp_rs_cpu(nrow, irow, ncol, icol, context);
            break;
        }
        default:
        {
            error_local(__FILE__, __LINE__, "wrong generalized eigen-value solver");
        }
    }

    int num_bands = int(0.2 * num_gkvec);
    std::vector<double> eval(num_gkvec);
    dmatrix<double_complex> evec(num_gkvec, num_bands, context);
    
    sirius::Timer t2("diag", sirius::_global_timer_);
    gen_evp_solver->solve(num_gkvec, h.num_rows_local(), h.num_cols_local(), num_bands, 
                          h.ptr(), h.ld(), o.ptr(), o.ld(), &eval[0], evec.ptr(), evec.ld());
    t2.stop();
    delete gen_evp_solver;
    //#ifdef _GPU_
    //cuda_device_synchronize();
    //#endif
    //Platform::barrier();
    //t1.stop();
    //if (Platform::mpi_rank() == 0)
    //{
    //    printf("execution time (sec) : %12.6f\n", t1.value());
    //    printf("performance (GFlops) : %12.6f\n", 8e-9 * M * N * K / t1.value() / nrow / ncol);
    //}
    Cblacs_gridexit(context);
    linalg<scalapack>::free_blacs_handler(blacs_handler);
}
#endif

int main(int argn, char **argv)
{
    cmd_args args;
    args.register_key("--num_gkvec=", "{int} number of G+k vectors");
    args.register_key("--num_aw=", "{int} number of matching coefficients");
    args.register_key("--nrow=", "{int} number of row MPI ranks");
    args.register_key("--ncol=", "{int} number of column MPI ranks");
    args.register_key("--bs=", "{int} cyclic block size");

    args.parse_args(argn, argv);
    if (argn == 1)
    {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        exit(0);
    }

    int nrow = 1;
    int ncol = 1;

    if (args.exist("nrow")) nrow = args.value<int>("nrow");
    if (args.exist("ncol")) ncol = args.value<int>("ncol");

    int num_gkvec = args.value<int>("num_gkvec");
    int num_aw = args.value<int>("num_aw");

    Platform::initialize(true);

    #ifdef _SCALAPACK_
    int bs = args.value<int>("bs");
    linalg<scalapack>::set_cyclic_block_size(bs);
    test_diag(num_gkvec, num_aw, nrow, ncol);
    #else
    terminate(__FILE__, __LINE__, "not compiled with ScaLAPACK support");
    #endif

    sirius::Timer::print();

    Platform::finalize();
}
