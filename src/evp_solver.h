/// \todo scapalack-based solvers can exctract grid information from blacs context

/// Base class for the standard eigen-value problem
class standard_evp
{
    public:

        virtual ~standard_evp()
        {
        }

        virtual void solve(int32_t matrix_size, double_complex* a, int32_t lda, double* eval, double_complex* z, int32_t ldz)
        {
            error_local(__FILE__, __LINE__, "standard eigen-value solver is not configured");
        }

        virtual bool parallel() = 0;

        virtual ev_solver_t type() = 0;
};

/// Interface for LAPACK standard eigen-value solver
class standard_evp_lapack: public standard_evp
{
    private:
     
        std::vector<int32_t> get_work_sizes(int32_t matrix_size)
        {
            std::vector<int32_t> work_sizes(3);
            
            work_sizes[0] = 2 * matrix_size + matrix_size * matrix_size;
            work_sizes[1] = 1 + 5 * matrix_size + 2 * matrix_size * matrix_size;
            work_sizes[2] = 3 + 5 * matrix_size;
            return work_sizes;
        }

    public:
        
        standard_evp_lapack()
        {
        }
       
        void solve(int32_t matrix_size, double_complex* a, int32_t lda, double* eval, double_complex* z, int32_t ldz)
        {
            std::vector<int32_t> work_sizes = get_work_sizes(matrix_size);
            
            std::vector<double_complex> work(work_sizes[0]);
            std::vector<double> rwork(work_sizes[1]);
            std::vector<int32_t> iwork(work_sizes[2]);
            int32_t info;

            FORTRAN(zheevd)("V", "U", &matrix_size, a, &lda, eval, &work[0], &work_sizes[0], &rwork[0], &work_sizes[1], 
                            &iwork[0], &work_sizes[2], &info, (int32_t)1, (int32_t)1);
            
            for (int i = 0; i < matrix_size; i++)
                memcpy(&z[ldz * i], &a[lda * i], matrix_size * sizeof(double_complex));
            
            if (info)
            {
                std::stringstream s;
                s << "zheevd returned " << info; 
                error_local(__FILE__, __LINE__, s);
            }
        }

        bool parallel()
        {
            return false;
        }

        ev_solver_t type()
        {
            return ev_lapack;
        }
};

#ifdef _PLASMA_
extern "C" void plasma_zheevd_wrapper(int32_t matrix_size, void* a, int32_t lda, void* z,
                                      int32_t ldz, double* eval);
#endif

/// Interface for PLASMA standard eigen-value solver
class standard_evp_plasma: public standard_evp
{
    public:

        standard_evp_plasma()
        {
        }

        #ifdef _PLASMA_
        void solve(int32_t matrix_size, double_complex* a, int32_t lda, double* eval, double_complex* z, int32_t ldz)
        {
            //plasma_set_num_threads(1);
            //omp_set_num_threads(1);
            //printf("before call to plasma_zheevd_wrapper\n");
            plasma_zheevd_wrapper(matrix_size, a, lda, z, lda, eval);
            //printf("after call to plasma_zheevd_wrapper\n");
            //plasma_set_num_threads(8);
            //omp_set_num_threads(8);
        }
        #endif
        
        bool parallel()
        {
            return false;
        }

        ev_solver_t type()
        {
            return ev_plasma;
        }
};

/// Interface for ScaLAPACK standard eigen-value solver
class standard_evp_scalapack: public standard_evp
{
    private:

        int32_t block_size_;
        int num_ranks_row_;
        int num_ranks_col_;
        int blacs_context_;
        
        #ifdef _SCALAPACK_
        std::vector<int32_t> get_work_sizes(int32_t matrix_size, int32_t nb, int32_t nprow, int32_t npcol, 
                                            int blacs_context)
        {
            std::vector<int32_t> work_sizes(3);
            
            int32_t nn = std::max(matrix_size, std::max(nb, 2));
            
            int32_t np0 = linalg<scalapack>::numroc(nn, nb, 0, 0, nprow);
            int32_t mq0 = linalg<scalapack>::numroc(nn, nb, 0, 0, npcol);
        
            work_sizes[0] = matrix_size + (np0 + mq0 + nb) * nb;
        
            work_sizes[1] = 1 + 9 * matrix_size + 3 * np0 * mq0;
        
            work_sizes[2] = 7 * matrix_size + 8 * npcol + 2;
            
            return work_sizes;
        }
        #endif

    public:

        standard_evp_scalapack(int num_ranks_row__, int num_ranks_col__, int blacs_context__) 
            :
              num_ranks_row_(num_ranks_row__), 
              num_ranks_col_(num_ranks_col__), 
              blacs_context_(blacs_context__)
        {
            #ifdef _SCALAPACK_
            block_size_ = linalg<scalapack>::cyclic_block_size();
            #endif
        }

        #ifdef _SCALAPACK_
        void solve(int32_t matrix_size, double_complex* a, int32_t lda, double* eval, double_complex* z, int32_t ldz)
        {

            int desca[9];
            linalg<scalapack>::descinit(desca, matrix_size, matrix_size, block_size_, block_size_, 0, 0, 
                                        blacs_context_, lda);
            
            int descz[9];
            linalg<scalapack>::descinit(descz, matrix_size, matrix_size, block_size_, block_size_, 0, 0, 
                                        blacs_context_, ldz);
            
            std::vector<int32_t> work_sizes = get_work_sizes(matrix_size, block_size_, num_ranks_row_, num_ranks_col_, 
                                                             blacs_context_);
            
            std::vector<double_complex> work(work_sizes[0]);
            std::vector<double> rwork(work_sizes[1]);
            std::vector<int32_t> iwork(work_sizes[2]);
            int32_t info;

            int32_t ione = 1;
            FORTRAN(pzheevd)("V", "U", &matrix_size, a, &ione, &ione, desca, eval, z, &ione, &ione, descz, &work[0], 
                             &work_sizes[0], &rwork[0], &work_sizes[1], &iwork[0], &work_sizes[2], &info, (int32_t)1, 
                             (int32_t)1);

            if (info)
            {
                std::stringstream s;
                s << "pzheevd returned " << info; 
                error_local(__FILE__, __LINE__, s);
            }
        }
        #endif

        bool parallel()
        {
            return true;
        }

        ev_solver_t type()
        {
            return ev_scalapack;
        }
};

/// Base class for generalized eigen-value problem
class generalized_evp
{
    public:

        virtual ~generalized_evp()
        {
        }

        virtual void solve(int32_t matrix_size, int32_t num_rows_loc, int32_t num_cols_loc, int32_t nevec, 
                           double_complex* a, int32_t lda, double_complex* b, int32_t ldb, double* eval, 
                           double_complex* z, int32_t ldz)
        {
            error_local(__FILE__, __LINE__, "generalized eigen-value solver is not configured");
        }

        virtual bool parallel() = 0;

        virtual ev_solver_t type() = 0;
};

/// Interface for LAPACK generalized eigen-value solver
class generalized_evp_lapack: public generalized_evp
{
    private:

        double abstol_;
    
    public:

        generalized_evp_lapack(double abstol__) : abstol_(abstol__)
        {
        }

        void solve(int32_t matrix_size, int32_t num_rows_loc, int32_t num_cols_loc, int32_t nevec, 
                   double_complex* a, int32_t lda, double_complex* b, int32_t ldb, double* eval, 
                   double_complex* z, int32_t ldz)
        {
            assert(nevec <= matrix_size);

            int nb = linalg<lapack>::ilaenv(1, "ZHETRD", "U", matrix_size, 0, 0, 0);
            int lwork = (nb + 1) * matrix_size;
            int lrwork = 7 * matrix_size;
            int liwork = 5 * matrix_size;
            
            std::vector<double_complex> work(lwork);
            std::vector<double> rwork(lrwork);
            std::vector<int32_t> iwork(liwork);
            std::vector<int32_t> ifail(matrix_size);
            std::vector<double> w(matrix_size);
            double vl = 0.0;
            double vu = 0.0;
            int32_t m;
            int32_t info;
       
            int32_t ione = 1;
            FORTRAN(zhegvx)(&ione, "V", "I", "U", &matrix_size, a, &lda, b, &ldb, &vl, &vu, &ione, &nevec, &abstol_, &m, 
                            &w[0], z, &ldz, &work[0], &lwork, &rwork[0], &iwork[0], &ifail[0], &info, (int32_t)1, 
                            (int32_t)1, (int32_t)1);

            if (m != nevec) error_local(__FILE__, __LINE__, "Not all eigen-values are found.");

            if (info)
            {
                std::stringstream s;
                s << "zhegvx returned " << info; 
                error_local(__FILE__, __LINE__, s);
            }

            memcpy(eval, &w[0], nevec * sizeof(double));
        }

        bool parallel()
        {
            return false;
        }

        ev_solver_t type()
        {
            return ev_lapack;
        }
};

/// Interface for ScaLAPACK generalized eigen-value solver
class generalized_evp_scalapack: public generalized_evp
{
    private:

        int32_t block_size_;
        int num_ranks_row_;
        int num_ranks_col_;
        int blacs_context_;
        double abstol_;
        
        #ifdef _SCALAPACK_
        std::vector<int32_t> get_work_sizes(int32_t matrix_size, int32_t nb, int32_t nprow, int32_t npcol, 
                                            int blacs_context)
        {
            std::vector<int32_t> work_sizes(3);
            
            int32_t nn = std::max(matrix_size, std::max(nb, 2));
            
            int32_t neig = std::max(1024, nb);

            int32_t nmax3 = std::max(neig, std::max(nb, 2));
            
            int32_t np = nprow * npcol;

            // due to the mess in the documentation, take the maximum of np0, nq0, mq0
            int32_t nmpq0 = std::max(linalg<scalapack>::numroc(nn, nb, 0, 0, nprow), 
                                  std::max(linalg<scalapack>::numroc(nn, nb, 0, 0, npcol),
                                           linalg<scalapack>::numroc(nmax3, nb, 0, 0, npcol))); 

            int32_t anb = linalg<scalapack>::pjlaenv(blacs_context, 3, "PZHETTRD", "L", 0, 0, 0, 0);
            int32_t sqnpc = (int32_t)pow(double(np), 0.5);
            int32_t nps = std::max(linalg<scalapack>::numroc(nn, 1, 0, 0, sqnpc), 2 * anb);

            work_sizes[0] = matrix_size + (2 * nmpq0 + nb) * nb;
            work_sizes[0] = std::max(work_sizes[0], matrix_size + 2 * (anb + 1) * (4 * nps + 2) + (nps + 1) * nps);
            work_sizes[0] = std::max(work_sizes[0], 3 * nmpq0 * nb + nb * nb);

            work_sizes[1] = 4 * matrix_size + std::max(5 * matrix_size, nmpq0 * nmpq0) + 
                            linalg<scalapack>::iceil(neig, np) * nn + neig * matrix_size;

            int32_t nnp = std::max(matrix_size, std::max(np + 1, 4));
            work_sizes[2] = 6 * nnp;

            return work_sizes;
        }
        #endif
    
    public:

        generalized_evp_scalapack(int num_ranks_row__, int num_ranks_col__, int blacs_context__, 
                                  double abstol__) 
            : num_ranks_row_(num_ranks_row__), 
              num_ranks_col_(num_ranks_col__), 
              blacs_context_(blacs_context__),
              abstol_(abstol__)
        {
            #ifdef _SCALAPACK_
            block_size_ = linalg<scalapack>::cyclic_block_size();
            #endif
        }

        #ifdef _SCALAPACK_
        void solve(int32_t matrix_size, int32_t num_rows_loc, int32_t num_cols_loc, int32_t nevec, 
                   double_complex* a, int32_t lda, double_complex* b, int32_t ldb, double* eval, 
                   double_complex* z, int32_t ldz)
        {
            assert(nevec <= matrix_size);
            
            int32_t desca[9];
            linalg<scalapack>::descinit(desca, matrix_size, matrix_size, block_size_, block_size_, 0, 0, 
                                        blacs_context_, lda);

            int32_t descb[9];
            linalg<scalapack>::descinit(descb, matrix_size, matrix_size, block_size_, block_size_, 0, 0, 
                                        blacs_context_, ldb); 

            int32_t descz[9];
            linalg<scalapack>::descinit(descz, matrix_size, matrix_size, block_size_, block_size_, 0, 0, 
                                        blacs_context_, ldz); 

            std::vector<int32_t> work_sizes = get_work_sizes(matrix_size, block_size_, num_ranks_row_, num_ranks_col_, 
                                                             blacs_context_);
            
            std::vector<double_complex> work(work_sizes[0]);
            std::vector<double> rwork(work_sizes[1]);
            std::vector<int32_t> iwork(work_sizes[2]);
            
            std::vector<int32_t> ifail(matrix_size);
            std::vector<int32_t> iclustr(2 * num_ranks_row_ * num_ranks_col_);
            std::vector<double> gap(num_ranks_row_ * num_ranks_col_);
            std::vector<double> w(matrix_size);
            
            double orfac = 1e-6;
            int32_t ione = 1;
            
            int32_t m;
            int32_t nz;
            double d1;
            int32_t info;

            FORTRAN(pzhegvx)(&ione, "V", "I", "U", &matrix_size, a, &ione, &ione, desca, b, &ione, &ione, descb, &d1, &d1, 
                             &ione, &nevec, &abstol_, &m, &nz, &w[0], &orfac, z, &ione, &ione, descz, &work[0], &work_sizes[0], 
                             &rwork[0], &work_sizes[1], &iwork[0], &work_sizes[2], &ifail[0], &iclustr[0], &gap[0], &info, 
                             (int32_t)1, (int32_t)1, (int32_t)1); 

            if (info)
            {
                if ((info / 2) % 2)
                {
                    std::stringstream s;
                    s << "eigenvectors corresponding to one or more clusters of eigenvalues" << std::endl  
                      << "could not be reorthogonalized because of insufficient workspace" << std::endl;

                    int k = num_ranks_row_ * num_ranks_col_;
                    for (int i = 0; i < num_ranks_row_ * num_ranks_col_ - 1; i++)
                    {
                        if ((iclustr[2 * i + 1] != 0) && (iclustr[2 * (i + 1)] == 0))
                        {
                            k = i + 1;
                            break;
                        }
                    }
                   
                    s << "number of eigenvalue clusters : " << k << std::endl;
                    for (int i = 0; i < k; i++) s << iclustr[2 * i] << " : " << iclustr[2 * i + 1] << std::endl; 
                    error_local(__FILE__, __LINE__, s);
                }

                std::stringstream s;
                s << "pzhegvx returned " << info; 
                error_local(__FILE__, __LINE__, s);
            }

            if ((m != nevec) || (nz != nevec))
                error_local(__FILE__, __LINE__, "Not all eigen-vectors or eigen-values are found.");

            memcpy(eval, &w[0], nevec * sizeof(double));
        }
        #endif

        bool parallel()
        {
            return true;
        }

        ev_solver_t type()
        {
            return ev_scalapack;
        }
};

#ifdef _RS_GEN_EIG_
void my_gen_eig(char uplo, int n, int nev, double_complex* a, int ia, int ja, int* desca,
                double_complex* b, int ib, int jb, int* descb, double* d,
                double_complex* q, int iq, int jq, int* descq, int* info);

void my_gen_eig_cpu(char uplo, int n, int nev, double_complex* a, int ia, int ja, int* desca,
                    double_complex* b, int ib, int jb, int* descb, double* d,
                    double_complex* q, int iq, int jq, int* descq, int* info);
#endif

class generalized_evp_rs_gpu: public generalized_evp
{
    private:

        int32_t block_size_;
        int num_ranks_row_;
        int rank_row_;
        int num_ranks_col_;
        int rank_col_;
        int blacs_context_;
        
    public:

        generalized_evp_rs_gpu(int num_ranks_row__, int rank_row__, int num_ranks_col__, int rank_col__, 
                               int blacs_context__)
            : num_ranks_row_(num_ranks_row__),
              rank_row_(rank_row__),
              num_ranks_col_(num_ranks_col__), 
              rank_col_(rank_col__),
              blacs_context_(blacs_context__)
        {
            #ifdef _SCALAPACK_
            block_size_ = linalg<scalapack>::cyclic_block_size();
            #endif
        }

        #ifdef _RS_GEN_EIG_
        void solve(int32_t matrix_size, int32_t num_rows_loc, int32_t num_cols_loc, int32_t nevec, 
                   double_complex* a, int32_t lda, double_complex* b, int32_t ldb, double* eval, 
                   double_complex* z, int32_t ldz)
        {
        
            assert(nevec <= matrix_size);
            
            int32_t desca[9];
            linalg<scalapack>::descinit(desca, matrix_size, matrix_size, block_size_, block_size_, 0, 0, 
                                        blacs_context_, lda);

            int32_t descb[9];
            linalg<scalapack>::descinit(descb, matrix_size, matrix_size, block_size_, block_size_, 0, 0, 
                                        blacs_context_, ldb); 

            mdarray<double_complex, 2> ztmp(num_rows_loc, num_cols_loc);
            ztmp.pin_memory();
            int32_t descz[9];
            linalg<scalapack>::descinit(descz, matrix_size, matrix_size, block_size_, block_size_, 0, 0, 
                                        blacs_context_, num_rows_loc); 
            
            std::vector<double> eval_tmp(matrix_size);

            int info;
            my_gen_eig('L', matrix_size, nevec, a, 1, 1, desca, b, 1, 1, descb, &eval_tmp[0], ztmp.ptr(), 1, 1, descz, &info);
            if (info)
            {
                std::stringstream s;
                s << "my_gen_eig " << info; 
                error_local(__FILE__, __LINE__, s);
            }
            ztmp.unpin_memory();

            for (int i = 0; i < linalg<scalapack>::numroc(nevec, block_size_, rank_col_, 0, num_ranks_col_); i++)
                memcpy(&z[ldz * i], &ztmp(0, i), num_rows_loc * sizeof(double_complex));

            memcpy(eval, &eval_tmp[0], nevec * sizeof(double));
        }
        #endif

        bool parallel()
        {
            return true;
        }

        ev_solver_t type()
        {
            return ev_rs_gpu;
        }
};

class generalized_evp_rs_cpu: public generalized_evp
{
    private:

        int32_t block_size_;
        int num_ranks_row_;
        int rank_row_;
        int num_ranks_col_;
        int rank_col_;
        int blacs_context_;
        
    public:

        generalized_evp_rs_cpu(int num_ranks_row__, int rank_row__, int num_ranks_col__, int rank_col__, 
                               int blacs_context__)
            : num_ranks_row_(num_ranks_row__),
              rank_row_(rank_row__),
              num_ranks_col_(num_ranks_col__), 
              rank_col_(rank_col__),
              blacs_context_(blacs_context__)
        {
            #ifdef _SCALAPACK_
            block_size_ = linalg<scalapack>::cyclic_block_size();
            #endif
        }

        #ifdef _RS_GEN_EIG_
        void solve(int32_t matrix_size, int32_t num_rows_loc, int32_t num_cols_loc, int32_t nevec, 
                   double_complex* a, int32_t lda, double_complex* b, int32_t ldb, double* eval, 
                   double_complex* z, int32_t ldz)
        {
        
            assert(nevec <= matrix_size);
            
            int32_t desca[9];
            linalg<scalapack>::descinit(desca, matrix_size, matrix_size, block_size_, block_size_, 0, 0, 
                                        blacs_context_, lda);

            int32_t descb[9];
            linalg<scalapack>::descinit(descb, matrix_size, matrix_size, block_size_, block_size_, 0, 0, 
                                        blacs_context_, ldb); 

            mdarray<double_complex, 2> ztmp(num_rows_loc, num_cols_loc);
            int32_t descz[9];
            linalg<scalapack>::descinit(descz, matrix_size, matrix_size, block_size_, block_size_, 0, 0, 
                                        blacs_context_, num_rows_loc); 
            
            std::vector<double> eval_tmp(matrix_size);

            int info;
            my_gen_eig_cpu('L', matrix_size, nevec, a, 1, 1, desca, b, 1, 1, descb, &eval_tmp[0], ztmp.ptr(), 1, 1, descz, &info);
            if (info)
            {
                std::stringstream s;
                s << "my_gen_eig " << info; 
                error_local(__FILE__, __LINE__, s);
            }

            for (int i = 0; i < linalg<scalapack>::numroc(nevec, block_size_, rank_col_, 0, num_ranks_col_); i++)
                memcpy(&z[ldz * i], &ztmp(0, i), num_rows_loc * sizeof(double_complex));

            memcpy(eval, &eval_tmp[0], nevec * sizeof(double));
        }
        #endif

        bool parallel()
        {
            return true;
        }

        ev_solver_t type()
        {
            return ev_rs_cpu;
        }
};

/// Interface for ELPA single stage generalized eigen-value solver
class generalized_evp_elpa1: public generalized_evp
{
    private:
        
        int32_t block_size_;
        int32_t num_ranks_row_;
        int32_t rank_row_;
        int32_t num_ranks_col_;
        int32_t rank_col_;
        int blacs_context_;
        MPI_Comm comm_row_;
        MPI_Comm comm_col_;

    public:
        
        generalized_evp_elpa1(int32_t num_ranks_row__, int32_t rank_row__, int32_t num_ranks_col__, int32_t rank_col__, 
                              int blacs_context__, MPI_Comm comm_row__, MPI_Comm comm_col__) 
            : num_ranks_row_(num_ranks_row__), 
              rank_row_(rank_row__),
              num_ranks_col_(num_ranks_col__), 
              rank_col_(rank_col__),
              blacs_context_(blacs_context__), 
              comm_row_(comm_row__), 
              comm_col_(comm_col__)
        {
            #ifdef _SCALAPACK_
            block_size_ = linalg<scalapack>::cyclic_block_size();
            #endif
        }
        
        #ifdef _ELPA_
        void solve(int32_t matrix_size, int32_t num_rows_loc, int32_t num_cols_loc, int32_t nevec, 
                   double_complex* a, int32_t lda, double_complex* b, int32_t ldb, double* eval, 
                   double_complex* z, int32_t ldz)
        {

            assert(nevec <= matrix_size);

            int32_t mpi_comm_rows = MPI_Comm_c2f(comm_row_);
            int32_t mpi_comm_cols = MPI_Comm_c2f(comm_col_);

            sirius::Timer *t;

            t = new sirius::Timer("elpa::ort");
            FORTRAN(elpa_cholesky_complex)(&matrix_size, b, &ldb, &block_size_, &mpi_comm_rows, &mpi_comm_cols);
            FORTRAN(elpa_invert_trm_complex)(&matrix_size, b, &ldb, &block_size_, &mpi_comm_rows, &mpi_comm_cols);
       
            mdarray<double_complex, 2> tmp1(num_rows_loc, num_cols_loc);
            mdarray<double_complex, 2> tmp2(num_rows_loc, num_cols_loc);

            FORTRAN(elpa_mult_ah_b_complex)("U", "L", &matrix_size, &matrix_size, b, &ldb, a, &lda, &block_size_, 
                                            &mpi_comm_rows, &mpi_comm_cols, tmp1.ptr(), &num_rows_loc, (int32_t)1, 
                                            (int32_t)1);

            int32_t descc[9];
            linalg<scalapack>::descinit(descc, matrix_size, matrix_size, block_size_, block_size_, 0, 0, 
                                        blacs_context_, lda);

            linalg<scalapack>::pztranc(matrix_size, matrix_size, complex_one, tmp1.ptr(), 1, 1, descc, 
                                       complex_zero, tmp2.ptr(), 1, 1, descc);

            FORTRAN(elpa_mult_ah_b_complex)("U", "U", &matrix_size, &matrix_size, b, &ldb, tmp2.ptr(), &num_rows_loc, 
                                            &block_size_, &mpi_comm_rows, &mpi_comm_cols, a, &lda, (int32_t)1, 
                                            (int32_t)1);

            linalg<scalapack>::pztranc(matrix_size, matrix_size, complex_one, a, 1, 1, descc, complex_zero, 
                                       tmp1.ptr(), 1, 1, descc);

            for (int i = 0; i < num_cols_loc; i++)
            {
                int32_t n_col = linalg<scalapack>::indxl2g(i + 1, block_size_, rank_col_, 0, num_ranks_col_);
                int32_t n_row = linalg<scalapack>::numroc(n_col, block_size_, rank_row_, 0, num_ranks_row_);
                for (int j = n_row; j < num_rows_loc; j++) 
                {
                    assert(j < num_rows_loc);
                    assert(i < num_cols_loc);
                    a[j + i * lda] = tmp1(j, i);
                }
            }
            delete t;
            
            t = new sirius::Timer("elpa::diag");
            std::vector<double> w(matrix_size);
            FORTRAN(elpa_solve_evp_complex)(&matrix_size, &nevec, a, &lda, &w[0], tmp1.ptr(), &num_rows_loc, 
                                            &block_size_, &mpi_comm_rows, &mpi_comm_cols);
            delete t;

            t = new sirius::Timer("elpa::bt");
            linalg<scalapack>::pztranc(matrix_size, matrix_size, complex_one, b, 1, 1, descc, complex_zero, 
                                       tmp2.ptr(), 1, 1, descc);

            FORTRAN(elpa_mult_ah_b_complex)("L", "N", &matrix_size, &nevec, tmp2.ptr(), &num_rows_loc, tmp1.ptr(), 
                                            &num_rows_loc, &block_size_, &mpi_comm_rows, &mpi_comm_cols, z, &ldz, 
                                            (int32_t)1, (int32_t)1);
            delete t;

            memcpy(eval, &w[0], nevec * sizeof(double));
        }
        #endif

        bool parallel()
        {
            return true;
        }

        ev_solver_t type()
        {
            return ev_elpa1;
        }
};

/// Interface for ELPA 2-stage generalized eigen-value solver
class generalized_evp_elpa2: public generalized_evp
{
    private:
        
        int32_t block_size_;
        int32_t num_ranks_row_;
        int32_t rank_row_;
        int32_t num_ranks_col_;
        int32_t rank_col_;
        int blacs_context_;
        MPI_Comm comm_row_;
        MPI_Comm comm_col_;
        MPI_Comm comm_all_;

    public:
        
        generalized_evp_elpa2(int32_t num_ranks_row__, int32_t rank_row__, int32_t num_ranks_col__, int32_t rank_col__, 
                              int blacs_context__, MPI_Comm comm_row__, MPI_Comm comm_col__, MPI_Comm comm_all__) 
            : num_ranks_row_(num_ranks_row__), 
              rank_row_(rank_row__),
              num_ranks_col_(num_ranks_col__), 
              rank_col_(rank_col__),
              blacs_context_(blacs_context__), 
              comm_row_(comm_row__), 
              comm_col_(comm_col__), 
              comm_all_(comm_all__)
        {
            #ifdef _SCALAPACK_
            block_size_ = linalg<scalapack>::cyclic_block_size();
            #endif
        }
        
        #ifdef _ELPA_
        void solve(int32_t matrix_size, int32_t num_rows_loc, int32_t num_cols_loc, int32_t nevec, 
                   double_complex* a, int32_t lda, double_complex* b, int32_t ldb, double* eval, 
                   double_complex* z, int32_t ldz)
        {

            assert(nevec <= matrix_size);

            int32_t mpi_comm_rows = MPI_Comm_c2f(comm_row_);
            int32_t mpi_comm_cols = MPI_Comm_c2f(comm_col_);
            int32_t mpi_comm_all = MPI_Comm_c2f(comm_all_);

            sirius::Timer *t;

            t = new sirius::Timer("elpa::ort");
            FORTRAN(elpa_cholesky_complex)(&matrix_size, b, &ldb, &block_size_, &mpi_comm_rows, &mpi_comm_cols);
            FORTRAN(elpa_invert_trm_complex)(&matrix_size, b, &ldb, &block_size_, &mpi_comm_rows, &mpi_comm_cols);
       
            mdarray<double_complex, 2> tmp1(num_rows_loc, num_cols_loc);
            mdarray<double_complex, 2> tmp2(num_rows_loc, num_cols_loc);

            FORTRAN(elpa_mult_ah_b_complex)("U", "L", &matrix_size, &matrix_size, b, &ldb, a, &lda, &block_size_, 
                                            &mpi_comm_rows, &mpi_comm_cols, tmp1.ptr(), &num_rows_loc, (int32_t)1, 
                                            (int32_t)1);

            int32_t descc[9];
            linalg<scalapack>::descinit(descc, matrix_size, matrix_size, block_size_, block_size_, 0, 0, 
                                        blacs_context_, lda);

            linalg<scalapack>::pztranc(matrix_size, matrix_size, complex_one, tmp1.ptr(), 1, 1, descc, 
                                       complex_zero, tmp2.ptr(), 1, 1, descc);

            FORTRAN(elpa_mult_ah_b_complex)("U", "U", &matrix_size, &matrix_size, b, &ldb, tmp2.ptr(), &num_rows_loc, 
                                            &block_size_, &mpi_comm_rows, &mpi_comm_cols, a, &lda, (int32_t)1, 
                                            (int32_t)1);

            linalg<scalapack>::pztranc(matrix_size, matrix_size, complex_one, a, 1, 1, descc, complex_zero, 
                                       tmp1.ptr(), 1, 1, descc);

            for (int i = 0; i < num_cols_loc; i++)
            {
                int32_t n_col = linalg<scalapack>::indxl2g(i + 1, block_size_, rank_col_, 0, num_ranks_col_);
                int32_t n_row = linalg<scalapack>::numroc(n_col, block_size_, rank_row_, 0, num_ranks_row_);
                for (int j = n_row; j < num_rows_loc; j++) 
                {
                    assert(j < num_rows_loc);
                    assert(i < num_cols_loc);
                    a[j + i * lda] = tmp1(j, i);
                }
            }
            delete t;
            
            t = new sirius::Timer("elpa::diag");
            std::vector<double> w(matrix_size);
            FORTRAN(elpa_solve_evp_complex_2stage)(&matrix_size, &nevec, a, &lda, &w[0], tmp1.ptr(), &num_rows_loc, 
                                                   &block_size_, &mpi_comm_rows, &mpi_comm_cols, &mpi_comm_all);
            delete t;

            t = new sirius::Timer("elpa::bt");
            linalg<scalapack>::pztranc(matrix_size, matrix_size, complex_one, b, 1, 1, descc, complex_zero, 
                                       tmp2.ptr(), 1, 1, descc);

            FORTRAN(elpa_mult_ah_b_complex)("L", "N", &matrix_size, &nevec, tmp2.ptr(), &num_rows_loc, tmp1.ptr(), 
                                            &num_rows_loc, &block_size_, &mpi_comm_rows, &mpi_comm_cols, z, &ldz, 
                                            (int32_t)1, (int32_t)1);
            delete t;

            memcpy(eval, &w[0], nevec * sizeof(double));
        }
        #endif

        bool parallel()
        {
            return true;
        }

        ev_solver_t type()
        {
            return ev_elpa2;
        }
};

/// Interface for MAGMA generalized eigen-value solver
class generalized_evp_magma: public generalized_evp
{
    private:

    public:
        generalized_evp_magma()
        {
        }

        #ifdef _MAGMA_
        void solve(int32_t matrix_size, int32_t num_rows_loc, int32_t num_cols_loc, int32_t nevec, 
                   double_complex* a, int32_t lda, double_complex* b, int32_t ldb, double* eval, 
                   double_complex* z, int32_t ldz)
        {
            assert(nevec <= matrix_size);
            
            magma_zhegvdx_2stage_wrapper(matrix_size, nevec, a, lda, b, ldb, eval);
            
            for (int i = 0; i < nevec; i++) memcpy(&z[ldz * i], &a[lda * i], matrix_size * sizeof(double_complex));
        }
        #endif

        bool parallel()
        {
            return false;
        }

        ev_solver_t type()
        {
            return ev_magma;
        }
};
