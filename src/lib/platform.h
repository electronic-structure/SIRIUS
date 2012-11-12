class Platform
{
    private:

        static bool verbose_;
    
    public:

        static void initialize()
        {
            if (call_mpi_init) MPI_Init(NULL, NULL);
            //MPI_Comm_size(MPI_COMM_WORLD, &size_);
            //MPI_Comm_rank(MPI_COMM_WORLD, &rank_);

            // rank 0 prints by default
            verbose_ = false;
            if (mpi_rank() == 0) verbose_ = true;
        }

        static int mpi_rank()
        {
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            return rank;
        }

        static int num_mpi_ranks()
        {
            int size;
            MPI_Comm_size(MPI_COMM_WORLD, &size);
            return size;
        }

        static void abort()
        {
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        static int num_threads()
        {
            return omp_get_max_threads();
        }

        static int thread_id()
        {
            return omp_get_thread_num();
        }
        
        static bool verbose()
        {
            return verbose_;
        }

        static void set_verbose(bool verbose__)
        {
            verbose_ = verbose__;
        }
};

bool Platform::verbose_ = false;

