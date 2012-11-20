class MPIWorld
{
    private:

        /// total number of MPI ranks
        static int size_;

        /// current MPI rank
        static int rank_;

        /// true if this rank is allowed to print to the standard output
        static bool verbose_;

    public:
        
        static void initialize()
        {
            if (call_mpi_init) MPI_Init(NULL, NULL);
            MPI_Comm_size(MPI_COMM_WORLD, &size_);
            MPI_Comm_rank(MPI_COMM_WORLD, &rank_);

            // rank 0 prints by default
            verbose_ = false;
            if (rank_ == 0) verbose_ = true;
        }

        static int size()
        {
            return size_;
        }

        static int rank()
        {
            return rank_;
        }
        
        static void abort()
        {
            MPI_Abort(MPI_COMM_WORLD, -1);
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

int MPIWorld::size_;
int MPIWorld::rank_;
bool MPIWorld::verbose_;

