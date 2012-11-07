namespace sirius
{

class MPIWorld
{
    private:

        /// total number of MPI ranks
        int size_;

        /// curent MPI rank
        int rank_;

    public:
        
        void initialize()
        {
            if (call_mpi_init) MPI_Init(NULL, NULL);
            MPI_Comm_size(MPI_COMM_WORLD, &size_);
            MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        }

        int size()
        {
            return size_;
        }

        int rank()
        {
            return rank_;
        }
        
        void abort()
        {
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
};

MPIWorld mpi_world;

};
