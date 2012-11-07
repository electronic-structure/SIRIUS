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
        
        void initialize(int init_mpi)
        {
            if (init_mpi) MPI_Init(NULL, NULL);
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
};

};
