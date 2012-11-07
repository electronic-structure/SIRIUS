namespace sirius
{
// TODO: who prints the output? root of the grid or rank=0
//       rank=0 may not be in grid, grid may not exist
class MPIGrid
{
    private:
        
        MPIWorld& mpi_world_;

        /// dimensions of the grid
        std::vector<int> dimensions_;

        /// coordinates of the MPI rank in the grid
        std::vector<int> coordinates_;

        /// total number of communicators
        int num_comm_;

        /// base grid communicator
        MPI_Comm base_grid_communicator_;

        /// grid communicators

        /** Grid comminicators are built for all possible combinations of directions, i.e. 000, 001, 010, 011, etc.
            First communicator is the trivial "self" communicator; the last comminicator handles the entire grid.
        */
        std::vector<MPI_Comm> communicators_;

        /// number of MPI ranks in each communicator
        std::vector<int> communicator_size_;

        /// true if this is the root of the communicator group
        std::vector<bool> communicator_root_;

    public:
        
        MPIGrid(MPIWorld& mpi_world__) : mpi_world_(mpi_world__)
        {
        }

        void initialize(const std::vector<int> dimensions__)
        {
            dimensions_ = dimensions__;

            if (mpi_world_.size() < size())
            {
                std::stringstream s;
                s << "Not enough processors to build a grid";
                error(__FILE__, __LINE__, s);
            }

            num_comm_ = pow(2, dimensions_.size());

            base_grid_communicator_ = MPI_COMM_NULL;

            communicators_ = std::vector<MPI_Comm>(num_comm_, MPI_COMM_NULL);

            coordinates_ = std::vector<int>(dimensions_.size(), -1);

            communicator_size_ = std::vector<int>(num_comm_, 0);

            communicator_root_ = std::vector<bool>(num_comm_, false);

            std::vector<int> periods(dimensions_.size(), 0);
            // communicator of the entire grid
            MPI_Cart_create(MPI_COMM_WORLD, dimensions_.size(), &dimensions_[0], &periods[0], 0, 
                            &base_grid_communicator_);

        }

        int size()
        {
            int sz = 1;
            for (int i = 0; i < (int)dimensions_.size(); i++)
                sz *= dimensions_[i];
            return sz;
        }

        int size(int i)
        {
            return dimensions_[i];
        }

        bool in_grid()
        {
            return (base_grid_communicator_ != MPI_COMM_NULL);
        }
};

};
