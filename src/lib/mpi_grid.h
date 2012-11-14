
class splindex
{
    private:

        int index_size_;
        
        std::vector<int> dimensions_;

        std::vector<int> coordinates_;

        std::vector<int> offset_;

        int num_ranks_;

        int rank_id_;

        int local_index_size_;

        int global_index_offset_;

    public:

        splindex() : index_size_(-1),
                     num_ranks_(-1),
                     rank_id_(-1),
                     local_index_size_(-1),
                     global_index_offset_(-1)
        {
        }
        
        splindex(int index_size__,
                 const std::vector<int>& dimensions__, 
                 const std::vector<int>& coordinates__) : index_size_(index_size__),
                                                          dimensions_(dimensions__),
                                                          coordinates_(coordinates__)
        {
            if (dimensions__.size() == 0)
                error(__FILE__, __LINE__, "empty array of dimensions");

            if (dimensions__.size() != coordinates__.size())
                error(__FILE__, __LINE__, "empty sizes don't match");

            for (int i = 0; i < (int)dimensions_.size(); i++)
                if (coordinates_[i] >= dimensions_[i])
                {
                    std::stringstream s;
                    s << "bad coordinates" << std::endl
                      << "  direction : " << i << std::endl
                      << "  coordinate : " << coordinates_[i] << std::endl
                      << "  dimension size : " << dimensions_[i];
                    error(__FILE__, __LINE__, s);
                }

            num_ranks_ = 1;
            for (int i = 0; i < (int)dimensions_.size(); i++)
                num_ranks_ *= dimensions_[i];

            offset_ = std::vector<int>(dimensions_.size(), 0);
            int n = 1;
            for (int i = 1; i < (int)dimensions_.size(); i++) 
            {
                n *= dimensions_[i - 1];
                offset_[i] = n;
            }

            bool out_of_grid = false;
            for (int i = 0; i < (int)dimensions_.size(); i++)
                if (coordinates_[i] < 0) out_of_grid = true;
            
            if (out_of_grid)
            {
                rank_id_ = -1;
                local_index_size_ = -1;
                global_index_offset_ = -1;
            }
            else
            {
                rank_id_ = coordinates_[0];
                for (int i = 1; i < (int)dimensions_.size(); i++) 
                    rank_id_ += offset_[i] * coordinates_[i];
                
                if (index_size_ == 0)
                    error(__FILE__, __LINE__, "need to think what to do with zero index size");

                // minimum size
                int n1 = index_size_ / num_ranks_;

                // first n2 ranks have one more index element
                int n2 = index_size_ % num_ranks_;

                if (rank_id_ < n2)
                {
                    local_index_size_ = n1 + 1;
                    global_index_offset_ = (n1 + 1) * rank_id_;
                }
                else
                {   
                    local_index_size_ = n1;
                    global_index_offset_ = (n1 > 0) ? (n1 + 1) * n2 + n1 * (rank_id_ - n2) : -1;
                }
            }
        }

        inline int begin()
        {
            return global_index_offset_;
        }

        inline int end()
        {
            return (global_index_offset_ + local_index_size_ - 1);
        }

        inline int local_index_size()
        {
            return local_index_size_;
        }
};


class MPIGrid
{
    private:
        
        /// dimensions of the grid
        std::vector<int> dimensions_;

        /// coordinates of the MPI rank in the grid
        std::vector<int> coordinates_;

        /// base grid communicator
        MPI_Comm base_grid_communicator_;

        /// grid communicators

        /** Grid comminicators are built for all possible combinations of 
            directions, i.e. 001, 010, 011, etc. First communicator is the 
            trivial "self" communicator (it is not created because it's not 
            used); the last communicator handles the entire grid.
        */
        std::vector<MPI_Comm> communicators_;

        /// number of MPI ranks in each communicator
        std::vector<int> communicator_size_;

        /// true if this is the root of the communicator group
        std::vector<bool> communicator_root_;

        /// return valid directions for the current grid dimensionality
        inline int valid_directions(int directions)
        {
            return (directions & ((1 << dimensions_.size()) - 1));
        }

    public:
        
        void initialize(const std::vector<int> dimensions__)
        {
            dimensions_ = dimensions__;

            if (Platform::num_mpi_ranks() < size())
            {
                std::stringstream s;
                s << "Not enough processors to build a grid";
                error(__FILE__, __LINE__, s);
            }

            // communicator of the entire grid
            base_grid_communicator_ = MPI_COMM_NULL;
            std::vector<int> periods(dimensions_.size(), 0);
            MPI_Cart_create(MPI_COMM_WORLD, (int)dimensions_.size(), &dimensions_[0], &periods[0], 0, 
                            &base_grid_communicator_);

            if (in_grid()) 
            {
                // total number of communicators
                int num_comm = 1 << dimensions_.size();

                communicators_ = std::vector<MPI_Comm>(num_comm, MPI_COMM_NULL);

                coordinates_ = std::vector<int>(dimensions_.size(), -1);

                communicator_size_ = std::vector<int>(num_comm, 0);

                communicator_root_ = std::vector<bool>(num_comm, false);

                // get coordinates
                MPI_Cart_get(base_grid_communicator_, (int)dimensions_.size(), &dimensions_[0], &periods[0], 
                             &coordinates_[0]);

                // get all possible communicators
                for (int i = 1; i < num_comm; i++) 
                {
                    bool is_root = true;
                    int comm_size = 1;
                    std::vector<int> flg(dimensions_.size(), 0);

                    // each bit represents a directions
                    for (int j = 0; j < (int)dimensions_.size(); j++) 
                    {
                        if (i & (1<<j)) 
                        {
                            flg[j] = 1;
                            is_root = is_root && (coordinates_[j] == 0);
                            comm_size *= dimensions_[j];
                        }
                    }

                    communicator_root_[i] = is_root;

                    communicator_size_[i] = comm_size;

                    // subcommunicators
                    MPI_Cart_sub(base_grid_communicator_, &flg[0], &communicators_[i]);
                }
                
                // root of the grig can print
                Platform::set_verbose(root());

                // double check the size of communicators
                for (int i = 1; i < num_comm; i++)
                {
                    int comm_size;
                    MPI_Comm_size(communicators_[i], &comm_size);

                    if (comm_size != communicator_size_[i]) 
                        error(__FILE__, __LINE__, "Communicator sizes don't match");
                }
            }
        }

        void finalize()
        {
            for (int i = 1; i < (int)communicators_.size(); i++)
                MPI_Comm_free(&communicators_[i]);

            if (in_grid())
                MPI_Comm_free(&base_grid_communicator_);

            communicators_.clear();
            communicator_root_.clear();
            communicator_size_.clear();
            coordinates_.clear();
            dimensions_.clear();
        }

        inline int size()
        {
            int sz = 1;
            for (int i = 0; i < (int)dimensions_.size(); i++)
                sz *= dimensions_[i];
            return sz;
        }

        inline int size(int i)
        {
            return dimensions_[i];
        }

        inline bool in_grid()
        {
            return (base_grid_communicator_ != MPI_COMM_NULL);
        }

        inline bool root(int directions = 0xFF)
        {
            return communicator_root_[valid_directions(directions)];
        }

        std::vector<int> sub_dimensions(int directions)
        {
            std::vector<int> sd;
            for (int i = 0; i < (int)dimensions_.size(); i++)
                if (directions & (1 << i))
                    sd.push_back(dimensions_[i]);
            return sd;
        }

        std::vector<int> sub_coordinates(int directions)
        {
            std::vector<int> sc;
            for (int i = 0; i < (int)coordinates_.size(); i++)
                if (directions & (1 << i))
                    sc.push_back(coordinates_[i]);
            return sc;
        }

        inline splindex split_index(int directions, int N)
        {
            int valid_dir = valid_directions(directions);
            return splindex(N, sub_dimensions(valid_dir), sub_coordinates(valid_dir)); 

        }
};
