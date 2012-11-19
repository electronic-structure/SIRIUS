
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

        MPI_Comm communicator_;

    public:

        splindex() : index_size_(-1),
                     num_ranks_(-1),
                     rank_id_(-1),
                     local_index_size_(-1),
                     global_index_offset_(-1),
                     communicator_(MPI_COMM_NULL)
        {
        }
        
        splindex(int index_size__,
                 const std::vector<int>& dimensions__, 
                 const std::vector<int>& coordinates__,
                 const MPI_Comm communicator__ = MPI_COMM_WORLD) : index_size_(index_size__),
                                                                   dimensions_(dimensions__),
                                                                   coordinates_(coordinates__),
                                                                   communicator_(communicator__)
        {
            if (dimensions__.size() == 0)
                error(__FILE__, __LINE__, "empty array of dimensions", fatal_err);

            if (dimensions__.size() != coordinates__.size())
                error(__FILE__, __LINE__, "sizes don't match", fatal_err);

            for (int i = 0; i < (int)dimensions_.size(); i++)
            {
                if ((coordinates_[i] < 0) || (coordinates_[i] >= dimensions_[i]))
                {
                    std::stringstream s;
                    s << "bad coordinates" << std::endl
                      << "  direction : " << i << std::endl
                      << "  coordinate : " << coordinates_[i] << std::endl
                      << "  dimension size : " << dimensions_[i];
                    error(__FILE__, __LINE__, s, fatal_err);
                }
            }

            if (index_size_ == 0)
                error(__FILE__, __LINE__, "need to think what to do with zero index size", fatal_err);

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
            
            rank_id_ = coordinates_[0];
            for (int i = 1; i < (int)dimensions_.size(); i++) 
                rank_id_ += offset_[i] * coordinates_[i];

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

        /*inline int size()
        {
            return local_index_size_;
        }*/

        inline int global_index(int idx_loc)
        {
            return (global_index_offset_ + idx_loc);
        }

        inline MPI_Comm& communicator()
        {
            return communicator_;
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

        /// rank (in the MPI_COMM_WORLD) of the grid root
        int world_root_;

        /// return valid directions for the current grid dimensionality
        inline int valid_directions(int directions)
        {
            return (directions & ((1 << dimensions_.size()) - 1));
        }

    public:
        
        void initialize(const std::vector<int> dimensions__)
        {
            dimensions_ = dimensions__;

            int sz = 1;
            for (int i = 0; i < (int)dimensions_.size(); i++)
                sz *= dimensions_[i];
            
            if (Platform::num_mpi_ranks() < sz)
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

            std::vector<int> v(Platform::num_mpi_ranks(), 0);
            if (in_grid() && root()) v[Platform::mpi_rank()] = 1;
            Platform::allreduce(&v[0], Platform::num_mpi_ranks(), MPI_COMM_WORLD);
            for (int i = 0; i < Platform::num_mpi_ranks(); i++)
                if (v[i] == 1) world_root_ = i;
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

        inline int size(int directions = 0xFF)
        {
            return communicator_size_[valid_directions(directions)];
        }

        inline bool in_grid()
        {
            return (base_grid_communicator_ != MPI_COMM_NULL);
        }

        inline bool root(int directions = 0xFF)
        {
            return communicator_root_[valid_directions(directions)];
        }

        /// Check if MPI ranks are at the side of the grid

        /** Side ranks are those for which remaining coordinates are zero.
        */
        inline bool side(int directions)
        {
            if (!in_grid()) return false;

            bool flg = true; 

            for (int i = 0; i < (int)dimensions_.size(); i++) 
                if (!(directions & (1 << i)) && coordinates_[i]) flg = false;

            return flg;
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

        inline splindex split_index(int directions, int size)
        {
            if (!in_grid())
                return splindex();

            MPI_Comm comm = communicators_[valid_directions(directions)];

            return splindex(size, sub_dimensions(directions), sub_coordinates(directions), comm); 
        }

        int cart_rank(const MPI_Comm& comm, std::vector<int>& coords)
        {
            int r;

            MPI_Cart_rank(comm, &coords[0], &r);

            return r;
        }

        template <typename T>
        void reduce(T* buffer, int count, int directions, bool side_only = false)
        {
            if (!in_grid()) return;
            
            if (side_only && (!side(directions))) return;

            std::vector<int> root_coords;

            for (int i = 0; i < (int)coordinates_.size(); i++)
                if (directions & (1 << i))
                    root_coords.push_back(0);
            
            MPI_Comm comm = communicators_[valid_directions(directions)];

            Platform::reduce(buffer, count, comm, cart_rank(comm, root_coords));

            //MPI_Reduce(MPI_IN_PLACE, buffer, count, primitive_type_wrapper<T>::mpi_type_id(), MPI_SUM, root_rank, 
            //           comm);
        }

        void barrier(int directions)
        {
           Platform::barrier(communicators_[valid_directions(directions)]);
        }

        inline int world_root()
        {
            return world_root_;
        }
};
