class MPIGrid
{
    private:
        
        /// dimensions of the grid
        std::vector<int> dimensions_;

        /// coordinates of the MPI rank in the grid
        std::vector<int> coordinates_;

        /// parent communicator
        MPI_Comm parent_communicator_;

        /// grid communicator of the enrire grid returned by MPI_Cart_create
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
        //int world_root_;

        /// return valid directions for the current grid dimensionality
        inline int valid_directions(int directions)
        {
            return (directions & ((1 << dimensions_.size()) - 1));
        }

        // forbid copy constructor
        MPIGrid(const MPIGrid& src);

        // forbid assignment operator
        MPIGrid& operator=(const MPIGrid& src);

    public:

        // default constructor
        MPIGrid() : parent_communicator_(MPI_COMM_WORLD),
                    base_grid_communicator_(MPI_COMM_NULL) 
        {
        }

        MPIGrid(const std::vector<int> dimensions__, 
                MPI_Comm parent_communicator__) : dimensions_(dimensions__),
                                                  parent_communicator_(parent_communicator__),
                                                  base_grid_communicator_(MPI_COMM_NULL)
        {
            initialize();
        }

        ~MPIGrid()
        {
            finalize();
        }

        void initialize(const std::vector<int> dimensions__)
        {
            dimensions_ = dimensions__;
            initialize();
        }
        
        /// Initialize the grid.
        void initialize()
        {
            if (dimensions_.size() == 0)
                error(__FILE__, __LINE__, "no dimensions for the grid", fatal_err);

            int sz = 1;
            for (int i = 0; i < (int)dimensions_.size(); i++)
                sz *= dimensions_[i];
            
            if (sz == 0) error(__FILE__, __LINE__, "One of the MPI grid dimensions has a zero length.", fatal_err);

            if (Platform::num_mpi_ranks(parent_communicator_) < sz)
            {
                std::stringstream s;
                s << "Not enough processors to build a grid." << std::endl
                  << "  grid dimensions :";
                for (int i = 0; i < (int)dimensions_.size(); i++) s << " " << dimensions_[i];
                s << std::endl
                  << "  available number of MPI ranks : " << Platform::num_mpi_ranks(parent_communicator_);

                error(__FILE__, __LINE__, s, fatal_err);
            }

            // communicator of the entire grid
            std::vector<int> periods(dimensions_.size(), 0);
            MPI_Cart_create(parent_communicator_, (int)dimensions_.size(), &dimensions_[0], &periods[0], 0, 
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
                //Platform::set_verbose(root());

                // double check the size of communicators
                for (int i = 1; i < num_comm; i++)
                    if (Platform :: num_mpi_ranks(communicators_[i]) != communicator_size_[i]) 
                        error(__FILE__, __LINE__, "Communicator sizes don't match");
            }

            //if (base_comm == MPI_COMM_WORLD)
            //{
            //    std::vector<int> v(Platform::num_mpi_ranks(), 0);
            //    if (in_grid() && root()) v[Platform::mpi_rank()] = 1;
            //    Platform::allreduce(&v[0], Platform::num_mpi_ranks(), MPI_COMM_WORLD);
            //    for (int i = 0; i < Platform::num_mpi_ranks(); i++)
            //        if (v[i] == 1) world_root_ = i;
            //}
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

        inline std::vector<int> sub_dimensions(int directions)
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

        std::vector<int> coordinates()
        {
            return coordinates_;
        }

        int coordinate(int idim)
        {
            assert(idim < (int)coordinates_.size());

            return coordinates_[idim];
        }
        
        std::vector<int> dimensions()
        {
            return dimensions_;
        }

        MPIGrid* sub_grid(int directions)
        {
            if (valid_directions(directions))
            {
                //std::vector<int> sd = sub_dimensions(directions);
                //MPI_Comm comm = communicator(directions);
                return new MPIGrid(sub_dimensions(directions), communicator(directions));
                //return new MPIGrid(sd, comm);
            }
            else
            {
                return NULL;
            }
        }

        int cart_rank(const MPI_Comm& comm, std::vector<int> coords)
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
        }

        void barrier(int directions = 0xFF)
        {
           Platform::barrier(communicators_[valid_directions(directions)]);
        }

        //inline int world_root()
        //{
        //    return world_root_;
        //}

        inline MPI_Comm& communicator(int directions = 0xFF)
        {
            assert(communicators_.size() != 0);

            return communicators_[valid_directions(directions)];
        }
};
