template <splindex_t type, int block_size__ = 1>
class splindex
{
    private:

        int rank_;

        int num_ranks_;

        //int index_size_;
        //
        //std::vector<int> dimensions_;

        //std::vector<int> coordinates_;

        //std::vector<int> offset_;

        //int num_ranks_;

        //int rank_id_;

        //int local_size_;

        //int global_index_offset_;

        //MPI_Comm communicator_;

        /// local index size for each rank
        std::vector<int> local_size_;
        
        /// global index by rank and local index
        mdarray<int, 2> global_index_;

        /// location (local index and rank) of global index
        mdarray<int, 2> location_;

        /// forbid copy constructor
        splindex(const splindex& src);

        /// forbid assigment operator
        splindex& operator=(const splindex& src);

    public:
        
        splindex() : rank_(-1), num_ranks_(-1)
        {
        }
        
        splindex(int global_index_size__, int num_ranks__, int rank__)
        {
            split(global_index_size__, num_ranks__, rank__); 
        }
        
        void split(int global_index_size__, int num_ranks__, int rank__)
        {
            assert(num_ranks__ > 0);
            assert((rank__ >= 0) && (rank__ < num_ranks__));

            rank_ = rank__;
            num_ranks_ = num_ranks__;
            
            local_size_.resize(num_ranks__);

            if (type == block)
            {
                // minimum size
                int n1 = global_index_size__ / num_ranks__;

                // first n2 ranks have one more index element
                int n2 = global_index_size__ % num_ranks__;
                
                global_index_.set_dimensions(n1 + std::min(1, n2), num_ranks__);
                global_index_.allocate();
                for (int i1 = 0; i1 < global_index_.size(1); i1++)
                    for (int i0 = 0; i0 < global_index_.size(0); i0++)
                        global_index_(i0, i1) = -1;

                for (int i1 = 0; i1 < num_ranks__; i1++)
                {
                    int global_index_offset;
                    if (i1 < n2)
                    {
                        local_size_[i1] = n1 + 1; 
                        global_index_offset = (n1 + 1) * i1;
                    }
                    else
                    {
                        local_size_[i1] = n1; 
                        global_index_offset = (n1 > 0) ? (n1 + 1) * n2 + n1 * (i1 - n2) : -1;
                    }
                    for (int i0 = 0; i0 < local_size_[i1]; i0++)
                        global_index_(i0, i1) = global_index_offset + i0;
                }
            }

            if (type == block_cyclic)
            {

                int nblocks = (global_index_size__ / block_size__) +           // number of full blocks
                              std::min(1, global_index_size__ % block_size__); // extra partial block

                int max_size = ((nblocks / num_ranks__) +            // minimum number of blocks per rank
                                std::min(1, nblocks % num_ranks__)); // some ranks get extra block
                max_size *= block_size__;
            
                global_index_.set_dimensions(max_size, num_ranks__);
                global_index_.allocate();
            
                int irank = 0;
                int iblock = 0;
                std::vector< std::vector<int> > iv(num_ranks__);
                for (int i = 0; i < global_index_size__; i++)
                {
                    iv[irank].push_back(i);
                    if ((++iblock) == block_size__)
                    {
                        iblock = 0;
                        irank = (irank + 1) % num_ranks__;
                    }
                }
           
                for (int i1 = 0; i1 < global_index_.size(1); i1++)
                {
                    local_size_[i1] = (int)iv[i1].size();
                    for (int i0 = 0; i0 < global_index_.size(0); i0++)
                        global_index_(i0, i1) = (i0 < (int)iv[i1].size()) ? iv[i1][i0] : -1;
                }
            }

            location_.set_dimensions(2, global_index_size__);
            location_.allocate();

            for (int i1 = 0; i1 < global_index_.size(1); i1++)
                for (int i0 = 0; i0 < global_index_.size(0); i0++)
                {
                    int j = global_index_(i0, i1);
                    if (j >= 0)
                    {
                        location_(0, j) = i0;
                        location_(1, j) = i1;
                    }
                }
        }
        

        //splindex() : index_size_(-1),
        //             num_ranks_(-1),
        //             rank_id_(-1),
        //             local_size_(-1),
        //             global_index_offset_(-1),
        //             communicator_(MPI_COMM_NULL)
        //{
        //}
        //
        //splindex(int index_size__,
        //         const std::vector<int>& dimensions__, 
        //         const std::vector<int>& coordinates__,
        //         const MPI_Comm communicator__ = MPI_COMM_WORLD) : index_size_(index_size__),
        //                                                           dimensions_(dimensions__),
        //                                                           coordinates_(coordinates__),
        //                                                           communicator_(communicator__)
        //{
        //    if (dimensions__.size() == 0)
        //        error(__FILE__, __LINE__, "empty array of dimensions", fatal_err);

        //    if (dimensions__.size() != coordinates__.size())
        //        error(__FILE__, __LINE__, "sizes don't match", fatal_err);

        //    for (int i = 0; i < (int)dimensions_.size(); i++)
        //    {
        //        if ((coordinates_[i] < 0) || (coordinates_[i] >= dimensions_[i]))
        //        {
        //            std::stringstream s;
        //            s << "bad coordinates" << std::endl
        //              << "  direction : " << i << std::endl
        //              << "  coordinate : " << coordinates_[i] << std::endl
        //              << "  dimension size : " << dimensions_[i];
        //            error(__FILE__, __LINE__, s, fatal_err);
        //        }
        //    }

        //    if (index_size_ == 0)
        //        error(__FILE__, __LINE__, "need to think what to do with zero index size", fatal_err);

        //    num_ranks_ = 1;
        //    for (int i = 0; i < (int)dimensions_.size(); i++)
        //        num_ranks_ *= dimensions_[i];

        //    offset_ = std::vector<int>(dimensions_.size(), 0);
        //    int n = 1;
        //    for (int i = 1; i < (int)dimensions_.size(); i++) 
        //    {
        //        n *= dimensions_[i - 1];
        //        offset_[i] = n;
        //    }
        //    
        //    rank_id_ = coordinates_[0];
        //    for (int i = 1; i < (int)dimensions_.size(); i++) 
        //        rank_id_ += offset_[i] * coordinates_[i];

        //    // minimum size
        //    int n1 = index_size_ / num_ranks_;

        //    // first n2 ranks have one more index element
        //    int n2 = index_size_ % num_ranks_;

        //    if (rank_id_ < n2)
        //    {
        //        local_size_ = n1 + 1;
        //        global_index_offset_ = (n1 + 1) * rank_id_;
        //    }
        //    else
        //    {   
        //        local_size_ = n1;
        //        global_index_offset_ = (n1 > 0) ? (n1 + 1) * n2 + n1 * (rank_id_ - n2) : -1;
        //    }
        //}

        inline int local_size()
        {
            assert(rank_ >= 0);
            
            return local_size_[rank_];
        }

        inline int local_size(int rank)
        {
            assert((rank >= 0) && rank < (num_ranks_));

            return local_size_[rank];
        }

        inline int global_index(int idxloc)
        {
            return global_index_(idxloc, rank_);
        }

        inline int global_index(int idxloc, int rank)
        {
            return global_index_(idxloc, rank);
        }

        inline int location(int i, int idxglob)
        {
            return location_(i, idxglob);
        }
};

