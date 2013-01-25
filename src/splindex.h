class splindex_base
{
    private:
        
        /// forbid copy constructor
        splindex_base(const splindex_base& src);

        /// forbid assigment operator
        splindex_base& operator=(const splindex_base& src);
    
    protected:
        
        /// rank of the block with local fraction of the global index
        int rank_;
        
        /// number of ranks over which the global index is distributed
        int num_ranks_;

        int global_index_size_;

        /// local index size for each rank
        std::vector<int> local_size_;
        
        /// global index by rank and local index
        mdarray<int, 2> global_index_;

        /// location (local index and rank) of global index
        mdarray<int, 2> location_;
        
        splindex_base() : rank_(-1), num_ranks_(-1)
        {
        }

        void init()
        {
            if (num_ranks_ == 1) assert(local_size_[0] == global_index_size_);

            location_.set_dimensions(2, global_index_size_);
            location_.allocate();

            for (int i1 = 0; i1 < global_index_.size(1); i1++)
            {
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
        }
    
    public:

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

        inline int global_index(int idxloc, int rank)
        {
            return global_index_(idxloc, rank);
        }

        inline int location(int i, int idxglob)
        {
            return location_(i, idxglob);
        }

        inline int operator[](int idxloc)
        {
            return global_index_(idxloc, rank_);
        }
};

template <splindex_t type> class splindex: public splindex_base
{
};

template<> class splindex<block>: public splindex_base
{
    public:
        
        splindex()
        {
        }
        
        splindex(int global_index_size__, int num_ranks__, int rank__)
        {
            assert(global_index_size__ > 0);

            split(global_index_size__, num_ranks__, rank__); 
        }
        
        void split(int global_index_size__, int num_ranks__, int rank__)
        {
            assert(num_ranks__ > 0);
            assert((rank__ >= 0) && (rank__ < num_ranks__));
            assert(global_index_size__ >= 0);

            if (global_index_size__ == 0)
                error(__FILE__, __LINE__, "need to think what to do with zero index size", fatal_err);

            rank_ = rank__;
            num_ranks_ = num_ranks__;
            global_index_size_ = global_index_size__;
            
            local_size_.resize(num_ranks__);

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

            init();
        }


};

template<> class splindex<block_cyclic>: public splindex_base
{
    private:
        
        int block_size_;

    public:
        
        splindex()
        {
        }

        splindex(int global_index_size__, int num_ranks__, int rank__, int block_size__)
        {
            assert(global_index_size__ > 0);

            split(global_index_size__, num_ranks__, rank__, block_size__); 
        }
        
        void split(int global_index_size__, int num_ranks__, int rank__, int block_size__)
        {
            assert(num_ranks__ > 0);
            assert((rank__ >= 0) && (rank__ < num_ranks__));
            assert(global_index_size__ >= 0);
            assert(block_size__ > 0);

            if (global_index_size__ == 0)
                error(__FILE__, __LINE__, "need to think what to do with zero index size", fatal_err);

            rank_ = rank__;
            num_ranks_ = num_ranks__;
            global_index_size_ = global_index_size__;
            block_size_ = block_size__;
            
            local_size_.resize(num_ranks__);


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

            init();
        }
};

