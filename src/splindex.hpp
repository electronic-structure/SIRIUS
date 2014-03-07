template<> 
class splindex<block>: public splindex_base
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

            //if (global_index_size__ == 0)
            //    error_local(__FILE__, __LINE__, "need to think what to do with zero index size");

            rank_ = rank__;
            num_ranks_ = num_ranks__;
            global_index_size_ = global_index_size__;
            
            local_size_.resize(num_ranks_);

            // minimum size
            int n1 = global_index_size_ / num_ranks_;

            // first n2 ranks have one more index element
            int n2 = global_index_size_ % num_ranks_;
            
            global_index_.set_dimensions(n1 + std::min(1, n2), num_ranks_);
            global_index_.allocate();
            for (int i1 = 0; i1 < global_index_.size(_splindex_rank_); i1++)
            {
                for (int i0 = 0; i0 < global_index_.size(_splindex_offs_); i0++) global_index_(i0, i1) = -1;
            }

            for (int i1 = 0; i1 < num_ranks_; i1++)
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
                for (int i0 = 0; i0 < local_size_[i1]; i0++) global_index_(i0, i1) = global_index_offset + i0;
            }

            init();
        }

        inline int global_offset()
        {
            return global_index_(0, rank_);
        }

};

template<> 
class splindex<block_cyclic>: public splindex_base
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
                error_local(__FILE__, __LINE__, "need to think what to do with zero index size");

            rank_ = rank__;
            num_ranks_ = num_ranks__;
            global_index_size_ = global_index_size__;
            block_size_ = block_size__;
            
            local_size_.resize(num_ranks_);


            int nblocks = (global_index_size_ / block_size_) +           // number of full blocks
                          std::min(1, global_index_size_ % block_size_); // extra partial block

            int max_size = ((nblocks / num_ranks_) +            // minimum number of blocks per rank
                            std::min(1, nblocks % num_ranks_)); // some ranks get extra block
            max_size *= block_size_;
            
            global_index_.set_dimensions(max_size, num_ranks_);
            global_index_.allocate();
            
            int irank = 0;
            int iblock = 0;
            std::vector< std::vector<int> > iv(num_ranks_);
            for (int i = 0; i < global_index_size_; i++)
            {
                iv[irank].push_back(i);
                if ((++iblock) == block_size_)
                {
                    iblock = 0;
                    irank = (irank + 1) % num_ranks_;
                }
            }
           
            for (int i1 = 0; i1 < global_index_.size(_splindex_rank_); i1++)
            {
                local_size_[i1] = (int)iv[i1].size();
                for (int i0 = 0; i0 < global_index_.size(_splindex_offs_); i0++)
                    global_index_(i0, i1) = (i0 < (int)iv[i1].size()) ? iv[i1][i0] : -1;
            }

            init();
        }
};

