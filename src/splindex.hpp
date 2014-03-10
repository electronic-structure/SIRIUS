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

        inline int local_size()
        {
            return local_size(rank_);
        }

        inline int local_size(int rank) // TODO: can be cached
        {
            int m = global_index_size_ % num_ranks_; // first m ranks get additional element
            return global_index_size_ / num_ranks_ + (rank < m ? 1 : 0); // minimum number of elements +1 if rank < m
        }
        
        inline int location(int offset_or_rank, int idxglob) // TODO: rename
        {
            assert(idxglob < global_index_size_);

            int m = global_index_size_ % num_ranks_; // first m ranks get additional element
            int bs_min = global_index_size_ / num_ranks_;

            int rank;
            int offs;

            if (idxglob < (bs_min + 1) * m) // TODO: can be cached
            {
                rank = idxglob / (bs_min + 1);
                offs = idxglob % (bs_min + 1);
            }
            else
            {
                int k = idxglob - m * (bs_min + 1);
                offs = k % bs_min;
                rank = m + k / bs_min;
            }

            switch (offset_or_rank)
            {
                case _splindex_offs_:
                {
                    return offs;
                    break;
                }
                case _splindex_rank_:
                {
                    return rank;
                    break;
                }
            }
            return -1; // make compiler happy
        }

        inline int global_index(int idxloc, int rank)
        {
            int m = global_index_size_ % num_ranks_; // first m ranks get additional element
            int bs_min = global_index_size_ / num_ranks_;

            return (rank < m) ? (bs_min + 1) * rank + idxloc : m + rank * bs_min + idxloc; 
        }

        inline int global_offset()
        {
            assert(rank_ >= 0);
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

        
        inline int local_size(int rank)
        {
            int num_blocks = global_index_size_ / block_size_; // number of full blocks

            int n = (num_blocks / num_ranks_) * block_size_;

            int rank_offs = num_blocks % num_ranks_;

            if (rank < rank_offs) 
            {
                n += block_size_;
            }
            else if (rank == rank_offs)
            {
                n += global_index_size_ % block_size_;
            }
            return n;
        }

        inline int local_size()
        {
            return local_size(rank_);
        }

        inline int location(int offset_or_rank, int idxglob) // TODO: rename
        {
            assert(idxglob < global_index_size_);

            int num_blocks = idxglob / block_size_; // number of full blocks

            int n = (num_blocks / num_ranks_) * block_size_ + idxglob % block_size_;

            int rank = num_blocks % num_ranks_;

            switch (offset_or_rank)
            {
                case _splindex_offs_:
                {
                    return n;
                    break;
                }
                case _splindex_rank_:
                {
                    return rank;
                    break;
                }
            }
            return -1; // make compiler happy
        }

        inline int global_index(int idxloc, int rank)
        {
            int nb = idxloc / block_size_;
            
            int idx = nb * num_ranks_ * block_size_ + idxloc % block_size_;
            idx += rank * block_size_;
            return idx;
        }

};

