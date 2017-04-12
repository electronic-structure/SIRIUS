#ifndef __BETA_PROJECTOR_CHUNKS_H__
#define __BETA_PROJECTOR_CHUNKS_H__

namespace sirius {

enum beta_desc_idx {
    nbf      = 0,
    offset   = 1,
    offset_t = 2,
    ia       = 3
};

struct beta_chunk_t
{
    int num_beta_;
    int num_atoms_;
    int offset_;
    mdarray<int, 2> desc_;
    mdarray<double, 2> atom_pos_;
};

/// Split beta-projectors and related arrays into chunks.
class Beta_projector_chunks
{
  protected:

    Unit_cell const& unit_cell_;

    std::vector<beta_chunk_t> beta_chunks_;

    int max_num_beta_;

    /// Total number of beta-projectors among atom types.
    int num_beta_t_;
    
    /// Split beta-projectors into chunks.
    void split_in_chunks()
    {   
        /* initial chunk size */
        int chunk_size = std::min(unit_cell_.num_atoms(), 256);
        /* maximum number of chunks */
        int num_chunks = unit_cell_.num_atoms() / chunk_size + std::min(1, unit_cell_.num_atoms() % chunk_size);
        /* final maximum chunk size */
        chunk_size = unit_cell_.num_atoms() / num_chunks + std::min(1, unit_cell_.num_atoms() % num_chunks);

        int offset_in_beta_gk{0};
        beta_chunks_ = std::vector<beta_chunk_t>(num_chunks);

        for (int ib = 0; ib < num_chunks; ib++) {
            /* number of atoms in this chunk */
            int na = std::min(unit_cell_.num_atoms(), (ib + 1) * chunk_size) - ib * chunk_size;
            beta_chunks_[ib].num_atoms_ = na;
            beta_chunks_[ib].desc_      = mdarray<int, 2>(4, na);
            beta_chunks_[ib].atom_pos_  = mdarray<double, 2>(3, na);

            int num_beta{0};
            for (int i = 0; i < na; i++) {
                /* global index of atom by local index and chunk */
                int ia     = ib * chunk_size + i;
                auto pos   = unit_cell_.atom(ia).position();
                auto& type = unit_cell_.atom(ia).type();
                /* atom fractional coordinates */
                for (int x: {0, 1, 2}) {
                    beta_chunks_[ib].atom_pos_(x, i) = pos[x];
                }
                /* number of beta functions for atom */
                beta_chunks_[ib].desc_(beta_desc_idx::nbf, i) = type.mt_basis_size();
                /* offset in beta_gk*/
                beta_chunks_[ib].desc_(beta_desc_idx::offset, i) = num_beta;
                /* offset in beta_gk_t */
                beta_chunks_[ib].desc_(beta_desc_idx::offset_t, i) = type.offset_lo();
                /* global index of atom */
                beta_chunks_[ib].desc_(beta_desc_idx::ia, i) = ia;

                num_beta += type.mt_basis_size();
            }
            /* number of beta-projectors in this chunk */
            beta_chunks_[ib].num_beta_ = num_beta;
            beta_chunks_[ib].offset_ = offset_in_beta_gk;
            offset_in_beta_gk += num_beta;

            if (unit_cell_.parameters().processing_unit() == GPU) {
                beta_chunks_[ib].desc_.allocate(memory_t::device);
                beta_chunks_[ib].desc_.copy<memory_t::host, memory_t::device>();

                beta_chunks_[ib].atom_pos_.allocate(memory_t::device);
                beta_chunks_[ib].atom_pos_.copy<memory_t::host, memory_t::device>();
            }
        }

        max_num_beta_ = 0;
        for (auto& e: beta_chunks_) {
            max_num_beta_ = std::max(max_num_beta_, e.num_beta_);
        }

        num_beta_t_ = 0;
        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
            num_beta_t_ += unit_cell_.atom_type(iat).mt_lo_basis_size();
        }

    }

    Beta_projector_chunks(Beta_projector_chunks& src) = delete;

    Beta_projector_chunks operator=(Beta_projector_chunks& src) = delete;

  public:

    Beta_projector_chunks(Unit_cell const& unit_cell__)
        : unit_cell_(unit_cell__)
    {
        split_in_chunks();
    }

    inline int num_beta_t() const
    {
        return num_beta_t_;
    }

    inline int num_chunks() const
    {
        return static_cast<int>(beta_chunks_.size());
    }

    inline beta_chunk_t const& operator()(int idx__) const
    {
        return beta_chunks_[idx__];
    }

    inline int max_num_beta() const
    {
        return max_num_beta_;
    }

    void print_info()
    {
        printf("num_beta_chunks: %i\n", num_chunks());
        for (int i = 0; i < num_chunks(); i++) {
            printf("  chunk: %i, num_atoms: %i, num_beta: %i\n", i, beta_chunks_[i].num_atoms_, beta_chunks_[i].num_beta_);
        }
    }
};

}

#endif //__BETA_PROJECTOR_CHUNKS_H__

