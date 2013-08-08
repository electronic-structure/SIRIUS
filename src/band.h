/** \file band.h
    \brief Setup and solve second-variational eigen value problem.
*/

namespace sirius
{

class Band
{
    private:

        /// Global set of parameters.
        Global& parameters_;
    
        /// Block-cyclic distribution of the first-variational states along columns of the MPI grid.
        /** Very important! The number of first-variational states is aligned in such a way that each row or 
            column MPI rank gets equal row or column fraction of the fv states. This is done in order to have a 
            simple acces to the fv states when, for example, spin blocks of the second-variational Hamiltonian are
            constructed.
        */
        splindex<block_cyclic> spl_fv_states_col_;

        /// Block-cyclic distribution of the first-variational states along rows of the MPI grid.
        splindex<block_cyclic> spl_fv_states_row_;
        
        splindex<block_cyclic> spl_spinor_wf_col_;
        
        splindex<block> sub_spl_spinor_wf_;
        
        //int num_fv_states_row_up_;
        
        //int num_fv_states_row_dn_;

        int rank_col_;

        int num_ranks_col_;

        int rank_row_;

        int num_ranks_row_;

        int num_ranks_;
        
        /// BLACS communication context
        int blacs_context_;
        
        // assumes that hpsi is zero on input
        void apply_magnetic_field(mdarray<complex16, 2>& fv_states, int mtgk_size, int num_gkvec, int* fft_index, 
                                  PeriodicFunction<double>* effective_magnetic_field[3], mdarray<complex16, 3>& hpsi);

        /// Apply SO correction to the scalar wave functions
        /** Raising lowering operators:
            \f[
                L_{\pm} Y_{\ell m}= (L_x \pm i L_y) Y_{\ell m}  = \sqrt{\ell(\ell+1) - m(m \pm 1)} Y_{\ell m \pm 1}
            \f]
        */
        void apply_so_correction(mdarray<complex16, 2>& fv_states, mdarray<complex16, 3>& hpsi);
        
        /// Apply UJ correction to scalar wave functions
        template <spin_block_t sblock>
        void apply_uj_correction(mdarray<complex16, 2>& fv_states, mdarray<complex16, 3>& hpsi);

        void init();
 
    public:
        
        /// Constructor
        Band(Global& parameters__);

        ~Band();

        void solve_sv(Global& parameters, int mtgk_size, int num_gkvec, int* fft_index, double* evalfv, 
                      mdarray<complex16, 2>& fv_states_row, mdarray<complex16, 2>& fv_states_col, 
                      PeriodicFunction<double>* effective_magnetic_field[3], double* band_energies, 
                      mdarray<complex16, 2>& sv_eigen_vectors);
        
        bool need_sv()
        {
            if (parameters_.num_spins() == 2 || parameters_.uj_correction() || parameters_.so_correction())
            {
                return true;
            }
            else
            {
                return false;
            }
        }
        
        inline splindex<block_cyclic>& spl_fv_states_col()
        {
            return spl_fv_states_col_;
        }
        
        inline int spl_fv_states_col(int icol_loc)
        {
            return spl_fv_states_col_[icol_loc];
        }

        inline splindex<block_cyclic>& spl_fv_states_row()
        {
            return spl_fv_states_row_;
        }
        
        inline int spl_fv_states_row(int irow_loc)
        {
            return spl_fv_states_row_[irow_loc];
        }
        
        inline splindex<block_cyclic>& spl_spinor_wf_col()
        {
            return spl_spinor_wf_col_;
        }
        
        inline int spl_spinor_wf_col(int jloc)
        {
            return spl_spinor_wf_col_[jloc];
        }
        
        inline int num_sub_bands()
        {
            return sub_spl_spinor_wf_.local_size();
        }

        inline int idxbandglob(int sub_index)
        {
            return spl_spinor_wf_col_[sub_spl_spinor_wf_[sub_index]];
        }
        
        inline int idxbandloc(int sub_index)
        {
            return sub_spl_spinor_wf_[sub_index];
        }

        inline int num_ranks_row()
        {
            return num_ranks_row_;
        }
        
        inline int rank_row()
        {
            return rank_row_;
        }
        
        inline int num_ranks_col()
        {
            return num_ranks_col_;
        }
        
        inline int rank_col()
        {
            return rank_col_;
        }
        
        inline int num_ranks()
        {
            return num_ranks_;
        }
        
        inline int blacs_context()
        {
            return blacs_context_;
        }
};

#include "band.hpp"

};
