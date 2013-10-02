/** \file band.h
    
    \brief Setup and solve eigen value problem.
*/

namespace sirius
{

class Band
{
    private:

        /// global set of parameters
        Global& parameters_;
    
        /// Apply effective magentic field to the first-variational state.
        /** Must be called first because hpsi is overwritten with B|fv_j>. */
        void apply_magnetic_field(mdarray<complex16, 2>& fv_states, int mtgk_size, int num_gkvec, int* fft_index, 
                                  Periodic_function<double>* effective_magnetic_field[3], mdarray<complex16, 3>& hpsi);

        /// Apply SO correction to the first-variational states.
        /** Raising and lowering operators:
            \f[
                L_{\pm} Y_{\ell m}= (L_x \pm i L_y) Y_{\ell m}  = \sqrt{\ell(\ell+1) - m(m \pm 1)} Y_{\ell m \pm 1}
            \f]
        */
        void apply_so_correction(mdarray<complex16, 2>& fv_states, mdarray<complex16, 3>& hpsi);
        
        /// Apply UJ correction to scalar wave functions
        template <spin_block_t sblock>
        void apply_uj_correction(mdarray<complex16, 2>& fv_states, mdarray<complex16, 3>& hpsi);

        /// Apply the muffin-tin part of the first-variational Hamiltonian to the apw basis function
        /** The following vector is computed:
            \f[
              b_{L_2 \nu_2}^{\alpha}({\bf G'}) = \sum_{L_1 \nu_1} \sum_{L_3} 
                a_{L_1\nu_1}^{\alpha*}({\bf G'}) 
                \langle u_{\ell_1\nu_1}^{\alpha} | h_{L3}^{\alpha} |  u_{\ell_2\nu_2}^{\alpha}  
                \rangle  \langle Y_{L_1} | R_{L_3} | Y_{L_2} \rangle +  
                \frac{1}{2} \sum_{\nu_1} a_{L_2\nu_1}^{\alpha *}({\bf G'})
                u_{\ell_2\nu_1}^{\alpha}(R_{\alpha})
                u_{\ell_2\nu_2}^{'\alpha}(R_{\alpha})R_{\alpha}^{2}
            \f] 
        */
        void apply_hmt_to_apw(int num_gkvec, int ia, mdarray<complex16, 2>& alm, mdarray<complex16, 2>& halm);
 
        void set_fv_h_o_apw_lo(K_point* kp, Atom_type* type, Atom* atom, int ia, mdarray<complex16, 2>& alm, 
                               mdarray<complex16, 2>& h, mdarray<complex16, 2>& o);
        
        void set_fv_h_o_it(K_point* kp, Periodic_function<double>* effective_potential, 
                           mdarray<complex16, 2>& h, mdarray<complex16, 2>& o);
        
        void set_fv_h_o_lo_lo(K_point* kp, mdarray<complex16, 2>& h, mdarray<complex16, 2>& o);

        void solve_fv_evp_1stage(K_point* kp, mdarray<complex16, 2>& h, mdarray<complex16, 2>& o, 
                                 std::vector<double>& fv_eigen_values, mdarray<complex16, 2>& fv_eigen_vectors);
    
    public:
        
        /// Constructor
        Band(Global& parameters__) : parameters_(parameters__)
        {
        }

        ~Band()
        {
        }

        template <processing_unit_t pu, basis_t basis>
        void set_fv_h_o(K_point* kp, Periodic_function<double>* effective_potential,
                        mdarray<complex16, 2>& h, mdarray<complex16, 2>& o);

        void solve_fv(K_point* kp, Periodic_function<double>* effective_potential);

        void solve_sv(K_point* kp, Periodic_function<double>* effective_magnetic_field[3]);
        
};

#include "band.hpp"

};
