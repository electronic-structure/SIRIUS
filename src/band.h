#ifndef __BAND_H__
#define __BAND_H__

/** \file band.h
    
    \brief Setup and solve eigen value problem.
*/

#include "global.h"
#include "periodic_function.h"
#include "k_point.h"

namespace sirius
{

// TODO: Band problem is a mess and needs more formal organizaiton. We have different basis functions. 
//       We can do first- and second-variation or a full variation. We can do iterative or exact diagonalization.
//       This has to be organized. 

class Band
{
    private:

        /// global set of parameters
        Global& parameters_;

        /// alias for FFT driver
        FFT3D<cpu>* fft_;
        
        /// Non-zero Gaunt coefficients
        Gaunt_coefficients<double_complex>* gaunt_coefs_;
    
        /// Apply effective magentic field to the first-variational state.
        /** Must be called first because hpsi is overwritten with B|fv_j>. */
        void apply_magnetic_field(mdarray<double_complex, 2>& fv_states, int num_gkvec, int* fft_index, 
                                  Periodic_function<double>* effective_magnetic_field[3], mdarray<double_complex, 3>& hpsi);

        /// Apply SO correction to the first-variational states.
        /** Raising and lowering operators:
            \f[
                L_{\pm} Y_{\ell m}= (L_x \pm i L_y) Y_{\ell m}  = \sqrt{\ell(\ell+1) - m(m \pm 1)} Y_{\ell m \pm 1}
            \f]
        */
        void apply_so_correction(mdarray<double_complex, 2>& fv_states, mdarray<double_complex, 3>& hpsi);
        
        /// Apply UJ correction to scalar wave functions
        template <spin_block_t sblock>
        void apply_uj_correction(mdarray<double_complex, 2>& fv_states, mdarray<double_complex, 3>& hpsi);

        /// Add interstitial contribution to apw-apw block of Hamiltonian and overlap
        void set_fv_h_o_it(K_point* kp, Periodic_function<double>* effective_potential, 
                           mdarray<double_complex, 2>& h, mdarray<double_complex, 2>& o);

        void set_o_it(K_point* kp, mdarray<double_complex, 2>& o);

        template <spin_block_t sblock>
        void set_h_it(K_point* kp, Periodic_function<double>* effective_potential, 
                      Periodic_function<double>* effective_magnetic_field[3], mdarray<double_complex, 2>& h);
        
        /// Setup lo-lo block of Hamiltonian and overlap matrices
        void set_fv_h_o_lo_lo(K_point* kp, mdarray<double_complex, 2>& h, mdarray<double_complex, 2>& o);

        template <spin_block_t sblock>
        void set_h_lo_lo(K_point* kp, mdarray<double_complex, 2>& h);
        
        void set_o_lo_lo(K_point* kp, mdarray<double_complex, 2>& o);
       
        void set_o(K_point* kp, mdarray<double_complex, 2>& o);
    
        template <spin_block_t sblock> 
        void set_h(K_point* kp, Periodic_function<double>* effective_potential, 
                   Periodic_function<double>* effective_magnetic_field[3], mdarray<double_complex, 2>& h);
       
        void diag_fv_full_potential(K_point* kp, Periodic_function<double>* effective_potential);

        void apply_h_local(K_point* kp, std::vector<double>& effective_potential, std::vector<double>& pw_ekin, 
                           int n, double_complex* phi__, double_complex* hphi__);

        void diag_fv_uspp_cpu(K_point* kp, Periodic_function<double>* effective_potential);
        
        void get_h_o_diag(K_point* kp, Periodic_function<double>* effective_potential, std::vector<double>& pw_ekin, 
                          std::vector<double_complex>& h_diag, std::vector<double_complex>& o_diag);

        void apply_h_o_uspp_cpu(K_point* kp, std::vector<double>& effective_potential, std::vector<double>& pw_ekin, int n,
                                double_complex* phi__, double_complex* hphi__, double_complex* ophi__);

        #ifdef _GPU_
        void apply_h_local_gpu(K_point* kp, std::vector<double>& effective_potential, std::vector<double>& pw_ekin, 
                               int num_phi, mdarray<double_complex, 2>& gamma, mdarray<double_complex, 2>& kappa, double_complex* phi__, 
                               double_complex* hphi__);

        void apply_h_o_uspp_gpu(K_point* kp, std::vector<double>& effective_potential, std::vector<double>& pw_ekin, int n,
                                mdarray<double_complex, 2>& gamma, mdarray<double_complex, 2>& kappa, double_complex* phi__, 
                                double_complex* hphi__, double_complex* ophi__);

        void diag_fv_uspp_gpu(K_point* kp, Periodic_function<double>* effective_potential);
        #endif

    public:
        
        /// Constructor
        Band(Global& parameters__) : parameters_(parameters__)
        {
            fft_ = parameters_.reciprocal_lattice()->fft();

            gaunt_coefs_ = new Gaunt_coefficients<double_complex>(parameters_.lmax_apw(), parameters_.lmax_pot(), 
                                                                  parameters_.lmax_apw());
        }

        ~Band()
        {
            delete gaunt_coefs_;
        }

        /// Apply the muffin-tin part of the first-variational Hamiltonian to the apw basis function
        /** The following vector is computed:
         *  \f[
         *    b_{L_2 \nu_2}^{\alpha}({\bf G'}) = \sum_{L_1 \nu_1} \sum_{L_3} 
         *      a_{L_1\nu_1}^{\alpha*}({\bf G'}) 
         *      \langle u_{\ell_1\nu_1}^{\alpha} | h_{L3}^{\alpha} |  u_{\ell_2\nu_2}^{\alpha}  
         *      \rangle  \langle Y_{L_1} | R_{L_3} | Y_{L_2} \rangle +  
         *      \frac{1}{2} \sum_{\nu_1} a_{L_2\nu_1}^{\alpha *}({\bf G'})
         *      u_{\ell_2\nu_1}^{\alpha}(R_{\alpha})
         *      u_{\ell_2\nu_2}^{'\alpha}(R_{\alpha})R_{\alpha}^{2}
         *  \f] 
         */
        template <spin_block_t sblock>
        void apply_hmt_to_apw(int num_gkvec, int ia, mdarray<double_complex, 2>& alm, mdarray<double_complex, 2>& halm);
 
        template <spin_block_t sblock>
        void apply_hmt_to_apw(mdarray<double_complex, 2>& alm, mdarray<double_complex, 2>& halm);


        //== template <spin_block_t sblock>
        //== void apply_hmt_to_apw(K_point* kp, dmatrix<double_complex>& halm_panel)
        //== {
        //==     Timer t("sirius::Band::apply_hmt_to_apw", _global_timer_);

        //==     std::map< int, mdarray<double_complex, 2>* > alm;

        //==     /* build list of matching coefficients */
        //==     for (int i = 0; i < halm_panel.num_rows_local(); i++)
        //==     {
        //==         int j = halm_panel.irow(i);
        //==         int ia = parameters_.unit_cell()->mt_aw_basis_descriptor(j).ia;
        //==         if (alm.count(ia) == 0)
        //==         {
        //==             alm[ia] = new mdarray<double_complex, 2>(kp->num_gkvec_loc(), parameters_.unit_cell()->max_mt_aw_basis_size());
        //==             kp->generate_matching_coefficients<false>(kp->num_gkvec_loc(), ia, *alm[ia]);
        //==         }
        //==     }

        //==     int apw_offset_col = kp->apw_offset_col();

        //==     Timer t1("sirius::Band::apply_hmt_to_apw:halm", _global_timer_);

        //==     #pragma omp parallel default(shared)
        //==     {
        //==         std::vector<double_complex> zv(kp->num_gkvec_col());

        //==         #pragma omp for
        //==         for (int i = 0; i < halm_panel.num_rows_local(); i++)
        //==         {
        //==             int j = halm_panel.irow(i);

        //==             int ia = parameters_.unit_cell()->mt_aw_basis_descriptor(j).ia;
        //==             int xi = parameters_.unit_cell()->mt_aw_basis_descriptor(j).xi;
        //==             Atom* atom = parameters_.unit_cell()->atom(ia);
        //==             Atom_type* type = atom->type();
        //==             int l1 = type->indexb(xi).l;
        //==             int lm1 = type->indexb(xi).lm;
        //==             int idxrf1 = type->indexb(xi).idxrf;
        //==             int order1 = type->indexb(xi).order;

        //==             memset(&zv[0], 0, zv.size() * sizeof(double_complex));

        //==             for (int j2 = 0; j2 < type->mt_aw_basis_size(); j2++)
        //==             {
        //==                 int lm2 = type->indexb(j2).lm;
        //==                 int idxrf2 = type->indexb(j2).idxrf;
        //==                 double_complex zsum = atom->hb_radial_integrals_sum_L3<sblock>(idxrf1, idxrf2, gaunt_coefs_->gaunt_vector(lm1, lm2));

        //==                 if (abs(zsum) > 1e-14)
        //==                 {
        //==                     for (int igk = 0; igk < kp->num_gkvec_col(); igk++) zv[igk] += zsum * (*alm[ia])(apw_offset_col + igk, j2); 
        //==                 }
        //==             }

        //==             /* surface contribution */
        //==             if (sblock == nm || sblock == uu || sblock == dd)
        //==             {
        //==                 for (int order2 = 0; order2 < (int)type->aw_descriptor(l1).size(); order2++)
        //==                 {
        //==                     double t1 = 0.5 * pow(type->mt_radius(), 2) *
        //==                                 atom->symmetry_class()->aw_surface_dm(l1, order1, 0) *
        //==                                 atom->symmetry_class()->aw_surface_dm(l1, order2, 1);

        //==                     for (int igk = 0; igk < kp->num_gkvec_col(); igk++)
        //==                         zv[igk] += t1 * (*alm[ia])(apw_offset_col + igk, type->indexb_by_lm_order(lm1, order2));
        //==                 }
        //==             }

        //==             for (int igk = 0; igk < kp->num_gkvec_col(); igk++) halm_panel(i, igk) = zv[igk];
        //==         }
        //==     }

        //==     for (auto& a: alm) delete a.second;
        //== }

        /// Setup apw-lo and lo-apw blocs of Hamiltonian and overlap matrices
        void set_fv_h_o_apw_lo(K_point* kp,
                               Atom_type* type,
                               Atom* atom,
                               int ia,
                               mdarray<double_complex, 2>& alm_row,
                               mdarray<double_complex, 2>& alm_col,
                               mdarray<double_complex, 2>& h,
                               mdarray<double_complex, 2>& o);
        
        template <spin_block_t sblock>
        void set_h_apw_lo(K_point* kp, Atom_type* type, Atom* atom, int ia, mdarray<double_complex, 2>& alm, 
                          mdarray<double_complex, 2>& h);

        void set_o_apw_lo(K_point* kp, Atom_type* type, Atom* atom, int ia, mdarray<double_complex, 2>& alm, 
                          mdarray<double_complex, 2>& o);

        /// Setup the Hamiltonian and overlap matrices in APW+lo basis
        /** The Hamiltonian matrix has the following expression:
         *  \f[
         *      H_{\mu' \mu} = \langle \varphi_{\mu'} | \hat H | \varphi_{\mu} \rangle
         *  \f]
         *
         *  \f[
         *      H_{\mu' \mu}=\langle \varphi_{\mu' } | \hat H | \varphi_{\mu } \rangle  = 
         *      \left( \begin{array}{cc} 
         *         H_{\bf G'G} & H_{{\bf G'}j} \\
         *         H_{j'{\bf G}} & H_{j'j}
         *      \end{array} \right)
         *  \f]
         *  
         *  The overlap matrix has the following expression:
         *  \f[
         *      O_{\mu' \mu} = \langle \varphi_{\mu'} | \varphi_{\mu} \rangle
         *  \f]
         *  APW-APW block:
         *  \f[
         *      O_{{\bf G'} {\bf G}}^{\bf k} = \sum_{\alpha} \sum_{L\nu} a_{L\nu}^{\alpha *}({\bf G'+k}) 
         *      a_{L\nu}^{\alpha}({\bf G+k})
         *  \f]
         *  
         *  APW-lo block:
         *  \f[
         *      O_{{\bf G'} j}^{\bf k} = \sum_{\nu'} a_{\ell_j m_j \nu'}^{\alpha_j *}({\bf G'+k}) 
         *      \langle u_{\ell_j \nu'}^{\alpha_j} | \phi_{\ell_j}^{\zeta_j \alpha_j} \rangle
         *  \f]
         *
         *  lo-APW block:
         *  \f[
         *      O_{j' {\bf G}}^{\bf k} = 
         *      \sum_{\nu'} \langle \phi_{\ell_{j'}}^{\zeta_{j'} \alpha_{j'}} | u_{\ell_{j'} \nu'}^{\alpha_{j'}} \rangle
         *      a_{\ell_{j'} m_{j'} \nu'}^{\alpha_{j'}}({\bf G+k}) 
         *  \f]
         *
         *  lo-lo block:
         *  \f[
         *      O_{j' j}^{\bf k} = \langle \phi_{\ell_{j'}}^{\zeta_{j'} \alpha_{j'}} | 
         *      \phi_{\ell_{j}}^{\zeta_{j} \alpha_{j}} \rangle \delta_{\alpha_{j'} \alpha_j} 
         *      \delta_{\ell_{j'} \ell_j} \delta_{m_{j'} m_j}
         *  \f]
         *
         */
        template <processing_unit_t pu, electronic_structure_method_t basis>
        void set_fv_h_o(K_point* kp, Periodic_function<double>* effective_potential, dmatrix<double_complex>& h, 
                        dmatrix<double_complex>& o);

        /// Solve first-variational problem
        void solve_fv(K_point* kp, Periodic_function<double>* effective_potential);

        /// Solve second-variational problem
        void solve_sv(K_point* kp, Periodic_function<double>* effective_magnetic_field[3]);

        void solve_fd(K_point* kp, Periodic_function<double>* effective_potential, 
                      Periodic_function<double>* effective_magnetic_field[3]);
};

#include "band.hpp"

}

#endif // __BAND_H__
