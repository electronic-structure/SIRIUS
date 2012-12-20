
/** \file band.h
    \brief Setup and solve first- and second-variational eigen value problem.
*/

namespace sirius
{

/// Descriptor of the APW+lo basis function

/** APW+lo basis consists of two different sets of functions: APW functions \f$ \varphi_{{\bf G+k}} \f$ defined over 
    entire unit cell:
    \f[
        \varphi_{{\bf G+k}}({\bf r}) = \left\{ \begin{array}{ll}
        \displaystyle \sum_{L} \sum_{\nu=1}^{O_{\ell}^{\alpha}} a_{L\nu}^{\alpha}({\bf G+k})u_{\ell \nu}^{\alpha}(r) 
        Y_{\ell m}(\hat {\bf r}) & {\bf r} \in {\rm MT} \alpha \\
        \displaystyle \frac{1}{\sqrt  \Omega} e^{i({\bf G+k}){\bf r}} & {\bf r} \in {\rm I} \end{array} \right.
    \f]  
    and Bloch sums of local orbitals defined inside muffin-tin spheres only:
    \f[
        \begin{array}{ll} \displaystyle \varphi_{j{\bf k}}({\bf r})=\sum_{{\bf T}} e^{i{\bf kT}} 
        \varphi_{j}({\bf r - T}) & {\rm {\bf r} \in MT} \end{array}
    \f]
    Each local orbital is composed of radial and angular parts:
    \f[
        \varphi_{j}({\bf r}) = \phi_{\ell_j}^{\zeta_j,\alpha_j}(r) Y_{\ell_j m_j}(\hat {\bf r})
    \f]
    Radial part of local orbital is defined as a linear combination of radial functions (minimum two radial functions 
    are required) such that local orbital vanishes at the sphere boundary:
    \f[
        \phi_{\ell}^{\zeta, \alpha}(r) = \sum_{p}\gamma_{p}^{\zeta,\alpha} u_{\ell \nu_p}^{\alpha}(r)  
    \f]
    
    Arbitrary number of local orbitals may be introduced for each angular quantum number.

    Radial functions are m-th order (with zero-order being a function itself) energy derivatives of the radial 
    Schrödinger equation:
    \f[
        u_{\ell \nu}^{\alpha}(r) = \frac{\partial^{m_{\nu}}}{\partial^{m_{\nu}}E}u_{\ell}^{\alpha}(r,E)\Big|_{E=E_{\nu}}
    \f]
*/
struct apwlo_basis_descriptor
{
    int igk;
    int ig;
    int ia;
    int l;
    int lm;
    int order;
    int idxrf;
};

class Band
{
    private:

        /// Global set of parameters
        Global& parameters_;
    
        /// Non-zero Gaunt coefficients
        mdarray<std::vector< std::pair<int, complex16> >, 2> complex_gaunt_packed_;
       
        /// Block-cyclic distribution of the first-variational states along columns of the MPI grid
        splindex<block_cyclic, scalapack_nb> spl_fv_states_col_;
        
        splindex<block_cyclic, scalapack_nb> spl_fv_states_row_;
        
        splindex<block_cyclic, scalapack_nb> spl_spinor_wf_col_;
        
        splindex<block> sub_spl_spinor_wf_;
        
        int num_fv_states_row_up_;
        
        int num_fv_states_row_dn_;

        int rank_col_;

        int num_ranks_col_;

        int dim_col_;

        int rank_row_;

        int num_ranks_row_;

        int dim_row_;

        int num_ranks_;
        
        /// BLACS communication context
        int blacs_context_;
        
        template <typename T>
        inline void sum_L3_complex_gaunt(int lm1, int lm2, T* v, complex16& zsum)
        {
            for (int k = 0; k < (int)complex_gaunt_packed_(lm1, lm2).size(); k++)
                zsum += complex_gaunt_packed_(lm1, lm2)[k].second * v[complex_gaunt_packed_(lm1, lm2)[k].first];
        }

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
        template <spin_block sblock>
        void apply_hmt_to_apw(int num_gkvec_row, mdarray<complex16, 2>& apw, mdarray<complex16, 2>& hapw)
        {
            Timer t("sirius::Band::apply_hmt_to_apw");
           
            #pragma omp parallel default(shared)
            {
                std::vector<complex16> zv(num_gkvec_row);
                
                #pragma omp for
                for (int ia = 0; ia < parameters_.num_atoms(); ia++)
                {
                    Atom* atom = parameters_.atom(ia);
                    AtomType* type = atom->type();

                    for (int j2 = 0; j2 < type->mt_aw_basis_size(); j2++)
                    {
                        memset(&zv[0], 0, num_gkvec_row * sizeof(complex16));

                        int lm2 = type->indexb(j2).lm;
                        int idxrf2 = type->indexb(j2).idxrf;

                        for (int j1 = 0; j1 < type->mt_aw_basis_size(); j1++)
                        {
                            int lm1 = type->indexb(j1).lm;
                            int idxrf1 = type->indexb(j1).idxrf;
                            
                            complex16 zsum(0.0, 0.0);
                            
                            if (sblock == nm) 
                                sum_L3_complex_gaunt(lm1, lm2, atom->h_radial_integral(idxrf1, idxrf2), zsum);
                            
                            if (abs(zsum) > 1e-14) 
                            {
                                for (int ig = 0; ig < num_gkvec_row; ig++) 
                                    zv[ig] += zsum * apw(ig, atom->offset_aw() + j1); 
                            }
                        } // j1
                         
                        if (sblock != ud)
                        {
                            int l2 = type->indexb(j2).l;
                            int order2 = type->indexb(j2).order;
                            
                            for (int order1 = 0; order1 < (int)type->aw_descriptor(l2).size(); order1++)
                            {
                                double t1 = 0.5 * pow(type->mt_radius(), 2) * 
                                            atom->symmetry_class()->aw_surface_dm(l2, order1, 0) * 
                                            atom->symmetry_class()->aw_surface_dm(l2, order2, 1);
                                
                                for (int ig = 0; ig < num_gkvec_row; ig++) 
                                    zv[ig] += t1 * apw(ig, atom->offset_aw() + type->indexb_by_lm_order(lm2, order1));
                            }
                        }

                        memcpy(&hapw(0, atom->offset_aw() + j2), &zv[0], num_gkvec_row * sizeof(complex16));

                    } // j2
                }
            }
        }

        /// Setup the Hamiltonian matrix in APW+lo basis

        /** The Hamiltonian matrix has the following expression:
            \f[
                H_{\mu' \mu} = \langle \varphi_{\mu'} | \hat H | \varphi_{\mu} \rangle
            \f]

            \f[
                H_{\mu' \mu}=\langle \varphi_{\mu' } | \hat H | \varphi_{\mu } \rangle  = 
                \left( \begin{array}{cc} 
                   H_{\bf G'G} & H_{{\bf G'}j} \\
                   H_{j'{\bf G}} & H_{j'j}
                \end{array} \right)
            \f]
        */
        template <spin_block sblock> 
        void set_fv_h(const std::vector<apwlo_basis_descriptor>& apwlo_basis_descriptors_row,
                      const int                                  apwlo_basis_size_row,
                      const int                                  num_gkvec_row,
                      const std::vector<apwlo_basis_descriptor>& apwlo_basis_descriptors_col,
                      const int                                  apwlo_basis_size_col,
                      const int                                  num_gkvec_col,
                      const int                                  apw_offset_col,
                      mdarray<double, 2>&                        gkvec,
                      mdarray<complex16, 2>&                     apw, 
                      PeriodicFunction<double>*                  effective_potential,
                      mdarray<complex16, 2>&                     h)
        {
            Timer t("sirius::Band::set_h");

            mdarray<complex16, 2> hapw(num_gkvec_row, parameters_.mt_aw_basis_size());

            apply_hmt_to_apw<sblock>(num_gkvec_row, apw, hapw);

            // apw-apw block
            gemm<cpu>(0, 2, num_gkvec_row, num_gkvec_col, parameters_.mt_aw_basis_size(), complex16(1, 0), 
                      &hapw(0, 0), hapw.ld(), &apw(apw_offset_col, 0), apw.ld(), complex16(0, 0), &h(0, 0), h.ld());

            // apw-lo block
            for (int icol = num_gkvec_col; icol < apwlo_basis_size_col; icol++)
            {
                int ia = apwlo_basis_descriptors_col[icol].ia;
                Atom* atom = parameters_.atom(ia);
                AtomType* type = atom->type();

                int lm = apwlo_basis_descriptors_col[icol].lm;
                int idxrf = apwlo_basis_descriptors_col[icol].idxrf;
                
                for (int j1 = 0; j1 < type->mt_aw_basis_size(); j1++) 
                {
                    int lm1 = type->indexb(j1).lm;
                    int idxrf1 = type->indexb(j1).idxrf;
                            
                    complex16 zsum(0, 0);
                            
                    if (sblock == nm)
                        sum_L3_complex_gaunt(lm1, lm, atom->h_radial_integral(idxrf, idxrf1), zsum);
        
                    if (abs(zsum) > 1e-14)
                    {
                        for (int igkloc = 0; igkloc < num_gkvec_row; igkloc++)
                            h(igkloc, icol) += zsum * apw(igkloc, atom->offset_aw() + j1);
                    }
                }
            }

            // lo-apw block
            std::vector<complex16> ztmp(num_gkvec_col);
            for (int irow = num_gkvec_row; irow < apwlo_basis_size_row; irow++)
            {
                int ia = apwlo_basis_descriptors_row[irow].ia;
                Atom* atom = parameters_.atom(ia);
                AtomType* type = atom->type();

                int lm = apwlo_basis_descriptors_row[irow].lm;
                int idxrf = apwlo_basis_descriptors_row[irow].idxrf;

                memset(&ztmp[0], 0, num_gkvec_col * sizeof(complex16));
                
                for (int j1 = 0; j1 < type->mt_aw_basis_size(); j1++) 
                {
                    int lm1 = type->indexb(j1).lm;
                    int idxrf1 = type->indexb(j1).idxrf;
                            
                    complex16 zsum(0, 0);
                            
                    if (sblock == nm)
                        sum_L3_complex_gaunt(lm, lm1, atom->h_radial_integral(idxrf, idxrf1), zsum);
        
                    if (abs(zsum) > 1e-14)
                    {
                        for (int igkloc = 0; igkloc < num_gkvec_col; igkloc++)
                            ztmp[igkloc] += zsum * conj(apw(apw_offset_col + igkloc, atom->offset_aw() + j1));
                    }
                }

                for (int igkloc = 0; igkloc < num_gkvec_col; igkloc++)
                    h(irow, igkloc) += ztmp[igkloc]; 
            }

            // lo-lo block
            for (int icol = num_gkvec_col; icol < apwlo_basis_size_col; icol++)
            {
                int ia = apwlo_basis_descriptors_col[icol].ia;
                int lm2 = apwlo_basis_descriptors_col[icol].lm; 
                int idxrf2 = apwlo_basis_descriptors_col[icol].idxrf; 

                for (int irow = num_gkvec_row; irow < apwlo_basis_size_row; irow++)
                    if (ia == apwlo_basis_descriptors_row[irow].ia)
                    {
                        Atom* atom = parameters_.atom(ia);
                        int lm1 = apwlo_basis_descriptors_row[irow].lm; 
                        int idxrf1 = apwlo_basis_descriptors_row[irow].idxrf; 

                        complex16 zsum(0, 0);
        
                        if (sblock == nm)
                            sum_L3_complex_gaunt(lm1, lm2, atom->h_radial_integral(idxrf1, idxrf2), zsum);

                        h(irow, icol) += zsum;
                    }
            }

            Timer *t1 = new Timer("sirius::Band::set_h:it");
            for (int igkloc2 = 0; igkloc2 < num_gkvec_col; igkloc2++) // loop over columns
            {
                double v2[3];
                double v2c[3];
                for (int x = 0; x < 3; x++) v2[x] = gkvec(x, apwlo_basis_descriptors_col[igkloc2].igk);
                parameters_.get_coordinates<cartesian, reciprocal>(v2, v2c);

                for (int igkloc1 = 0; igkloc1 < num_gkvec_row; igkloc1++) // for each column loop over rows
                {
                    int ig12 = parameters_.index_g12(apwlo_basis_descriptors_row[igkloc1].ig,
                                                     apwlo_basis_descriptors_col[igkloc2].ig);
                    double v1[3];
                    double v1c[3];
                    for (int x = 0; x < 3; x++) v1[x] = gkvec(x, apwlo_basis_descriptors_row[igkloc1].igk);
                    parameters_.get_coordinates<cartesian, reciprocal>(v1, v1c);
                    
                    double t1 = 0.5 * Utils::scalar_product(v1c, v2c);
                                       
                    if (sblock == nm)
                        h(igkloc1, igkloc2) += (effective_potential->f_pw(ig12) + 
                                                t1 * parameters_.step_function_pw(ig12));
                }
            }
            delete t1;

            if ((debug_level > 0) && (eigen_value_solver != scalapack))
            {
                // check hermiticity
                if (apwlo_basis_descriptors_row[0].igk == apwlo_basis_descriptors_col[0].igk)
                {
                    int n = std::min(apwlo_basis_size_col, apwlo_basis_size_row);

                    for (int i = 0; i < n; i++)
                        for (int j = 0; j < n; j++)
                            if (abs(h(j, i) - conj(h(i, j))) > 1e-12)
                            {
                                std::stringstream s;
                                s << "Hamiltonian matrix is not hermitian for the following elements : " 
                                  << i << " " << j << ", difference : " << abs(h(j, i) - conj(h(i, j)));
                                warning(__FILE__, __LINE__, s, 0);
                            }
                }
            }
        }
        
        /// Setup the overlap matrix in the APW+lo basis

        /** The overlap matrix has the following expression:
            \f[
                O_{\mu' \mu} = \langle \varphi_{\mu'} | \varphi_{\mu} \rangle
            \f]
            APW-APW block:
            \f[
                O_{{\bf G'} {\bf G}}^{\bf k} = \sum_{\alpha} \sum_{L\nu} a_{L\nu}^{\alpha *}({\bf G'+k}) 
                a_{L\nu}^{\alpha}({\bf G+k})
            \f]
            
            APW-lo block:
            \f[
                O_{{\bf G'} j}^{\bf k} = \sum_{\nu'} a_{\ell_j m_j \nu'}^{\alpha_j *}({\bf G'+k}) 
                \langle u_{\ell_j \nu'}^{\alpha_j} | \phi_{\ell_j}^{\zeta_j \alpha_j} \rangle
            \f]

            lo-APW block:
            \f[
                O_{j' {\bf G}}^{\bf k} = 
                \sum_{\nu'} \langle \phi_{\ell_{j'}}^{\zeta_{j'} \alpha_{j'}} | u_{\ell_{j'} \nu'}^{\alpha_{j'}} \rangle
                a_{\ell_{j'} m_{j'} \nu'}^{\alpha_{j'}}({\bf G+k}) 
            \f]

            lo-lo block:
            \f[
                O_{j' j}^{\bf k} = \langle \phi_{\ell_{j'}}^{\zeta_{j'} \alpha_{j'}} | 
                \phi_{\ell_{j}}^{\zeta_{j} \alpha_{j}} \rangle \delta_{\alpha_{j'} \alpha_j} 
                \delta_{\ell_{j'} \ell_j} \delta_{m_{j'} m_j}
            \f]

        */
        void set_fv_o(const std::vector<apwlo_basis_descriptor>& apwlo_basis_descriptors_row,
                      const int                                  apwlo_basis_size_row,
                      const int                                  num_gkvec_row,
                      const std::vector<apwlo_basis_descriptor>& apwlo_basis_descriptors_col,
                      const int                                  apwlo_basis_size_col,
                      const int                                  num_gkvec_col,
                      const int                                  apw_col_offset,
                      mdarray<complex16, 2>&                     apw, 
                      mdarray<complex16, 2>&                     o)
        {
            Timer t("sirius::Band::set_o");
            
            // compute APW-APW block
            gemm<cpu>(0, 2, num_gkvec_row, num_gkvec_col, parameters_.mt_aw_basis_size(), complex16(1, 0), 
                      &apw(0, 0), apw.ld(), &apw(apw_col_offset, 0), apw.ld(), complex16(0, 0), &o(0, 0), o.ld()); 

            // TODO: multithread 

            // apw-lo block 
            for (int icol = num_gkvec_col; icol < apwlo_basis_size_col; icol++)
            {
                int ia = apwlo_basis_descriptors_col[icol].ia;
                Atom* atom = parameters_.atom(ia);
                AtomType* type = atom->type();

                int l = apwlo_basis_descriptors_col[icol].l;
                int lm = apwlo_basis_descriptors_col[icol].lm;
                int order = apwlo_basis_descriptors_col[icol].order;

                for (int order1 = 0; order1 < (int)type->aw_descriptor(l).size(); order1++)
                    for (int igkloc = 0; igkloc < num_gkvec_row; igkloc++)
                        o(igkloc, icol) += atom->symmetry_class()->o_radial_integral(l, order1, order) * 
                                           apw(igkloc, atom->offset_aw() + type->indexb_by_lm_order(lm, order1));
            }

            // lo-apw block 
            for (int irow = num_gkvec_row; irow < apwlo_basis_size_row; irow++)
            {
                int ia = apwlo_basis_descriptors_row[irow].ia;
                Atom* atom = parameters_.atom(ia);
                AtomType* type = atom->type();

                int l = apwlo_basis_descriptors_row[irow].l;
                int lm = apwlo_basis_descriptors_row[irow].lm;
                int order = apwlo_basis_descriptors_row[irow].order;

                for (int order1 = 0; order1 < (int)type->aw_descriptor(l).size(); order1++)
                    for (int igkloc = 0; igkloc < num_gkvec_col; igkloc++)
                        o(irow, igkloc) += atom->symmetry_class()->o_radial_integral(l, order, order1) * 
                                           conj(apw(igkloc, atom->offset_aw() + type->indexb_by_lm_order(lm, order1)));
            }

            // lo-lo block
            for (int irow = num_gkvec_row; irow < apwlo_basis_size_row; irow++)
                for (int icol = num_gkvec_col; icol < apwlo_basis_size_col; icol++)
                    if ((apwlo_basis_descriptors_col[icol].ia == apwlo_basis_descriptors_row[irow].ia) &&
                        (apwlo_basis_descriptors_col[icol].lm == apwlo_basis_descriptors_row[irow].lm))
                    {
                        int ia = apwlo_basis_descriptors_row[irow].ia;
                        Atom* atom = parameters_.atom(ia);
                        int l = apwlo_basis_descriptors_row[irow].l;
                        int order1 = apwlo_basis_descriptors_row[irow].order; 
                        int order2 = apwlo_basis_descriptors_col[icol].order; 
                        o(irow, icol) += atom->symmetry_class()->o_radial_integral(l, order1, order2);
                    }
                    
            Timer t1("sirius::Band::set_o:it");
            for (int igkloc2 = 0; igkloc2 < num_gkvec_col; igkloc2++) // loop over columns
                for (int igkloc1 = 0; igkloc1 < num_gkvec_row; igkloc1++) // for each column loop over rows
                {
                    int ig12 = parameters_.index_g12(apwlo_basis_descriptors_row[igkloc1].ig,
                                                     apwlo_basis_descriptors_col[igkloc2].ig);
                    o(igkloc1, igkloc2) += parameters_.step_function_pw(ig12);
                }

            if ((debug_level > 0) && (eigen_value_solver != scalapack))
            {
                if (apwlo_basis_descriptors_row[0].igk == apwlo_basis_descriptors_col[0].igk)
                {
                    int n = std::min(apwlo_basis_size_col, apwlo_basis_size_row);

                    for (int i = 0; i < n; i++)
                        for (int j = 0; j < n; j++)
                            if (abs(o(j, i) - conj(o(i, j))) > 1e-12)
                                printf("Overlap matrix is not hermitian\n");
                }
            }
        }
        
        // assumes that hpsi is zero on input
        void apply_magnetic_field(mdarray<complex16, 2>& fv_states, int mtgk_size, int num_gkvec, int* fft_index, 
                                  PeriodicFunction<double>* effective_magnetic_field[3], mdarray<complex16, 3>& hpsi)
        {
            assert(hpsi.size(2) >= 2);
            assert(fv_states.size(0) == hpsi.size(0));
            assert(fv_states.size(1) == hpsi.size(1));

            Timer t("sirius::Band::apply_magnetic_field");

            complex16 zzero = complex16(0, 0);
            complex16 zone = complex16(1, 0);
            complex16 zi = complex16(0, 1);

            mdarray<complex16, 3> zm(parameters_.max_mt_basis_size(), parameters_.max_mt_basis_size(), 
                                     parameters_.num_mag_dims());
                    
            for (int ia = 0; ia < parameters_.num_atoms(); ia++)
            {
                int offset = parameters_.atom(ia)->offset_wf();
                int mt_basis_size = parameters_.atom(ia)->type()->mt_basis_size();
                
                zm.zero();
        
                for (int j2 = 0; j2 < mt_basis_size; j2++)
                {
                    int lm2 = parameters_.atom(ia)->type()->indexb(j2).lm;
                    int idxrf2 = parameters_.atom(ia)->type()->indexb(j2).idxrf;
                    
                    for (int i = 0; i < parameters_.num_mag_dims(); i++)
                    {
                        for (int j1 = 0; j1 <= j2; j1++)
                        {
                            int lm1 = parameters_.atom(ia)->type()->indexb(j1).lm;
                            int idxrf1 = parameters_.atom(ia)->type()->indexb(j1).idxrf;

                            sum_L3_complex_gaunt(lm1, lm2, parameters_.atom(ia)->b_radial_integral(idxrf1, idxrf2, i), 
                                                 zm(j1, j2, i));
                        }
                    }
                }
                // compute bwf = B_z*|wf_j>
                hemm<cpu>(0, 0, mt_basis_size, spl_fv_states_col_.local_size(), zone, &zm(0, 0, 0), zm.ld(), 
                          &fv_states(offset, 0), fv_states.ld(), zzero, &hpsi(offset, 0, 0), hpsi.ld());
                
                // compute bwf = (B_x - iB_y)|wf_j>
                if (hpsi.size(2) >= 3)
                {
                    // reuse first (z) component of zm matrix to store (Bx - iBy)
                    for (int j2 = 0; j2 < mt_basis_size; j2++)
                    {
                        for (int j1 = 0; j1 <= j2; j1++)
                            zm(j1, j2, 0) = zm(j1, j2, 1) - zi * zm(j1, j2, 2);
                        
                        for (int j1 = j2 + 1; j1 < mt_basis_size; j1++)
                            zm(j1, j2, 0) = conj(zm(j2, j1, 1)) - zi * conj(zm(j2, j1, 2));
                    }
                      
                    gemm<cpu>(0, 0, mt_basis_size, spl_fv_states_col_.local_size(), mt_basis_size, zone, &zm(0, 0, 0), 
                              zm.ld(), &fv_states(offset, 0), fv_states.ld(), zzero, &hpsi(offset, 0, 2), hpsi.ld());
                }
                
                // compute bwf = (B_x + iB_y)|wf_j>
                if ((hpsi.size(2)) == 4 && (eigen_value_solver == scalapack))
                {
                    // reuse first (z) component of zm matrix to store (Bx + iBy)
                    for (int j2 = 0; j2 < mt_basis_size; j2++)
                    {
                        for (int j1 = 0; j1 <= j2; j1++)
                            zm(j1, j2, 0) = zm(j1, j2, 1) + zi * zm(j1, j2, 2);
                        
                        for (int j1 = j2 + 1; j1 < mt_basis_size; j1++)
                            zm(j1, j2, 0) = conj(zm(j2, j1, 1)) + zi * conj(zm(j2, j1, 2));
                    }
                      
                    gemm<cpu>(0, 0, mt_basis_size, spl_fv_states_col_.local_size(), mt_basis_size, zone, &zm(0, 0, 0), 
                              zm.ld(), &fv_states(offset, 0), fv_states.ld(), zzero, &hpsi(offset, 0, 3), hpsi.ld());
                }
            }
            
            Timer *t1 = new Timer("sirius::Band::apply_magnetic_field:it");
            #pragma omp parallel default(shared)
            {        
                int thread_id = omp_get_thread_num();
                
                std::vector<complex16> psi_it(parameters_.fft().size());
                std::vector<complex16> hpsi_it(parameters_.fft().size());
                
                #pragma omp for
                for (int i = 0; i < spl_fv_states_col_.local_size(); i++)
                {
                    parameters_.fft().input(num_gkvec, fft_index, &fv_states(parameters_.mt_basis_size(), i), 
                                            thread_id);
                    parameters_.fft().transform(1, thread_id);
                    parameters_.fft().output(&psi_it[0], thread_id);
                                                
                    for (int ir = 0; ir < parameters_.fft().size(); ir++)
                        hpsi_it[ir] = psi_it[ir] * effective_magnetic_field[0]->f_it(ir) * 
                                      parameters_.step_function(ir);
                    
                    parameters_.fft().input(&hpsi_it[0], thread_id);
                    parameters_.fft().transform(-1, thread_id);
                    parameters_.fft().output(num_gkvec, fft_index, &hpsi(parameters_.mt_basis_size(), i, 0), thread_id); 

                    if (hpsi.size(2) >= 3)
                    {
                        for (int ir = 0; ir < parameters_.fft().size(); ir++)
                            hpsi_it[ir] = psi_it[ir] * (effective_magnetic_field[1]->f_it(ir) - 
                                                        zi * effective_magnetic_field[2]->f_it(ir)) * 
                                                       parameters_.step_function(ir);
                        
                        parameters_.fft().input(&hpsi_it[0], thread_id);
                        parameters_.fft().transform(-1, thread_id);
                        parameters_.fft().output(num_gkvec, fft_index, &hpsi(parameters_.mt_basis_size(), i, 2), 
                                                 thread_id); 
                    }
                    
                    if ((hpsi.size(2)) == 4 && (eigen_value_solver == scalapack))
                    {
                        for (int ir = 0; ir < parameters_.fft().size(); ir++)
                            hpsi_it[ir] = psi_it[ir] * (effective_magnetic_field[1]->f_it(ir) + 
                                                        zi * effective_magnetic_field[2]->f_it(ir)) * 
                                                       parameters_.step_function(ir);
                        
                        parameters_.fft().input(&hpsi_it[0], thread_id);
                        parameters_.fft().transform(-1, thread_id);
                        parameters_.fft().output(num_gkvec, fft_index, &hpsi(parameters_.mt_basis_size(), i, 3), 
                                                 thread_id); 
                    }
                }
            }
            delete t1;
           
            // copy Bz|\psi> to -Bz|\psi>
            for (int i = 0; i < spl_fv_states_col_.local_size(); i++)
                for (int j = 0; j < mtgk_size; j++)
                    hpsi(j, i, 1) = -hpsi(j, i, 0);
        }

        /// Apply SO correction to the scalar wave functions

        /** Raising lowering operators:
            \f[
                L_{\pm} Y_{\ell m}= (L_x \pm i L_y) Y_{\ell m}  = \sqrt{\ell(\ell+1) - m(m \pm 1)} Y_{\ell m \pm 1}
            \f]
        */
        void apply_so_correction(mdarray<complex16, 2>& fv_states, mdarray<complex16, 3>& hpsi)
        {
            Timer t("sirius::Band::apply_so_correction");

            for (int ia = 0; ia < parameters_.num_atoms(); ia++)
            {
                AtomType* type = parameters_.atom(ia)->type();

                int offset = parameters_.atom(ia)->offset_wf();

                for (int l = 0; l <= parameters_.lmax_apw(); l++)
                {
                    int nrf = type->indexr().num_rf(l);

                    for (int order1 = 0; order1 < nrf; order1++)
                    {
                        for (int order2 = 0; order2 < nrf; order2++)
                        {
                            double sori = parameters_.atom(ia)->symmetry_class()->so_radial_integral(l, order1, order2);
                            
                            for (int m = -l; m <= l; m++)
                            {
                                int idx1 = type->indexb_by_l_m_order(l, m, order1);
                                int idx2 = type->indexb_by_l_m_order(l, m, order2);
                                int idx3 = (m + l != 0) ? type->indexb_by_l_m_order(l, m - 1, order2) : 0;
                                int idx4 = (m - l != 0) ? type->indexb_by_l_m_order(l, m + 1, order2) : 0;

                                for (int ist = 0; ist < spl_fv_states_col_.local_size(); ist++)
                                {
                                    complex16 z1 = fv_states(offset + idx2, ist) * double(m) * sori;
                                    hpsi(offset + idx1, ist, 0) += z1;
                                    hpsi(offset + idx1, ist, 1) -= z1;
                                    // apply L_{-} operator
                                    if (m + l) hpsi(offset + idx1, ist, 2) += fv_states(offset + idx3, ist) * sori * 
                                                                              sqrt(double(l * (l + 1) - m * (m - 1)));
                                    // apply L_{+} operator
                                    if (m - l) hpsi(offset + idx1, ist, 3) += fv_states(offset + idx4, ist) * sori * 
                                                                              sqrt(double(l * (l + 1) - m * (m + 1)));
                                }
                            }
                        }
                    }
                }
            }
        }

        /// Apply UJ correction to scalar wave functions
        template <spin_block sblock>
        void apply_uj_correction(mdarray<complex16, 2>& fv_states, mdarray<complex16, 3>& hpsi)
        {
            Timer t("sirius::Band::apply_uj_correction");

            for (int ia = 0; ia < parameters_.num_atoms(); ia++)
            {
                if (parameters_.atom(ia)->apply_uj_correction())
                {
                    AtomType* type = parameters_.atom(ia)->type();

                    int offset = parameters_.atom(ia)->offset_wf();

                    int l = parameters_.atom(ia)->uj_correction_l();

                    int nrf = type->indexr().num_rf(l);

                    for (int order2 = 0; order2 < nrf; order2++)
                    {
                        for (int lm2 = Utils::lm_by_l_m(l, -l); lm2 <= Utils::lm_by_l_m(l, l); lm2++)
                        {
                            int idx2 = type->indexb_by_lm_order(lm2, order2);
                            for (int order1 = 0; order1 < nrf; order1++)
                            {
                                double ori = parameters_.atom(ia)->symmetry_class()->o_radial_integral(l, order2, order1);
                                
                                for (int ist = 0; ist < spl_fv_states_col_.local_size(); ist++)
                                {
                                    for (int lm1 = Utils::lm_by_l_m(l, -l); lm1 <= Utils::lm_by_l_m(l, l); lm1++)
                                    {
                                        int idx1 = type->indexb_by_lm_order(lm1, order1);
                                        complex16 z1 = fv_states(offset + idx1, ist) * ori;

                                        if (sblock == uu)
                                            hpsi(offset + idx2, ist, 0) += z1 * 
                                                parameters_.atom(ia)->uj_correction_matrix(lm2, lm1, 0, 0);

                                        if (sblock == dd)
                                            hpsi(offset + idx2, ist, 1) += z1 *
                                                parameters_.atom(ia)->uj_correction_matrix(lm2, lm1, 1, 1);

                                        if (sblock == ud)
                                            hpsi(offset + idx2, ist, 2) += z1 *
                                                parameters_.atom(ia)->uj_correction_matrix(lm2, lm1, 0, 1);
                                        
                                        if (sblock == du)
                                            hpsi(offset + idx2, ist, 3) += z1 *
                                                parameters_.atom(ia)->uj_correction_matrix(lm2, lm1, 1, 0);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        void init_blacs_context()
        {
            mdarray<int, 2> map_ranks(num_ranks_row_, num_ranks_col_);
            int rc = (1 << dim_row_) | 1 << (dim_col_);
            for (int i1 = 0; i1 < num_ranks_col_; i1++)
                for (int i0 = 0; i0 < num_ranks_row_; i0++)
                    map_ranks(i0, i1) = parameters_.mpi_grid().cart_rank(parameters_.mpi_grid().communicator(rc), 
                                                                         intvec(i0, i1));
 
            // create BLACS context
            blacs_context_ = Csys2blacs_handle(parameters_.mpi_grid().communicator(rc));

            // create grid of MPI ranks 
            Cblacs_gridmap(&blacs_context_, map_ranks.get_ptr(), map_ranks.ld(), num_ranks_row_, num_ranks_col_);

            // check the grid
            int nrow, ncol, irow, icol;
            Cblacs_gridinfo(blacs_context_, &nrow, &ncol, &irow, &icol);
            if ((rank_row_ != irow) || (rank_col_ != icol) || (num_ranks_row_ != nrow) || (num_ranks_col_ != ncol)) 
                error(__FILE__, __LINE__, "wrong grid", fatal_err);
        }

        void init()
        {
            complex_gaunt_packed_.set_dimensions(parameters_.lmmax_apw(), parameters_.lmmax_apw());
            complex_gaunt_packed_.allocate();

            for (int l1 = 0; l1 <= parameters_.lmax_apw(); l1++) 
            for (int m1 = -l1; m1 <= l1; m1++)
            {
                int lm1 = Utils::lm_by_l_m(l1, m1);
                for (int l2 = 0; l2 <= parameters_.lmax_apw(); l2++)
                for (int m2 = -l2; m2 <= l2; m2++)
                {
                    int lm2 = Utils::lm_by_l_m(l2, m2);
                    for (int l3 = 0; l3 <= parameters_.lmax_pot(); l3++)
                    for (int m3 = -l3; m3 <= l3; m3++)
                    {
                        int lm3 = Utils::lm_by_l_m(l3, m3);
                        complex16 z = SHT::complex_gaunt(l1, l3, l2, m1, m3, m2);
                        if (abs(z) > 1e-12) complex_gaunt_packed_(lm1, lm2).push_back(std::pair<int,complex16>(lm3, z));
                    }
                }
            }
            
            // distribue first-variational states
            spl_fv_states_col_.split(parameters_.num_fv_states(), num_ranks_col_, rank_col_);
           
            spl_spinor_wf_col_.split(parameters_.num_bands(), num_ranks_col_, rank_col_);
            
            // split along rows 
            sub_spl_spinor_wf_.split(spl_spinor_wf_col_.local_size(), num_ranks_row_, rank_row_);
            
            if (parameters_.num_mag_dims() != 3)
            {
                spl_fv_states_row_.split(parameters_.num_fv_states(), num_ranks_row_, rank_row_);
                
                num_fv_states_row_up_ = num_fv_states_row_dn_ = spl_fv_states_row_.local_size();
            }
            else
            {
                spl_fv_states_row_.split(parameters_.num_fv_states() * parameters_.num_spins(), num_ranks_row_, 
                                         rank_row_);

                num_fv_states_row_up_ = 0;
                num_fv_states_row_dn_ = 0;

                for (int i = 0; i < spl_fv_states_row_.local_size(); i++)
                {
                    int j = spl_fv_states_row_[i];
                    if (j < parameters_.num_fv_states()) 
                    {
                        num_fv_states_row_up_++;
                    }
                    else
                    {
                        num_fv_states_row_dn_++;
                    }
                }
            }

            // check if the distribution of fv states is consistent with the distribtion of spinor wave functions
            for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
                for (int i = 0; i < spl_fv_states_col_.local_size(); i++)
                    if (spl_spinor_wf_col_[i + ispn * spl_fv_states_col_.local_size()] != 
                        (spl_fv_states_col_[i] + ispn * parameters_.num_fv_states()))
                        error(__FILE__, __LINE__, "Wrong distribution of wave-functions");
            
            if ((verbosity_level > 0) && (Platform::mpi_rank() == 0))
            {
                printf("\n");
                printf("table of column distribution of first-variational states\n");
                printf("(columns of the table correspond to MPI ranks)\n");
                for (int i0 = 0; i0 < spl_fv_states_col_.local_size(0); i0++)
                {
                    for (int i1 = 0; i1 < num_ranks_col_; i1++)
                        printf("%6i", spl_fv_states_col_.global_index(i0, i1));
                    printf("\n");
                }
                
                /*printf("\n");
                printf("table of row distribution of first-variational states\n");
                printf("columns of the table correspond to MPI ranks\n");
                for (int i0 = 0; i0 < fv_states_distribution_row_.size(0); i0++)
                {
                    for (int i1 = 0; i1 < fv_states_distribution_row_.size(1); i1++)
                        printf("%6i", fv_states_distribution_row_(i0, i1));
                    printf("\n");
                }*/

                printf("\n");
                printf("First-variational states index -> (local index, rank) for column distribution\n");
                for (int i = 0; i < parameters_.num_fv_states(); i++)
                    printf("%6i -> (%6i %6i)\n", i, spl_fv_states_col_.location(0, i), 
                                                    spl_fv_states_col_.location(1, i));
                
                printf("\n");
                printf("table of column distribution of spinor wave functions\n");
                printf("(columns of the table correspond to MPI ranks)\n");
                for (int i0 = 0; i0 < spl_spinor_wf_col_.local_size(0); i0++)
                {
                    for (int i1 = 0; i1 < num_ranks_row_; i1++)
                        printf("%6i", spl_spinor_wf_col_.global_index(i0, i1));
                    printf("\n");
                }
            }
        }
 
    public:
        
        /// Constructor
        Band(Global& parameters__) : parameters_(parameters__), blacs_context_(-1)
        {
            if (!parameters_.initialized()) error(__FILE__, __LINE__, "Parameters are not initialized.");

            dim_row_ = 1;
            dim_col_ = 2;

            num_ranks_row_ = parameters_.mpi_grid().dimension_size(dim_row_);
            num_ranks_col_ = parameters_.mpi_grid().dimension_size(dim_col_);

            num_ranks_ = num_ranks_row_ * num_ranks_col_;

            rank_row_ = parameters_.mpi_grid().coordinate(dim_row_);
            rank_col_ = parameters_.mpi_grid().coordinate(dim_col_);

            init();
            
            if (eigen_value_solver == scalapack) init_blacs_context();
        }

        ~Band()
        {
            if (eigen_value_solver == scalapack) Cfree_blacs_system_handle(blacs_context_);
        }
      
        /// Setup and solve the first-variational problem

        /** Solve \$ H\psi = E\psi \$.

            \param [in] parameters Global variables
            \param [in] fv_basis_size Size of the first-variational (APW) basis
            \param [in] num_gkvec Total number of G+k vectors

        */
        void solve_fv(Global&                                    parameters,
                      const int                                  apwlo_basis_size, 
                      const int                                  num_gkvec,
                      const std::vector<apwlo_basis_descriptor>& apwlo_basis_descriptors_row,
                      const int                                  apwlo_basis_size_row,
                      const int                                  num_gkvec_row,
                      const std::vector<apwlo_basis_descriptor>& apwlo_basis_descriptors_col,
                      const int                                  apwlo_basis_size_col,
                      const int                                  num_gkvec_col,
                      mdarray<complex16, 2>&                     matching_coefficients, 
                      mdarray<double, 2>&                        gkvec,
                      PeriodicFunction<double>*                  effective_potential, 
                      PeriodicFunction<double>*                  effective_magnetic_field[3], 
                      std::vector<double>&                       fv_eigen_values, 
                      mdarray<complex16, 2>&                     fv_eigen_vectors)
        {
            if (&parameters != &parameters_) error(__FILE__, __LINE__, "different set of parameters");

            if ((eigen_value_solver != scalapack) && (num_ranks_ > 1))
                error(__FILE__, __LINE__, "Only one MPI rank is allowed."); 

            Timer t("sirius::Band::solve_fv");

            int apw_offset_col = (num_ranks_ > 1) ? num_gkvec_row : 0;

            mdarray<complex16, 2> h(apwlo_basis_size_row, apwlo_basis_size_col);
            mdarray<complex16, 2> o(apwlo_basis_size_row, apwlo_basis_size_col);

            o.zero();
            set_fv_o(apwlo_basis_descriptors_row, apwlo_basis_size_row, num_gkvec_row, 
                     apwlo_basis_descriptors_col, apwlo_basis_size_col, num_gkvec_col, 
                     apw_offset_col, matching_coefficients, o);

            h.zero();
            set_fv_h<nm>(apwlo_basis_descriptors_row, apwlo_basis_size_row, num_gkvec_row,
                         apwlo_basis_descriptors_col, apwlo_basis_size_col, num_gkvec_col,
                         apw_offset_col, gkvec, matching_coefficients, effective_potential, h);

            Timer *t1 = new Timer("sirius::Band::solve_fv:hegv");

            generalized_eigenproblem<eigen_value_solver>(apwlo_basis_size, 
                                                         num_ranks_row(), 
                                                         num_ranks_col(), 
                                                         blacs_context_, 
                                                         parameters_.num_fv_states(), 
                                                         -1.0, 
                                                         h.get_ptr(), 
                                                         h.ld(), 
                                                         o.get_ptr(), 
                                                         o.ld(), 
                                                         &fv_eigen_values[0], 
                                                         fv_eigen_vectors.get_ptr(),
                                                         fv_eigen_vectors.ld());
            delete t1;
        }

        void solve_sv(Global&                   parameters,
                      int                       mtgk_size, 
                      int                       num_gkvec,
                      int*                      fft_index, 
                      double*                   evalfv, 
                      mdarray<complex16, 2>&    fv_states_row, 
                      mdarray<complex16, 2>&    fv_states_col, 
                      PeriodicFunction<double>* effective_magnetic_field[3],
                      double*                   band_energies,
                      mdarray<complex16, 2>&    sv_eigen_vectors)

        {
            if (&parameters != &parameters_) error(__FILE__, __LINE__, "different set of parameters");

            Timer t("sirius::Band::solve_sv");

            // number of h|\psi> components 
            int nhpsi = parameters_.num_mag_dims() + 1;

            // product of the second-variational Hamiltonian and a wave-function
            mdarray<complex16, 3> hpsi(mtgk_size, spl_fv_states_col_.local_size(), nhpsi);
            hpsi.zero();

            // compute product of magnetic field and wave-function 
            if (parameters_.num_spins() == 2)
                apply_magnetic_field(fv_states_col, mtgk_size, num_gkvec, fft_index, effective_magnetic_field, hpsi);

            if (parameters_.uj_correction())
            {
                apply_uj_correction<uu>(fv_states_col, hpsi);
                if (parameters_.num_mag_dims() != 0) apply_uj_correction<dd>(fv_states_col, hpsi);
                if (parameters_.num_mag_dims() == 3) apply_uj_correction<ud>(fv_states_col, hpsi);
                if ((parameters_.num_mag_dims() == 3) && (eigen_value_solver == scalapack)) 
                    apply_uj_correction<du>(fv_states_col, hpsi);
            }

            if (parameters_.so_correction()) apply_so_correction(fv_states_col, hpsi);

            Timer t1("sirius::Band::solve_sv:heev", false);
            
            if (parameters.num_mag_dims() == 1)
            {
                mdarray<complex16, 2> h(spl_fv_states_row_.local_size(), spl_fv_states_col_.local_size());

                //perform two consecutive diagonalizations
                for (int ispn = 0; ispn < 2; ispn++)
                {
                    // compute <wf_i | (h * wf_j)> for up-up or dn-dn block
                    gemm<cpu>(2, 0, spl_fv_states_row_.local_size(), spl_fv_states_col_.local_size(), 
                              mtgk_size, complex16(1, 0), &fv_states_row(0, 0), fv_states_row.ld(), 
                              &hpsi(0, 0, ispn), hpsi.ld(), complex16(0, 0), &h(0, 0), h.ld());

                    for (int icol = 0; icol < spl_fv_states_col_.local_size(); icol++)
                    {
                        int i = spl_fv_states_col_[icol];
                        for (int irow = 0; irow < spl_fv_states_row_.local_size(); irow++)
                            if (spl_fv_states_row_[irow] == i) h(irow, icol) += evalfv[i];
                    }
                
                    t1.start();
                    standard_eigenproblem<eigen_value_solver>(parameters_.num_fv_states(), 
                                                              num_ranks_row(), 
                                                              num_ranks_col(), 
                                                              blacs_context_, 
                                                              h.get_ptr(), 
                                                              h.ld(),
                                                              &band_energies[ispn * parameters_.num_fv_states()],
                                                              &sv_eigen_vectors(0, ispn * spl_fv_states_col_.local_size()),
                                                              sv_eigen_vectors.ld());
                    t1.stop();
                }
            }

            if (parameters.num_mag_dims() == 3)
            {
                mdarray<complex16, 2> h(spl_fv_states_row_.local_size(), spl_spinor_wf_col_.local_size());
                h.zero();

                // compute <wf_i | (h * wf_j)> for up-up block
                gemm<cpu>(2, 0, num_fv_states_row_up_, spl_fv_states_col_.local_size(), mtgk_size, complex16(1, 0), 
                          &fv_states_row(0, 0), fv_states_row.ld(), &hpsi(0, 0, 0), hpsi.ld(), complex16(0, 0), 
                          &h(0, 0), h.ld());

                // compute <wf_i | (h * wf_j)> for up-dn block
                gemm<cpu>(2, 0, num_fv_states_row_up_, spl_fv_states_col_.local_size(), mtgk_size, complex16(1, 0), 
                          &fv_states_row(0, 0), fv_states_row.ld(), &hpsi(0, 0, 2), hpsi.ld(), complex16(0, 0), 
                          &h(0, spl_fv_states_col_.local_size()), h.ld());
               
                int fv_states_up_offset = (num_ranks_ == 1) ? 0 : num_fv_states_row_up_;
                // compute <wf_i | (h * wf_j)> for dn-dn block
                gemm<cpu>(2, 0, num_fv_states_row_dn_, spl_fv_states_col_.local_size(), mtgk_size, complex16(1, 0), 
                          &fv_states_row(0, fv_states_up_offset), fv_states_row.ld(), &hpsi(0, 0, 1), hpsi.ld(), 
                          complex16(0, 0), &h(num_fv_states_row_up_, spl_fv_states_col_.local_size()), h.ld());

                if (eigen_value_solver == scalapack)
                {
                    // compute <wf_i | (h * wf_j)> for dn-up block
                    gemm<cpu>(2, 0, num_fv_states_row_dn_, spl_fv_states_col_.local_size(), mtgk_size, complex16(1, 0), 
                              &fv_states_row(0, fv_states_up_offset), fv_states_row.ld(), &hpsi(0, 0, 3), hpsi.ld(), 
                              complex16(0, 0), &h(num_fv_states_row_up_, 0), h.ld());
                }
              
                for (int ispn = 0; ispn < 2; ispn++)
                {
                    for (int icol = 0; icol < spl_fv_states_col_.local_size(); icol++)
                    {
                        int i = spl_fv_states_col_[icol] + ispn * parameters_.num_fv_states();
                        for (int irow = 0; irow < spl_fv_states_row_.local_size(); irow++)
                            if (spl_fv_states_row_[irow] == i) 
                                h(irow, icol + ispn * spl_fv_states_col_.local_size()) += 
                                    evalfv[spl_fv_states_col_[icol]];
                    }
                }
            
                t1.start();
                standard_eigenproblem<eigen_value_solver>(parameters_.num_bands(), 
                                                          num_ranks_row(), 
                                                          num_ranks_col(), 
                                                          blacs_context_, 
                                                          h.get_ptr(), 
                                                          h.ld(),
                                                          &band_energies[0], 
                                                          sv_eigen_vectors.get_ptr(),
                                                          sv_eigen_vectors.ld());
                t1.stop();
            }
        }
        
        inline splindex<block_cyclic, scalapack_nb>& spl_fv_states_col()
        {
            return spl_fv_states_col_;
        }
        
        inline splindex<block_cyclic, scalapack_nb>& spl_fv_states_row()
        {
            return spl_fv_states_row_;
        }
        
        inline splindex<block_cyclic, scalapack_nb>& spl_spinor_wf_col()
        {
            return spl_spinor_wf_col_;
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

        inline int num_fv_states_row_up()
        {
            return num_fv_states_row_up_;
        }

        inline int num_fv_states_row_dn()
        {
            return num_fv_states_row_dn_;
        }

        inline int num_fv_states_row(int ispn)
        {
            assert((ispn == 0) || (ispn == 1));

            return (ispn == 0) ? num_fv_states_row_up_ : num_fv_states_row_dn_;
        }

        inline int offs_fv_states_row(int ispn)
        {
            assert((ispn == 0) || (ispn == 1));
            
            if (parameters_.num_mag_dims() != 3) return 0;
            if (num_ranks_ == 1) return 0;
            return (ispn == 0) ? 0 : num_fv_states_row_up_;
        }    

        inline int dim_row()
        {
            return dim_row_;
        }
        
        inline int dim_col()
        {
            return dim_col_;
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
};

};
