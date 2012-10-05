
namespace sirius
{

void write_h_o(mdarray<complex16,2>& h, mdarray<complex16,2>& o)
{
    std::ofstream out;
    out.open("h_sirius.txt");
    out << h.size(0) << std::endl;
    out.precision(16); 
    for (int i = 0; i < h.size(0); i++)
    {
        for (int j = 0; j <= i; j++)
        {
            out << real(h(j, i)) << " " << imag(h(j, i)) << std::endl;
        }
    }
    out.close();
    
    out.open("o_sirius.txt");
    out << h.size(0) << std::endl;
    out.precision(16); 
    for (int i = 0; i < h.size(0); i++)
    {
        for (int j = 0; j <= i; j++)
        {
            out << real(o(j, i)) << " " << imag(o(j, i)) << std::endl;
        }
    }
    out.close();
}

class Band
{
    public:

        void radial()
        {
            Timer t("sirius::Band::radial");

            // save spherical part of the potential
            for (int ic = 0; ic < global.num_atom_symmetry_classes(); ic++)
            {
               int ia = global.atom_symmetry_class(ic)->atom_id(0);
               int nmtp = global.atom(ia)->type()->num_mt_points();
               
               std::vector<double> veff(nmtp);
               
               for (int ir = 0; ir < nmtp; ir++)
                   veff[ir] = y00 * global.effective_potential().f_rlm(0, ir, ia);

               global.atom_symmetry_class(ic)->set_spherical_potential(veff);
            }

            for (int ic = 0; ic < global.num_atom_symmetry_classes(); ic++)
            {
                global.atom_symmetry_class(ic)->generate_radial_functions();
                global.atom_symmetry_class(ic)->generate_radial_integrals();
            }

            for (int ia = 0; ia < global.num_atoms(); ia++)
                global.atom(ia)->generate_radial_integrals(global.lmax_pot(), 
                                                           &global.effective_potential().f_rlm(0, 0, ia));
        }
        
        /*! \brief Apply the muffin-tin part of the first-variational Hamiltonian to the
                   apw basis function
            
            The following vector is computed:
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
        void apply_hmt_to_apw(kpoint_data_set& kp, mdarray<complex16,2>& hapw)
        {
            Timer t("sirius::Band::apply_hmt_to_apw");
           
            #pragma omp parallel default(shared)
            {
                std::vector<complex16> zv(kp.num_gkvec());
                
                #pragma omp for
                for (int ia = 0; ia < global.num_atoms(); ia++)
                {
                    Atom* atom = global.atom(ia);
                    AtomType* type = atom->type();

                    for (int j2 = 0; j2 < type->mt_aw_basis_size(); j2++)
                    {
                        memset(&zv[0], 0, kp.num_gkvec() * sizeof(complex16));

                        int lm2 = type->indexb(j2).lm;
                        int idxrf2 = type->indexb(j2).idxrf;

                        for (int j1 = 0; j1 < type->mt_aw_basis_size(); j1++)
                        {
                            int lm1 = type->indexb(j1).lm;
                            int idxrf1 = type->indexb(j1).idxrf;
                            
                            complex16 zsum(0.0, 0.0);
                            
                            if (sblock == nm)
                                global.sum_L3_complex_gaunt(lm1, lm2, atom->h_radial_integral(idxrf1, idxrf2), zsum);
                            
                            if (abs(zsum) > 1e-14) 
                                for (int ig = 0; ig < kp.num_gkvec(); ig++) 
                                    zv[ig] += zsum * kp.matching_coefficient(ig, atom->offset_aw() + j1); 
                        } // j1
                         
                        if (sblock != ud)
                        {
                            int l2 = type->indexb(j2).l;
                            int order2 = type->indexb(j2).order;
                            
                            for (int order1 = 0; order1 < (int)type->aw_descriptor(l2).size(); order1++)
                            {
                                double t1 = 0.5 * pow(type->mt_radius(), 2) * atom->symmetry_class()->aw_surface_dm(l2, order1, 0) * 
                                    atom->symmetry_class()->aw_surface_dm(l2, order2, 1);
                                
                                for (int ig = 0; ig < kp.num_gkvec(); ig++) 
                                    zv[ig] += t1 * kp.matching_coefficient(ig, atom->offset_aw() + type->indexb_by_lm_order(lm2, order1));
                            }
                        }

                        memcpy(&hapw(0, atom->offset_aw() + j2), &zv[0], kp.num_gkvec() * sizeof(complex16));

                    } // j2
                }
            }
 #if 0           
            #pragma omp parallel default(shared)
            {
                std::vector<complex16> zv(ks->ngk);
                std::vector<double> v1(lapw_global.lmmaxvr);
                std::vector<complex16> v2(lapw_global.lmmaxvr);
                #pragma omp for
                for (int ias = 0; ias < (int)lapw_global.atoms.size(); ias++)
                {
                    Atom *atom = lapw_global.atoms[ias];
                    Species *species = atom->species;
                
                    // precompute apw block
                    for (int j2 = 0; j2 < (int)species->index.apw_size(); j2++)
                    {
                        memset(&zv[0], 0, ks->ngk * sizeof(complex16));
                        
                        int lm2 = species->index[j2].lm;
                        int idxrf2 = species->index[j2].idxrf;
                        
                        for (int j1 = 0; j1 < (int)species->index.apw_size(); j1++)
                        {
                            int lm1 = species->index[j1].lm;
                            int idxrf1 = species->index[j1].idxrf;
                            
                            complex16 zsum(0, 0);
                            
                            if (sblock == nm)
                            {
                                L3_sum_gntyry(lm1, lm2, &lapw_runtime.hmltrad(0, idxrf1, idxrf2, ias), zsum);
                            }
        
                            if (sblock == uu)
                            {
                                for (int lm3 = 0; lm3 < lapw_global.lmmaxvr; lm3++) 
                                    v1[lm3] = lapw_runtime.hmltrad(lm3, idxrf1, idxrf2, ias) + lapw_runtime.beffrad(lm3, idxrf1, idxrf2, ias, 0);
                                L3_sum_gntyry(lm1, lm2, &v1[0], zsum);
                            }
                            
                            if (sblock == dd)
                            {
                                for (int lm3 = 0; lm3 < lapw_global.lmmaxvr; lm3++) 
                                    v1[lm3] = lapw_runtime.hmltrad(lm3, idxrf1, idxrf2, ias) - lapw_runtime.beffrad(lm3, idxrf1, idxrf2, ias, 0);
                                L3_sum_gntyry(lm1, lm2, &v1[0], zsum);
                            }
                            
                            if (sblock == ud)
                            {
                                for (int lm3 = 0; lm3 < lapw_global.lmmaxvr; lm3++) 
                                    v2[lm3] = complex16(lapw_runtime.beffrad(lm3, idxrf1, idxrf2, ias, 1), -lapw_runtime.beffrad(lm3, idxrf1, idxrf2, ias, 2));
                                L3_sum_gntyry(lm1, lm2, &v2[0], zsum);
                            }
                
                            if (abs(zsum) > 1e-14) 
                                for (int ig = 0; ig < ks->ngk; ig++) 
                                    zv[ig] += zsum * ks->apwalm(ig, atom->offset_apw + j1); 
                        }
                        
                        // surface term
                        if (sblock != ud)
                        {
                            int l2 = species->index[j2].l;
                            int io2 = species->index[j2].order;
                            
                            for (int io1 = 0; io1 < (int)species->apw_descriptors[l2].radial_solution_descriptors.size(); io1++)
                            {
                                double t1 = 0.5 * pow(species->rmt, 2) * lapw_runtime.apwfr(species->nrmt - 1, 0, io1, l2, ias) * lapw_runtime.apwdfr(io2, l2, ias); 
                                for (int ig = 0; ig < ks->ngk; ig++) 
                                    zv[ig] += t1 * ks->apwalm(ig, atom->offset_apw + species->index(lm2, io1));
                            }
                        }
        
                        memcpy(&hapw(0, atom->offset_apw + j2), &zv[0], ks->ngk * sizeof(complex16));
                    }
                } 
            }
#endif
        }

        template <spin_block sblock> 
        void set_h(kpoint_data_set& kp, mdarray<complex16,2>& h)
        {
            Timer t("sirius::Band::set_h");

            mdarray<complex16,2> hapw(kp.num_gkvec(), global.mt_aw_basis_size());

            apply_hmt_to_apw<sblock>(kp, hapw);

            gemm<cpu>(0, 2, kp.num_gkvec(), kp.num_gkvec(), global.mt_aw_basis_size(), complex16(1.0, 0.0), 
                &hapw(0, 0), hapw.size(0), &kp.matching_coefficient(0, 0), kp.num_gkvec(), complex16(0.0, 0.0), 
                &h(0, 0), h.size(0));

            #pragma omp parallel default(shared)
            {
                std::vector<double> v1(global.lmmax_pot());
                std::vector<complex16> v2(global.lmmax_pot());
                #pragma omp for
                for (int ia = 0; ia < global.num_atoms(); ia++)
                {
                    Atom* atom = global.atom(ia);
                    AtomType* type = atom->type();
            
                    int lo_index_offset = type->mt_aw_basis_size();
                    
                    for (int j2 = 0; j2 < type->mt_lo_basis_size(); j2++) // loop over columns (local-orbital block) 
                    {
                        int lm2 = type->indexb(lo_index_offset + j2).lm;
                        int idxrf2 = type->indexb(lo_index_offset + j2).idxrf;
                        
                        // apw-lo block
                        for (int j1 = 0; j1 < type->mt_aw_basis_size(); j1++) // loop over rows
                        {
                            int lm1 = type->indexb(j1).lm;
                            int idxrf1 = type->indexb(j1).idxrf;
                            
                            complex16 zsum(0, 0);
                            
                            if (sblock == nm)
                                global.sum_L3_complex_gaunt(lm1, lm2, atom->h_radial_integral(idxrf2, idxrf1), zsum);
        
                            /*if (sblock == uu)
                            {
                                for (int lm3 = 0; lm3 < lapw_global.lmmaxvr; lm3++) 
                                    v1[lm3] = lapw_runtime.hmltrad(lm3, idxrf2, idxrf1, ias) + lapw_runtime.beffrad(lm3, idxrf1, idxrf2, ias, 0);
                                L3_sum_gntyry(lm1, lm2, &v1[0], zsum);
                            }
                            
                            if (sblock == dd)
                            {
                                for (int lm3 = 0; lm3 < lapw_global.lmmaxvr; lm3++) 
                                    v1[lm3] = lapw_runtime.hmltrad(lm3, idxrf2, idxrf1, ias) - lapw_runtime.beffrad(lm3, idxrf1, idxrf2, ias, 0);
                                L3_sum_gntyry(lm1, lm2, &v1[0], zsum);
                            }
                            
                            if (sblock == ud)
                            {
                                for (int lm3 = 0; lm3 < lapw_global.lmmaxvr; lm3++) 
                                    v2[lm3] = complex16(lapw_runtime.beffrad(lm3, idxrf1, idxrf2, ias, 1), -lapw_runtime.beffrad(lm3, idxrf1, idxrf2, ias, 2));
                                L3_sum_gntyry(lm1, lm2, &v2[0], zsum);
                            }*/
                
                            if (abs(zsum) > 1e-14)
                                for (int ig = 0; ig < kp.num_gkvec(); ig++)
                                    h(ig, kp.num_gkvec() + atom->offset_lo() + j2) += zsum * kp.matching_coefficient(ig, atom->offset_aw() + j1);
                        }
                        
                        int j1_last = j2;
                        if (sblock == ud) j1_last = type->mt_lo_basis_size() - 1;
                        
                        // lo-lo block 
                        for (int j1 = 0; j1 <= j1_last; j1++)
                        {
                            int lm1 = type->indexb(lo_index_offset + j1).lm;
                            int idxrf1 = type->indexb(lo_index_offset + j1).idxrf;
                            
                            complex16 zsum(0.0, 0.0);
        
                            if (sblock == nm)
                                global.sum_L3_complex_gaunt(lm1, lm2, atom->h_radial_integral(idxrf1, idxrf2), zsum);
        
                            /*if (sblock == uu)
                            {
                                for (int lm3 = 0; lm3 < lapw_global.lmmaxvr; lm3++) 
                                    v1[lm3] = lapw_runtime.hmltrad(lm3, idxrf1, idxrf2, ias) + lapw_runtime.beffrad(lm3, idxrf1, idxrf2, ias, 0);
                                L3_sum_gntyry(lm1, lm2, &v1[0], zsum);
                            }
                            
                            if (sblock == dd)
                            {
                                for (int lm3 = 0; lm3 < lapw_global.lmmaxvr; lm3++) 
                                    v1[lm3] = lapw_runtime.hmltrad(lm3, idxrf1, idxrf2, ias) - lapw_runtime.beffrad(lm3, idxrf1, idxrf2, ias, 0);
                                L3_sum_gntyry(lm1, lm2, &v1[0], zsum);
                            }
                            
                            if (sblock == ud)
                            {
                                for (int lm3 = 0; lm3 < lapw_global.lmmaxvr; lm3++) 
                                    v2[lm3] = complex16(lapw_runtime.beffrad(lm3, idxrf1, idxrf2, ias, 1), -lapw_runtime.beffrad(lm3, idxrf1, idxrf2, ias, 2));
                                L3_sum_gntyry(lm1, lm2, &v2[0], zsum);
                            }*/
        
                            h(kp.num_gkvec() + atom->offset_lo() + j1, kp.num_gkvec() + atom->offset_lo() + j2) += zsum;
                        }
                    }
        
                    /*if (sblock == ud)
                    {
                        for (int j2 = 0; j2 < species->index.apw_size(); j2++)
                        {
                            int lm2 = species->index[j2].lm;
                            int idxrf2 = species->index[j2].idxrf;
                            
                            for (int j1 = 0; j1 < species->index.lo_size(); j1++)
                            {
                                int lm1 = species->index[lo_index_offset + j1].lm;
                                int idxrf1 = species->index[lo_index_offset + j1].idxrf;
                                
                                complex16 zsum(0, 0);
                                
                                for (int lm3 = 0; lm3 < lapw_global.lmmaxvr; lm3++) 
                                    v2[lm3] = complex16(lapw_runtime.beffrad(lm3, idxrf1, idxrf2, ias, 1), -lapw_runtime.beffrad(lm3, idxrf1, idxrf2, ias, 2));
                                L3_sum_gntyry(lm1, lm2, &v2[0], zsum);
                                
                                if (abs(zsum) > 1e-14)
                                    for (int ig = 0; ig < ks->ngk; ig++)
                                        h(ks->ngk + atom->offset_lo + j1, ig) += zsum * conj(ks->apwalm(ig, atom->offset_apw + j2));
                            }
                        }
                    }*/
                } 
            }




            
            Timer *t1 = new Timer("sirius::Band::set_h::it");
            for (int ig2 = 0; ig2 < kp.num_gkvec(); ig2++) // loop over columns
            {
                int g1_last = ig2;
                if (sblock == ud) g1_last = kp.num_gkvec() - 1;
                
                // TODO: optimize scalar product if it takes time
                double v2[3];
                double v2c[3];
                for (int x = 0; x < 3; x++) v2[x] = kp.gkvec(ig2)[x];
                global.get_coordinates<cartesian,reciprocal>(v2, v2c);
                
                for (int ig1 = 0; ig1 <= g1_last; ig1++) // for each column loop over rows
                {
                    int ig = global.index_g12(kp.gvec_index(ig1), kp.gvec_index(ig2));
                    double v1[3];
                    double v1c[3];
                    for (int x = 0; x < 3; x++) v1[x] = kp.gkvec(ig1)[x];
                    global.get_coordinates<cartesian,reciprocal>(v1, v1c);
                    
                    double t1 = 0.5 * scalar_product(v1c, v2c);
                                       
                    if (sblock == nm)
                        h(ig1, ig2) += (global.effective_potential().f_pw(ig) + t1 * global.step_function_pw(ig));
                    
                    /*if (sblock == uu)
                        h(j1, j2) += (lapw_runtime.veffig(ig) + t1 * lapw_global.cfunig[ig] + lapw_runtime.beffig(ig, 0));
                    
                    if (sblock == dd)
                        h(j1, j2) += (lapw_runtime.veffig(ig) + t1 * lapw_global.cfunig[ig] - lapw_runtime.beffig(ig, 0));
                    
                    if (sblock == ud)
                        h(j1, j2) += (lapw_runtime.beffig(ig, 1) - zi * lapw_runtime.beffig(ig, 2));*/
                }
            }
            delete t1;
        }
        
        void set_o(kpoint_data_set& kp, mdarray<complex16,2>& o)
        {
            Timer t("sirius::Band::set_o");

            gemm<cpu>(0, 2, kp.num_gkvec(), kp.num_gkvec(), global.mt_aw_basis_size(), complex16(1.0, 0.0), 
                &kp.matching_coefficient(0, 0), kp.num_gkvec(), &kp.matching_coefficient(0, 0), kp.num_gkvec(), 
                complex16(0.0, 0.0), &o(0, 0), o.size(0)); 

            for (int ia = 0; ia < global.num_atoms(); ia++)
            {
                Atom* atom = global.atom(ia);
                AtomType* type = atom->type();

                int lo_index_offset = type->mt_aw_basis_size();

                for (int j2 = 0; j2 < type->mt_lo_basis_size(); j2++) // loop over columns (local-orbital block) 
                {
                    int l2 = type->indexb(lo_index_offset + j2).l;
                    int lm2 = type->indexb(lo_index_offset + j2).lm;
                    int order2 = type->indexb(lo_index_offset + j2).order;

                    // apw-lo block 
                    for (int order1 = 0; order1 < (int)type->aw_descriptor(l2).size(); order1++)
                        for (int ig = 0; ig < kp.num_gkvec(); ig++)
                            o(ig, kp.num_gkvec() + atom->offset_lo() + j2) += atom->symmetry_class()->o_radial_integral(l2, order1, order2) * 
                                kp.matching_coefficient(ig, atom->offset_aw() + type->indexb_by_lm_order(lm2, order1));

                    // lo-lo block
                    for (int j1 = 0; j1 <= j2; j1++)
                    {
                        int lm1 = type->indexb(lo_index_offset + j1).lm;
                        int order1 = type->indexb(lo_index_offset + j1).order;
                        if (lm1 == lm2) 
                            o(kp.num_gkvec() + atom->offset_lo() + j1, kp.num_gkvec() + atom->offset_lo() + j2) += 
                                atom->symmetry_class()->o_radial_integral(l2, order1, order2);
                    }
                }
            }
            
            for (int ig2 = 0; ig2 < kp.num_gkvec(); ig2++) // loop over columns
                for (int ig1 = 0; ig1 <= ig2; ig1++) // for each column loop over rows
                    o(ig1, ig2) += global.step_function_pw(global.index_g12(kp.gvec_index(ig1), kp.gvec_index(ig2)));
        }

        void find_eigen_states(kpoint_data_set& kp)
        {
            Timer t("sirius::Band::find_eigen_states");
            
            kp.generate_matching_coefficients();

            mdarray<complex16,2> h(kp.fv_basis_size(), kp.fv_basis_size());
            mdarray<complex16,2> o(kp.fv_basis_size(), kp.fv_basis_size());

            o.zero();
            set_o(kp, o);

            h.zero();
            set_h<nm>(kp, h);

            //write_h_o(h, o);
            assert(kp.fv_basis_size() > global.num_fv_states());

            kp.allocate();
            Timer *t1 = new Timer("sirius::band::find_eigen_states::hegv<impl>");
            int info = hegvx<cpu>(kp.fv_basis_size(), global.num_fv_states(), -1.0, &h(0, 0), &o(0, 0), kp.evalfv(), 
                                  kp.evecfv(), kp.fv_basis_size());
            delete t1;

            if (info)
            {
                std::stringstream s;
                s << "hegvx returned " << info;
                error(__FILE__, __LINE__, s);
            }

            //for (int i = 0; i < global.num_fv_states(); i++)
            //    std::cout << "i = " << i << "  e = " << kp.evalfv(i) << std::endl;
            
            kp.generate_scalar_wave_functions();

            //for (int i = 0; i < 3; i++)
            //    kp.test_scalar_wave_functions(i);

            //if (global.num_spins() == 1)
            
            
            kp.generate_spinor_wave_functions(-1);

        }

};

Band band;

};
