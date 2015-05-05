#include "band.h"

namespace sirius {

void Band::apply_magnetic_field(mdarray<double_complex, 2>& fv_states, int num_gkvec, int const* fft_index, 
                                Periodic_function<double>* effective_magnetic_field[3], mdarray<double_complex, 3>& hpsi)
{
    assert(hpsi.size(2) >= 2);
    assert(fv_states.size(0) == hpsi.size(0));
    assert(fv_states.size(1) == hpsi.size(1));

    int nfv = (int)fv_states.size(1);

    Timer t("sirius::Band::apply_magnetic_field");

    mdarray<double_complex, 3> zm(parameters_.unit_cell()->max_mt_basis_size(), parameters_.unit_cell()->max_mt_basis_size(), 
                                  parameters_.num_mag_dims());

    for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
    {
        int offset = parameters_.unit_cell()->atom(ia)->offset_wf();
        int mt_basis_size = parameters_.unit_cell()->atom(ia)->type()->mt_basis_size();
        Atom* atom = parameters_.unit_cell()->atom(ia);
        
        zm.zero();
        
        // only upper triangular part of zm is computed because it's a hermitian matrix
        #pragma omp parallel for default(shared)
        for (int j2 = 0; j2 < mt_basis_size; j2++)
        {
            int lm2 = atom->type()->indexb(j2).lm;
            int idxrf2 = atom->type()->indexb(j2).idxrf;
            
            for (int i = 0; i < parameters_.num_mag_dims(); i++)
            {
                for (int j1 = 0; j1 <= j2; j1++)
                {
                    int lm1 = atom->type()->indexb(j1).lm;
                    int idxrf1 = atom->type()->indexb(j1).idxrf;

                    zm(j1, j2, i) = gaunt_coefs_->sum_L3_gaunt(lm1, lm2, atom->b_radial_integrals(idxrf1, idxrf2, i)); 
                }
            }
        }
        /* compute bwf = B_z*|wf_j> */
        linalg<CPU>::hemm(0, 0, mt_basis_size, nfv, complex_one, &zm(0, 0, 0), zm.ld(), 
                          &fv_states(offset, 0), fv_states.ld(), complex_zero, &hpsi(offset, 0, 0), hpsi.ld());
        
        // compute bwf = (B_x - iB_y)|wf_j>
        if (hpsi.size(2) >= 3)
        {
            // reuse first (z) component of zm matrix to store (Bx - iBy)
            for (int j2 = 0; j2 < mt_basis_size; j2++)
            {
                for (int j1 = 0; j1 <= j2; j1++) zm(j1, j2, 0) = zm(j1, j2, 1) - complex_i * zm(j1, j2, 2);
                
                // remember: zm is hermitian and we computed only the upper triangular part
                for (int j1 = j2 + 1; j1 < mt_basis_size; j1++) zm(j1, j2, 0) = conj(zm(j2, j1, 1)) - complex_i * conj(zm(j2, j1, 2));
            }
              
            linalg<CPU>::gemm(0, 0, mt_basis_size, nfv, mt_basis_size, &zm(0, 0, 0), zm.ld(), 
                              &fv_states(offset, 0), fv_states.ld(), &hpsi(offset, 0, 2), hpsi.ld());
        }
        
        // compute bwf = (B_x + iB_y)|wf_j>
        if (hpsi.size(2) == 4 && std_evp_solver()->parallel())
        {
            // reuse first (z) component of zm matrix to store (Bx + iBy)
            for (int j2 = 0; j2 < mt_basis_size; j2++)
            {
                for (int j1 = 0; j1 <= j2; j1++) zm(j1, j2, 0) = zm(j1, j2, 1) + complex_i * zm(j1, j2, 2);
                
                for (int j1 = j2 + 1; j1 < mt_basis_size; j1++) zm(j1, j2, 0) = conj(zm(j2, j1, 1)) + complex_i * conj(zm(j2, j1, 2));
            }
              
            linalg<CPU>::gemm(0, 0, mt_basis_size, nfv, mt_basis_size, &zm(0, 0, 0), zm.ld(), 
                              &fv_states(offset, 0), fv_states.ld(), &hpsi(offset, 0, 3), hpsi.ld());
        }
    }
    
    Timer t1("sirius::Band::apply_magnetic_field|it");

    int offset = parameters_.unit_cell()->mt_basis_size();

    #pragma omp parallel default(shared) num_threads(fft_->num_fft_threads())
    {        
        int thread_id = omp_get_thread_num();
        
        std::vector<double_complex> psi_it(fft_->size());
        std::vector<double_complex> hpsi_it(fft_->size());
        
        #pragma omp for
        for (int i = 0; i < nfv; i++)
        {
            fft_->input(num_gkvec, fft_index, &fv_states(offset, i), thread_id);
            fft_->transform(1, thread_id);
                                        
            for (int ir = 0; ir < fft_->size(); ir++)
            {
                /* hpsi(r) = psi(r) * Bz(r) * Theta(r) */
                fft_->buffer(ir, thread_id) *= (effective_magnetic_field[0]->f_it<global>(ir) * parameters_.step_function(ir));
            }
            
            fft_->transform(-1, thread_id);
            fft_->output(num_gkvec, fft_index, &hpsi(offset, i, 0), thread_id); 

            if (hpsi.size(2) >= 3)
            {
                for (int ir = 0; ir < fft_->size(); ir++)
                {
                    /* hpsi(r) = psi(r) * (Bx(r) - iBy(r)) * Theta(r) */
                    hpsi_it[ir] = psi_it[ir] * parameters_.step_function(ir) * 
                                  (effective_magnetic_field[1]->f_it<global>(ir) - 
                                   complex_i * effective_magnetic_field[2]->f_it<global>(ir));
                }
                
                fft_->input(&hpsi_it[0], thread_id);
                fft_->transform(-1, thread_id);
                fft_->output(num_gkvec, fft_index, &hpsi(offset, i, 2), thread_id); 
            }
            
            if (hpsi.size(2) == 4 && std_evp_solver()->parallel())
            {
                for (int ir = 0; ir < fft_->size(); ir++)
                {
                    /* hpsi(r) = psi(r) * (Bx(r) + iBy(r)) * Theta(r) */
                    hpsi_it[ir] = psi_it[ir] * parameters_.step_function(ir) *
                                  (effective_magnetic_field[1]->f_it<global>(ir) + 
                                   complex_i * effective_magnetic_field[2]->f_it<global>(ir));
                }
                
                fft_->input(&hpsi_it[0], thread_id);
                fft_->transform(-1, thread_id);
                fft_->output(num_gkvec, fft_index, &hpsi(offset, i, 3), thread_id); 
            }
        }
    }

    /* copy Bz|\psi> to -Bz|\psi> */
    for (int i = 0; i < nfv; i++)
    {
        for (int j = 0; j < (int)fv_states.size(0); j++) hpsi(j, i, 1) = -hpsi(j, i, 0);
    }
}

};
