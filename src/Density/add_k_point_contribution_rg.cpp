#include <thread>
#include <mutex>
#include "density.h"

namespace sirius {

#ifdef __GPU
extern "C" void update_density_rg_1_gpu(int size__, 
                                        cuDoubleComplex const* psi_rg__, 
                                        double wt__, 
                                        double* density_rg__);
#endif

template <bool mt_spheres>
void Density::add_k_point_contribution_rg(K_point* kp__)
{
    PROFILE_WITH_TIMER("sirius::Density::add_k_point_contribution_rg");

    int nfv = ctx_.num_fv_states();
    double omega = unit_cell_.omega();

    mdarray<double, 2> density_rg(ctx_.fft().local_size(), ctx_.num_mag_dims() + 1);
    density_rg.zero();

    #ifdef __GPU
    if (ctx_.fft().hybrid()) {
        density_rg.allocate_on_device();
        density_rg.zero_on_device();
    }
    #endif

    ctx_.fft().prepare(kp__->gkvec().partition());

    int wf_pw_offset = kp__->wf_pw_offset();
        
    /* non-magnetic or collinear case */
    if (ctx_.num_mag_dims() != 3) {
        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            if (!kp__->spinor_wave_functions<mt_spheres>(ispn).spl_num_swapped().global_index_size()) {
                continue;
            }

            #pragma omp for schedule(dynamic, 1)
            for (int i = 0; i < kp__->spinor_wave_functions<mt_spheres>(ispn).spl_num_swapped().local_size(); i++) {
                int j = kp__->spinor_wave_functions<mt_spheres>(ispn).spl_num_swapped()[i];
                double w = kp__->band_occupancy(j + ispn * nfv) * kp__->weight() / omega;

                    /* transform to real space; in case of GPU wave-function stays in GPU memory */
                if (ctx_.fft().gpu_only()) {
                    ctx_.fft().transform<1>(kp__->gkvec().partition(),
                                            kp__->spinor_wave_functions<mt_spheres>(ispn).coeffs_swapped().template at<GPU>(wf_pw_offset, i));
                } else {
                    ctx_.fft().transform<1>(kp__->gkvec().partition(),
                                            kp__->spinor_wave_functions<mt_spheres>(ispn)[i] + wf_pw_offset);
                }

                if (ctx_.fft().hybrid()) {
                    #ifdef __GPU
                    update_density_rg_1_gpu(ctx_.fft().local_size(), ctx_.fft().buffer<GPU>(), w, density_rg.at<GPU>(0, ispn));
                    #else
                    TERMINATE_NO_GPU
                    #endif
                } else {
                    #pragma omp parallel for
                    for (int ir = 0; ir < ctx_.fft().local_size(); ir++) {
                        auto z = ctx_.fft().buffer(ir);
                        density_rg(ir, ispn) += w * (std::pow(z.real(), 2) + std::pow(z.imag(), 2));
                    }
                }
            }
        }
    }
    else
    {
        assert(kp__->spinor_wave_functions<mt_spheres>(0).spl_num_swapped().local_size() ==
               kp__->spinor_wave_functions<mt_spheres>(1).spl_num_swapped().local_size());

        std::vector<double_complex> psi_r(ctx_.fft().local_size());

        #pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < kp__->spinor_wave_functions<mt_spheres>(0).spl_num_swapped().local_size(); i++)
        {
            int j = kp__->spinor_wave_functions<mt_spheres>(0).spl_num_swapped()[i];
            double w = kp__->band_occupancy(j) * kp__->weight() / omega;

            /* transform up- component of spinor function to real space; in case of GPU wave-function stays in GPU memory */
            ctx_.fft().transform<1>(kp__->gkvec().partition(), kp__->spinor_wave_functions<mt_spheres>(0)[i] + wf_pw_offset);
            /* save in auxiliary buffer */
            ctx_.fft().output(&psi_r[0]);
            /* transform dn- component of spinor wave function */
            ctx_.fft().transform<1>(kp__->gkvec().partition(), kp__->spinor_wave_functions<mt_spheres>(1)[i] + wf_pw_offset);

            if (ctx_.fft().hybrid())
            {
                STOP();
                //#ifdef __GPU
                //update_it_density_matrix_1_gpu(ctx_.fft(thread_id)->local_size(), ispn, ctx_.fft(thread_id)->buffer<GPU>(), w,
                //                               it_density_matrix_gpu.at<GPU>());
                //#else
                //TERMINATE_NO_GPU
                //#endif
            }
            else
            {
                #pragma omp parallel for
                for (int ir = 0; ir < ctx_.fft().local_size(); ir++)
                {
                    auto r0 = (std::pow(psi_r[ir].real(), 2) + std::pow(psi_r[ir].imag(), 2)) * w;
                    auto r1 = (std::pow(ctx_.fft().buffer(ir).real(), 2) +
                               std::pow(ctx_.fft().buffer(ir).imag(), 2)) * w;

                    auto z2 = psi_r[ir] * std::conj(ctx_.fft().buffer(ir)) * w;

                    density_rg(ir, 0) += r0;
                    density_rg(ir, 1) += r1;
                    density_rg(ir, 2) += 2.0 * std::real(z2);
                    density_rg(ir, 3) -= 2.0 * std::imag(z2);
                }
            }
        }
    }

    #ifdef __GPU
    if (ctx_.fft().hybrid()) density_rg.copy_to_host();
    #endif
    
    /* switch from real density matrix to density and magnetization */
    switch (ctx_.num_mag_dims())
    {
        case 3:
        {
            #pragma omp parallel for
            for (int ir = 0; ir < ctx_.fft().local_size(); ir++)
            {
                magnetization_[1]->f_rg(ir) += density_rg(ir, 2);
                magnetization_[2]->f_rg(ir) += density_rg(ir, 3);
            }
        }
        case 1:
        {
            #pragma omp parallel for
            for (int ir = 0; ir < ctx_.fft().local_size(); ir++)
            {
                rho_->f_rg(ir) += (density_rg(ir, 0) + density_rg(ir, 1));
                magnetization_[0]->f_rg(ir) += (density_rg(ir, 0) - density_rg(ir, 1));
            }
            break;
        }
        case 0:
        {
            #pragma omp parallel for
            for (int ir = 0; ir < ctx_.fft().local_size(); ir++) rho_->f_rg(ir) += density_rg(ir, 0);
        }
    }

    ctx_.fft().dismiss();
}

template void Density::add_k_point_contribution_rg<true>(K_point* kp__);
template void Density::add_k_point_contribution_rg<false>(K_point* kp__);

};
