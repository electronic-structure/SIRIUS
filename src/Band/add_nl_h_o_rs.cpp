#include "band.h"

namespace sirius {

void Band::add_nl_h_o_rs(K_point* kp__,
                         int n__,
                         matrix<double_complex>& phi__,
                         matrix<double_complex>& hphi__,
                         matrix<double_complex>& ophi__,
                         mdarray<int, 1>& packed_mtrx_offset__,
                         mdarray<double_complex, 1>& d_mtrx_packed__,
                         mdarray<double_complex, 1>& q_mtrx_packed__,
                         mdarray<double_complex, 1>& kappa__)
{
    PROFILE();

    STOP();

    //== auto rsp = ctx_.real_space_prj();
    //== auto fft = rsp->fft();

    //== if (kappa__.size() < size_t(2 * fft->size() + rsp->max_num_points_) * ctx_.num_fft_streams())
    //== {
    //==     TERMINATE("wrong size of work array");
    //== }

    //== std::vector<int> fft_index(kp__->num_gkvec());
    //== for (int igk = 0; igk < kp__->num_gkvec(); igk++)
    //== {
    //==     //vector3d<int> gvec = kp__->gvec(igk);
    //==     STOP();
    //==     /* linear index inside coarse FFT buffer */
    //==     //fft_index[igk] = fft->index(gvec[0], gvec[1], gvec[2]);
    //== }

    //== STOP();
    //== 
    //== //std::vector<double_complex> k_phase(fft->size());
    //== ///* loop over 3D array (real space) */
    //== //for (int j0 = 0; j0 < fft->fft_grid().size(0); j0++)
    //== //{
    //== //    for (int j1 = 0; j1 < fft->fft_grid().size(1); j1++)
    //== //    {
    //== //        for (int j2 = 0; j2 < fft->fft_grid().size(2); j2++)
    //== //        {
    //== //            /* get real space fractional coordinate */
    //== //            vector3d<double> v0(double(j0) / fft->size(0), double(j1) / fft->size(1), double(j2) / fft->size(2));
    //== //            int ir = static_cast<int>(j0 + j1 * fft->size(0) + j2 * fft->size(0) * fft->size(1));
    //== //            k_phase[ir] = std::exp(double_complex(0.0, twopi * (kp__->vk() * v0)));
    //== //        }
    //== //    }
    //== //}

    //== mdarray<double_complex, 3> T_phase(mdarray_index_descriptor(-1, 1),
    //==                                    mdarray_index_descriptor(-1, 1),
    //==                                    mdarray_index_descriptor(-1, 1));
    //== for (int t0 = -1; t0 <= 1; t0++)
    //== {
    //==     for (int t1 = -1; t1 <= 1; t1++)
    //==     {
    //==         for (int t2 = -1; t2 <= 1; t2++)
    //==         {
    //==             vector3d<int> T(t0, t1, t2);
    //==             T_phase(t0, t1, t2) = std::exp(double_complex(0.0, twopi * (kp__->vk() * T)));
    //==         }
    //==     }
    //== }

    //== mdarray<double_complex, 2> hphi_rs(kappa__.at<CPU>(),               fft->size(), ctx_.num_fft_streams());
    //== mdarray<double_complex, 2> ophi_rs(kappa__.at<CPU>(hphi_rs.size()), fft->size(), ctx_.num_fft_streams());
    //== 
    //== mdarray<double, 2> timers(4, omp_get_max_threads());
    //== timers.zero();

    //== /* <\beta_{\xi}^{\alpha}|\phi_j> */
    //== mdarray<double, 2> beta_phi_re(unit_cell_.max_mt_basis_size(), ctx_.num_fft_streams());
    //== mdarray<double, 2> beta_phi_im(unit_cell_.max_mt_basis_size(), ctx_.num_fft_streams());

    //== /* Q or D multiplied by <\beta_{\xi}^{\alpha}|\phi_j> */
    //== mdarray<double, 2> d_beta_phi_re(unit_cell_.max_mt_basis_size(), ctx_.num_fft_streams());
    //== mdarray<double, 2> d_beta_phi_im(unit_cell_.max_mt_basis_size(), ctx_.num_fft_streams());
    //== mdarray<double, 2> q_beta_phi_re(unit_cell_.max_mt_basis_size(), ctx_.num_fft_streams());
    //== mdarray<double, 2> q_beta_phi_im(unit_cell_.max_mt_basis_size(), ctx_.num_fft_streams());
    //== 
    //== double* ptr = (double*)kappa__.at<CPU>(2 * hphi_rs.size());
    //== mdarray<double, 2> phi_tmp_re(ptr,                     rsp->max_num_points_, ctx_.num_fft_streams());
    //== mdarray<double, 2> phi_tmp_im(ptr + phi_tmp_re.size(), rsp->max_num_points_, ctx_.num_fft_streams());

    //== mdarray<double_complex, 2> phase(rsp->max_num_points_, ctx_.num_fft_streams());
    //== 
    //== //double w1 = std::sqrt(unit_cell_.omega()) / fft->size();
    //== //double w2 = std::sqrt(unit_cell_.omega());

    //== STOP();

    //== //Timer t5("sirius::Band::apply_h_o_serial|real_space_kernel");
    //== //#pragma omp parallel num_threads(ctx_.num_fft_threads())
    //== //{
    //== //    int thread_id = Platform::thread_id();

    //== //    #pragma omp for
    //== //    for (int ib = 0; ib < n__; ib++)
    //== //    {
    //== //        memset(&hphi_rs(0, thread_id), 0, fft->size() * sizeof(double_complex));
    //== //        memset(&ophi_rs(0, thread_id), 0, fft->size() * sizeof(double_complex));

    //== //        double t0 = omp_get_wtime();
    //== //        fft->input(kp__->num_gkvec(), &fft_index[0], &phi__(0, ib), thread_id);
    //== //        /* phi(G) -> phi(r) */
    //== //        STOP();
    //== //        //fft->transform(1, kp__->gkvec().z_sticks_coord(), thread_id);
    //== //        timers(0, thread_id) += omp_get_wtime() - t0;

    //== //        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
    //== //        {
    //== //            auto type = unit_cell_.atom(ia)->type();
    //== //            auto& beta_prj = rsp->beta_projectors_[ia];
    //== //            int nbf = type->mt_basis_size();
    //== //            int npt = beta_prj.num_points_;

    //== //            double t0 = omp_get_wtime();

    //== //            for (int j = 0; j < npt; j++)
    //== //            {
    //== //                int ir = beta_prj.ir_[j];
    //== //                auto T = beta_prj.T_[j];
    //== //                phase(j, thread_id) = conj(T_phase(T[0], T[1], T[2])) * k_phase[ir];
    //== //                double_complex z = fft->buffer(ir, thread_id) * w1 * phase(j, thread_id);
    //== //                phi_tmp_re(j, thread_id) = real(z);
    //== //                phi_tmp_im(j, thread_id) = imag(z);
    //== //            }

    //== //            linalg<CPU>::gemv(1, npt, nbf, 1.0, &beta_prj.beta_(0, 0), beta_prj.beta_.ld(), 
    //== //                              &phi_tmp_re(0, thread_id), 1, 0.0, &beta_phi_re(0, thread_id), 1);

    //== //            linalg<CPU>::gemv(1, npt, nbf, 1.0, &beta_prj.beta_(0, 0), beta_prj.beta_.ld(), 
    //== //                              &phi_tmp_im(0, thread_id), 1, 0.0, &beta_phi_im(0, thread_id), 1);

    //== //            timers(1, thread_id) += omp_get_wtime() - t0;

    //== //            t0 = omp_get_wtime();
    //== //            for (int xi1 = 0; xi1 < nbf; xi1++)
    //== //            {
    //== //                double_complex z1(0, 0);
    //== //                double_complex z2(0, 0);
    //== //                for (int xi2 = 0; xi2 < nbf; xi2++)
    //== //                {
    //== //                    z1 += d_mtrx_packed__(packed_mtrx_offset__(ia) + xi2 * nbf + xi1) *
    //== //                          double_complex(beta_phi_re(xi2, thread_id), beta_phi_im(xi2, thread_id));
    //== //                    z2 += q_mtrx_packed__(packed_mtrx_offset__(ia) + xi2 * nbf + xi1) *
    //== //                          double_complex(beta_phi_re(xi2, thread_id), beta_phi_im(xi2, thread_id));
    //== //                }
    //== //                d_beta_phi_re(xi1, thread_id) = real(z1);
    //== //                d_beta_phi_im(xi1, thread_id) = imag(z1);
    //== //                q_beta_phi_re(xi1, thread_id) = real(z2);
    //== //                q_beta_phi_im(xi1, thread_id) = imag(z2);
    //== //            }
    //== //            timers(2, thread_id) += omp_get_wtime() - t0;

    //== //            t0 = omp_get_wtime();
    //== //            
    //== //            linalg<CPU>::gemv(0, npt, nbf, 1.0, &beta_prj.beta_(0, 0), beta_prj.beta_.ld(), 
    //== //                              &d_beta_phi_re(0, thread_id), 1, 0.0, &phi_tmp_re(0, thread_id), 1);
    //== //            linalg<CPU>::gemv(0, npt, nbf, 1.0, &beta_prj.beta_(0, 0), beta_prj.beta_.ld(), 
    //== //                              &d_beta_phi_im(0, thread_id), 1, 0.0, &phi_tmp_im(0, thread_id), 1);

    //== //            for (int j = 0; j < npt; j++)
    //== //            {
    //== //                int ir = beta_prj.ir_[j];
    //== //                hphi_rs(ir, thread_id) += conj(phase(j, thread_id)) * double_complex(phi_tmp_re(j, thread_id), phi_tmp_im(j, thread_id)) * w2;
    //== //            }

    //== //            linalg<CPU>::gemv(0, npt, nbf, 1.0, &beta_prj.beta_(0, 0), beta_prj.beta_.ld(), 
    //== //                              &q_beta_phi_re(0, thread_id), 1, 0.0, &phi_tmp_re(0, thread_id), 1);
    //== //            linalg<CPU>::gemv(0, npt, nbf, 1.0, &beta_prj.beta_(0, 0), beta_prj.beta_.ld(), 
    //== //                              &q_beta_phi_im(0, thread_id), 1, 0.0, &phi_tmp_im(0, thread_id), 1);

    //== //            for (int j = 0; j < npt; j++)
    //== //            {
    //== //                int ir = beta_prj.ir_[j];
    //== //                ophi_rs(ir, thread_id) += conj(phase(j, thread_id)) * double_complex(phi_tmp_re(j, thread_id), phi_tmp_im(j, thread_id)) * w2;
    //== //            }
    //== //            
    //== //            timers(3, thread_id) += omp_get_wtime() - t0;
    //== //        }

    //== //        t0 = omp_get_wtime();
    //== //        fft->input(&hphi_rs(0, thread_id), thread_id);
    //== //        STOP();
    //== //        //fft->transform(-1, thread_id);
    //== //        fft->output(kp__->num_gkvec(), &fft_index[0], &hphi__(0, ib), thread_id, 1.0);

    //== //        fft->input(&ophi_rs(0, thread_id), thread_id);
    //== //        STOP();
    //== //        //fft->transform(-1, thread_id);
    //== //        fft->output(kp__->num_gkvec(), &fft_index[0], &ophi__(0, ib), thread_id, 1.0);

    //== //        timers(0, thread_id) += omp_get_wtime() - t0;
    //== //    }
    //== //}
    //== //t5.stop();
    //== 
    //== //== if (kp__->comm().rank() == 0)
    //== //== {
    //== //==     std::cout << "------------------------------------------------------------" << std::endl;
    //== //==     //std::cout << "thread_id  |    fft    |  beta_phi  | apply_d_q |   add_nl  " << std::endl;
    //== //==     //std::cout << "------------------------------------------------------------" << std::endl;
    //== //==     //for (int i = 0; i < Platform::max_num_threads(); i++)
    //== //==     //{
    //== //==     //    printf("   %2i      | %8.4f  | %8.4f   | %8.4f  | %8.4f\n", i, timers(0, i), timers(1, i), timers(2, i), timers(3, i));
    //== //==     //}
    //== //==     double tot[] = {0, 0, 0, 0};
    //== //==     for (int i = 0; i < Platform::max_num_threads(); i++)
    //== //==     {
    //== //==         tot[0] += timers(0, i);
    //== //==         tot[1] += timers(1, i);
    //== //==         tot[2] += timers(2, i);
    //== //==         tot[3] += timers(3, i);
    //== //==     }
    //== //==     printf("fft       : %8.4f\n", tot[0]);
    //== //==     printf("beta_phi  : %8.4f\n", tot[1]);
    //== //==     printf("apply_d_q : %8.4f\n", tot[2]);
    //== //==     printf("add_nl    : %8.4f\n", tot[3]);
    //== //==     printf("total     : %8.4f\n", tval);

    //== //==     std::cout << "------------------------------------------------------------" << std::endl;
    //== //== }
}

};
