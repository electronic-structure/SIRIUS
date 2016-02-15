#include "atom.h"

namespace sirius {

extern "C" void spline_inner_product_gpu_v3(int const* idx_ri__,
                                            int num_ri__,
                                            int num_points__,
                                            double const* x__,
                                            double const* dx__,
                                            double const* f__, 
                                            double const* g__,
                                            double* result__);

void Atom::generate_radial_integrals(processing_unit_t pu__, Communicator const& comm__)
{
    PROFILE_WITH_TIMER("sirius::Atom::generate_radial_integrals");
    
    int lmmax = Utils::lmmax(lmax_pot_);
    int nmtp = type().num_mt_points();
    int nrf = type().indexr().size();

    if (comm__.size() != 1) TERMINATE("not yet mpi parallel");

    splindex<block> spl_lm(lmmax, comm__.size(), comm__.rank());

    std::vector<int> l_by_lm = Utils::l_by_lm(lmax_pot_);

    h_radial_integrals_.zero();
    if (num_mag_dims_) b_radial_integrals_.zero();
    
    /* copy radial functions to spline objects */
    std::vector< Spline<double> > rf_spline(nrf);
    #pragma omp parallel for
    for (int i = 0; i < nrf; i++)
    {
        rf_spline[i] = Spline<double>(type().radial_grid());
        for (int ir = 0; ir < nmtp; ir++) rf_spline[i][ir] = symmetry_class().radial_function(ir, i);
    }

    /* copy effective potential components to spline objects */
    std::vector< Spline<double> > v_spline(lmmax * (1 + num_mag_dims_));
    #pragma omp parallel for
    for (int lm = 0; lm < lmmax; lm++)
    {
        v_spline[lm] = Spline<double>(type().radial_grid());
        for (int ir = 0; ir < nmtp; ir++) v_spline[lm][ir] = veff_(lm, ir);

        for (int j = 0; j < num_mag_dims_; j++)
        {
            v_spline[lm + (j + 1) * lmmax] = Spline<double>(type().radial_grid());
            for (int ir = 0; ir < nmtp; ir++) v_spline[lm + (j + 1) * lmmax][ir] = beff_[j](lm, ir);
        }
    }

    /* interpolate potential multiplied by a radial function */
    std::vector< Spline<double> > vrf_spline(lmmax * nrf * (1 + num_mag_dims_));

    auto& idx_ri = type().idx_radial_integrals();

    mdarray<double, 1> result(idx_ri.size(1));

    if (pu__ == GPU)
    {
        #ifdef __GPU
        auto& rgrid = type().radial_grid();
        auto& rf_coef = type().rf_coef();
        auto& vrf_coef = type().vrf_coef();

        runtime::Timer t1("sirius::Atom::generate_radial_integrals|interp");
        #pragma omp parallel
        {
            //int tid = Platform::thread_id();
            #pragma omp for
            for (int i = 0; i < nrf; i++)
            {
                rf_spline[i].interpolate();
                std::memcpy(rf_coef.at<CPU>(0, 0, i), rf_spline[i].coeffs().at<CPU>(), nmtp * 4 * sizeof(double));
                //cuda_async_copy_to_device(rf_coef.at<GPU>(0, 0, i), rf_coef.at<CPU>(0, 0, i), nmtp * 4 * sizeof(double), tid);
            }
            #pragma omp for
            for (int i = 0; i < lmmax * (1 + num_mag_dims_); i++) v_spline[i].interpolate();
        }
        rf_coef.async_copy_to_device();

        #pragma omp parallel for
        for (int lm = 0; lm < lmmax; lm++)
        {
            for (int i = 0; i < nrf; i++)
            {
                for (int j = 0; j < num_mag_dims_ + 1; j++)
                {
                    int idx = lm + lmmax * i + lmmax * nrf * j;
                    vrf_spline[idx] = rf_spline[i] * v_spline[lm + j * lmmax];
                    std::memcpy(vrf_coef.at<CPU>(0, 0, idx), vrf_spline[idx].coeffs().at<CPU>(), nmtp * 4 * sizeof(double));
                    //cuda_async_copy_to_device(vrf_coef.at<GPU>(0, 0, idx), vrf_coef.at<CPU>(0, 0, idx), nmtp * 4 *sizeof(double), tid);
                }
            }
        }
        vrf_coef.copy_to_device();
        t1.stop();

        result.allocate_on_device();
        runtime::Timer t2("sirius::Atom::generate_radial_integrals|inner");
        spline_inner_product_gpu_v3(idx_ri.at<GPU>(), (int)idx_ri.size(1), nmtp, rgrid.x().at<GPU>(), rgrid.dx().at<GPU>(),
                                    rf_coef.at<GPU>(), vrf_coef.at<GPU>(), result.at<GPU>());
        cuda_device_synchronize();
        double tval = t2.stop();
        DUMP("spline GPU integration performance: %12.6f GFlops", 1e-9 * double(idx_ri.size(1)) * nmtp * 85 / tval);
        result.copy_to_host();
        result.deallocate_on_device();
        #else
        TERMINATE_NO_GPU
        #endif
    }
    if (pu__ == CPU)
    {
        runtime::Timer t1("sirius::Atom::generate_radial_integrals|interp");
        #pragma omp parallel
        {
            #pragma omp for
            for (int i = 0; i < nrf; i++) rf_spline[i].interpolate();
            #pragma omp for
            for (int i = 0; i < lmmax * (1 + num_mag_dims_); i++) v_spline[i].interpolate();
            
            #pragma omp for
            for (int lm = 0; lm < lmmax; lm++)
            {
                for (int i = 0; i < nrf; i++)
                {
                    for (int j = 0; j < num_mag_dims_ + 1; j++)
                    {
                        vrf_spline[lm + lmmax * i + lmmax * nrf * j] = rf_spline[i] * v_spline[lm + j * lmmax];
                    }
                }
            }
        }
        t1.stop();

        runtime::Timer t2("sirius::Atom::generate_radial_integrals|inner");
        #pragma omp parallel for
        for (int j = 0; j < (int)idx_ri.size(1); j++)
        {
            result(j) = inner(rf_spline[idx_ri(0, j)], vrf_spline[idx_ri(1, j)], 2);
        }
        double tval = t2.stop();
        DUMP("spline CPU integration performance: %12.6f GFlops", 1e-9 * double(idx_ri.size(1)) * nmtp * 85 / tval);
    }
    
    int n = 0;
    for (int lm = 0; lm < lmmax; lm++)
    {
        int l = l_by_lm[lm];

        for (int i2 = 0; i2 < type().indexr().size(); i2++)
        {
            int l2 = type().indexr(i2).l;
            
            for (int i1 = 0; i1 <= i2; i1++)
            {
                int l1 = type().indexr(i1).l;
                if ((l + l1 + l2) % 2 == 0)
                {
                    if (lm)
                    {
                        h_radial_integrals_(lm, i1, i2) = h_radial_integrals_(lm, i2, i1) = result(n++);
                    }
                    else
                    {
                        h_radial_integrals_(0, i1, i2) = symmetry_class().h_spherical_integral(i1, i2);
                        h_radial_integrals_(0, i2, i1) = symmetry_class().h_spherical_integral(i2, i1);
                    }
                    for (int j = 0; j < num_mag_dims_; j++)
                    {
                        b_radial_integrals_(lm, i1, i2, j) = b_radial_integrals_(lm, i2, i1, j) = result(n++);
                    }
                }
            }
        }
    }
    
    //== #pragma omp parallel default(shared)
    //== {
    //==     /* potential or magnetic field times a radial function */
    //==     std::vector< Spline<double> > vrf_spline(1 + num_mag_dims_);
    //==     for (int i = 0; i < 1 + num_mag_dims_; i++) vrf_spline[i] = Spline<double>(type().radial_grid());

    //==     for (int lm_loc = 0; lm_loc < (int)spl_lm.local_size(); lm_loc++)
    //==     {
    //==         int lm = (int)spl_lm[lm_loc];
    //==         int l = l_by_lm[lm];

    //==         #pragma omp for
    //==         for (int i2 = 0; i2 < type().indexr().size(); i2++)
    //==         {
    //==             int l2 = type().indexr(i2).l;
    //==             
    //==             /* multiply potential by a radial function */
    //==             for (int ir = 0; ir < nmtp; ir++) 
    //==                 vrf_spline[0][ir] = symmetry_class()->radial_function(ir, i2) * veff_(lm, ir);
    //==             vrf_spline[0].interpolate();
    //==             /* multiply magnetic field by a radial function */
    //==             for (int j = 0; j < num_mag_dims_; j++)
    //==             {
    //==                 for (int ir = 0; ir < nmtp; ir++) 
    //==                     vrf_spline[1 + j][ir] = symmetry_class()->radial_function(ir, i2) * beff_[j](lm, ir);
    //==                 vrf_spline[1 + j].interpolate();
    //==             }
    //==             
    //==             for (int i1 = 0; i1 <= i2; i1++)
    //==             {
    //==                 int l1 = type().indexr(i1).l;
    //==                 if ((l + l1 + l2) % 2 == 0)
    //==                 {
    //==                     if (lm)
    //==                     {
    //==                         double a = inner(rf_spline[i1], vrf_spline[0], 2);
    //==                         if (std::abs(h_radial_integrals_(lm, i1, i2)-a)>1e-10) 
    //==                         {
    //==                             std::cout << i1 << " " << i2 << " " << lm << std::endl;
    //==                             STOP();

    //==                         }
    //==                         h_radial_integrals_(lm, i1, i2) = h_radial_integrals_(lm, i2, i1) = 
    //==                             inner(rf_spline[i1], vrf_spline[0], 2);
    //==                     }
    //==                     else
    //==                     {
    //==                         h_radial_integrals_(0, i1, i2) = symmetry_class()->h_spherical_integral(i1, i2);
    //==                         h_radial_integrals_(0, i2, i1) = symmetry_class()->h_spherical_integral(i2, i1);
    //==                     }
    //==                     for (int j = 0; j < num_mag_dims_; j++)
    //==                     {
    //==                         double a = inner(rf_spline[i1], vrf_spline[1+j], 2);
    //==                         if (std::abs( b_radial_integrals_(lm, i1, i2, j) - a)>1e-10)
    //==                         {
    //==                             std::cout << i1 << " " << i2 << " " << lm << std::endl;
    //==                             std::cout << "a=" << a <<" b_radial_integrals_(lm, i1, i2, j)="<<  b_radial_integrals_(lm, i1, i2, j)<<std::endl;
    //==                             STOP();
    //==                         }
    //==                         b_radial_integrals_(lm, i1, i2, j) = b_radial_integrals_(lm, i2, i1, j) = 
    //==                             inner(rf_spline[i1], vrf_spline[1 + j], 2);
    //==                     }
    //==                 }
    //==             }
    //==         }
    //==     }
    //== }

    //== comm__.reduce(h_radial_integrals_.at<CPU>(), (int)h_radial_integrals_.size(), 0);
    //== if (num_mag_dims_) comm__.reduce(b_radial_integrals_.at<CPU>(), (int)b_radial_integrals_.size(), 0);

    #ifdef __PRINT_OBJECT_HASH
    DUMP("hash(veff): %16llX", veff_.hash());
    DUMP("hash(h_radial_integrals): %16llX", h_radial_integrals_.hash());
    #endif
    #ifdef __PRINT_OBJECT_CHECKSUM
    DUMP("checksum(h_radial_integrals): %18.10f", h_radial_integrals_.checksum());
    #endif
}

}
