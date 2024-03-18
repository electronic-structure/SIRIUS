// Copyright (c) 2013-2019 Anton Kozhevnikov, Thomas Schulthess
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that

// the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
//    following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
//    and the following disclaimer in the documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/** \file density.cpp
 *
 *  \brief Contains implementation of sirius::Density class.
 */

#include "core/profiler.hpp"
#include "core/serialize_mdarray.hpp"
#include "beta_projectors/beta_projectors_base.hpp"
#include "symmetry/symmetrize_field4d.hpp"
#include "symmetry/symmetrize_density_matrix.hpp"
#include "symmetry/symmetrize_occupation_matrix.hpp"
#include "mixer/mixer_functions.hpp"
#include "mixer/mixer_factory.hpp"
#include "lapw/generate_gvec_ylm.hpp"
#include "lapw/sum_fg_fl_yg.hpp"
#include "lapw/generate_sbessel_mt.hpp"
#include "density.hpp"

namespace sirius {

#if defined(SIRIUS_GPU)
void
update_density_rg_1_real_gpu(int size__, float const* psi_rg__, float wt__, float* density_rg__)
{
    update_density_rg_1_real_gpu_float(size__, psi_rg__, wt__, density_rg__);
}

void
update_density_rg_1_real_gpu(int size__, double const* psi_rg__, double wt__, double* density_rg__)
{
    update_density_rg_1_real_gpu_double(size__, psi_rg__, wt__, density_rg__);
}

void
update_density_rg_1_complex_gpu(int size__, std::complex<float> const* psi_rg__, float wt__, float* density_rg__)
{
    update_density_rg_1_complex_gpu_float(size__, psi_rg__, wt__, density_rg__);
}

void
update_density_rg_1_complex_gpu(int size__, std::complex<double> const* psi_rg__, double wt__, double* density_rg__)
{
    update_density_rg_1_complex_gpu_double(size__, psi_rg__, wt__, density_rg__);
}

void
update_density_rg_2_gpu(int size__, std::complex<float> const* psi_rg_up__, std::complex<float> const* psi_rg_dn__,
                        float wt__, float* density_x_rg__, float* density_y_rg__)
{
    update_density_rg_2_gpu_float(size__, psi_rg_up__, psi_rg_dn__, wt__, density_x_rg__, density_y_rg__);
}

void
update_density_rg_2_gpu(int size__, std::complex<double> const* psi_rg_up__, std::complex<double> const* psi_rg_dn__,
                        double wt__, double* density_x_rg__, double* density_y_rg__)
{
    update_density_rg_2_gpu_double(size__, psi_rg_up__, psi_rg_dn__, wt__, density_x_rg__, density_y_rg__);
}
#endif

Density::Density(Simulation_context& ctx__)
    : Field4D(ctx__, lmax_t(ctx__.lmax_rho()),
              {ctx__.periodic_function_ptr("rho"), ctx__.periodic_function_ptr("magz"),
               ctx__.periodic_function_ptr("magx"), ctx__.periodic_function_ptr("magy")})
    , unit_cell_(ctx_.unit_cell())
{
    PROFILE("sirius::Density");

    if (!ctx_.initialized()) {
        RTE_THROW("Simulation_context is not initialized");
    }

    using spf = Smooth_periodic_function<double>;

    /*  allocate charge density and magnetization on a coarse grid */
    for (int i = 0; i < ctx_.num_mag_dims() + 1; i++) {
        rho_mag_coarse_[i] = std::make_unique<spf>(ctx_.spfft_coarse<double>(), ctx_.gvec_coarse_fft_sptr());
    }

    /* core density of the pseudopotential method */
    if (!ctx_.full_potential()) {
        rho_pseudo_core_ = std::make_unique<spf>(ctx_.spfft<double>(), ctx_.gvec_fft_sptr());
    }

    l_by_lm_ = sf::l_by_lm(ctx_.lmax_rho());

    density_matrix_ = std::make_unique<density_matrix_t>(ctx_.unit_cell(), ctx_.num_mag_comp());

    if (ctx_.hubbard_correction()) {
        occupation_matrix_ = std::make_unique<Occupation_matrix>(ctx_);
    }

    if (unit_cell_.num_paw_atoms()) {
        paw_density_ = std::make_unique<PAW_density<double>>(unit_cell_);
    }

    update();
}

void
Density::update()
{
    PROFILE("sirius::Density::update");

    if (!ctx_.full_potential()) {
        rho_pseudo_core_->zero();
        bool is_empty{true};
        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
            is_empty &= unit_cell_.atom_type(iat).ps_core_charge_density().empty();
        }
        if (!is_empty) {
            generate_pseudo_core_charge_density();
        }
    }
}

/// Find the total leakage of the core states out of the muffin-tins
double
Density::core_leakage() const
{
    double sum{0};
    for (int ic = 0; ic < unit_cell_.num_atom_symmetry_classes(); ic++) {
        sum += core_leakage(ic) * unit_cell_.atom_symmetry_class(ic).num_atoms();
    }
    return sum;
}

void
Density::initial_density()
{
    PROFILE("sirius::Density::initial_density");

    zero();

    if (ctx_.full_potential()) {
        initial_density_full_pot();
    } else {
        initial_density_pseudo();

        init_density_matrix_for_paw();

        generate_paw_density();

        if (occupation_matrix_) {
            occupation_matrix_->init();
        }
    }
    if (ctx_.use_symmetry()) {
        symmetrize_field4d(*this);
    }
}

void
Density::initial_density_pseudo()
{
    /* get lenghts of all G shells */
    auto q = ctx_.gvec().shells_len();
    /* get form-factors for all G shells */
    // TODO: MPI parallelise over G-shells
    auto const ff = ctx_.ri().ps_rho_->values(q, ctx_.comm());
    /* make Vloc(G) */
    auto v = make_periodic_function<true>(ctx_.unit_cell(), ctx_.gvec(), ctx_.phase_factors_t(), ff);

    if (env::print_checksum()) {
        auto z1 = mdarray<std::complex<double>, 1>({ctx_.gvec().count()}, &v[0]).checksum();
        ctx_.comm().allreduce(&z1, 1);
        print_checksum("rho_pw_init", z1, ctx_.out());
    }
    std::copy(v.begin(), v.end(), &rho().rg().f_pw_local(0));

    if (env::print_hash()) {
        auto h = mdarray<std::complex<double>, 1>({ctx_.gvec().count()}, &v[0]).hash();
        print_hash("rho_pw_init", h, ctx_.out());
    }

    double charge = rho().rg().f_0().real() * unit_cell_.omega();

    if (std::abs(charge - unit_cell_.num_valence_electrons()) > 1e-6) {
        std::stringstream s;
        s << "wrong initial charge density" << std::endl
          << "  integral of the density : " << std::setprecision(12) << charge << std::endl
          << "  target number of electrons : " << std::setprecision(12) << unit_cell_.num_valence_electrons();
        if (ctx_.comm().rank() == 0) {
            RTE_WARNING(s);
        }
    }
    rho().rg().fft_transform(1);
    if (env::print_hash()) {
        auto h = rho().rg().values().hash();
        print_hash("rho_rg_init", h, ctx_.out());
    }

    /* remove possible negative noise */
    for (int ir = 0; ir < ctx_.spfft<double>().local_slice_size(); ir++) {
        rho().rg().value(ir) = std::max(rho().rg().value(ir), 0.0);
    }
    /* renormalize charge */
    normalize();

    if (env::print_checksum()) {
        auto cs = rho().rg().checksum_rg();
        print_checksum("rho_rg_init", cs, ctx_.out());
    }

    /* initialize the magnetization */
    if (ctx_.num_mag_dims()) {
        auto Rmt = unit_cell_.find_mt_radii(1, true);

        /* auxiliary weight function; the volume integral of this function is equal to 1 */
        auto w = [](double R, double x) {
            double norm = 3.1886583903476735 * std::pow(R, 3);

            return (1 - std::pow(x / R, 2)) * std::exp(x / R) / norm;
        };

        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
            auto& atom_to_grid_map = ctx_.atoms_to_grid_idx_map(ia);

            auto v = unit_cell_.atom(ia).vector_field();

            for (auto coord : atom_to_grid_map) {
                int ir   = coord.first;
                double r = coord.second;
                double f = w(Rmt[unit_cell_.atom(ia).type_id()], r);
                mag(0).rg().value(ir) += v[2] * f;
                if (ctx_.num_mag_dims() == 3) {
                    mag(1).rg().value(ir) += v[0] * f;
                    mag(2).rg().value(ir) += v[1] * f;
                }
            }
        }
    }
    this->fft_transform(-1);

    if (env::print_checksum()) {
        for (int i = 0; i < ctx_.num_mag_dims() + 1; i++) {
            auto cs  = component(i).rg().checksum_rg();
            auto cs1 = component(i).rg().checksum_pw();
            std::stringstream s;
            s << "component[" << i << "]_rg";
            print_checksum(s.str(), cs, ctx_.out());
            std::stringstream s1;
            s1 << "component[" << i << "]_pw";
            print_checksum(s1.str(), cs1, ctx_.out());
        }
    }
}

void
Density::initial_density_full_pot()
{
    /* initialize smooth density of free atoms */
    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
        unit_cell_.atom_type(iat).init_free_atom_density(true);
    }

    /* compute radial integrals */
    Radial_integrals_rho_free_atom ri(ctx_.unit_cell(), ctx_.pw_cutoff(), 40);

    /* compute contribution from free atoms to the interstitial density */
    auto v = make_periodic_function<true>(ctx_.unit_cell(), ctx_.gvec(), ctx_.phase_factors_t(),
                                          [&ri](int iat, double g) { return ri.value(iat, g); });

    /* initialize density of free atoms (not smoothed) */
    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
        unit_cell_.atom_type(iat).init_free_atom_density(false);
    }

    if (env::print_checksum()) {
        auto z = mdarray<std::complex<double>, 1>({ctx_.gvec().count()}, &v[0]).checksum();
        ctx_.comm().allreduce(&z, 1);
        print_checksum("rho_pw", z, ctx_.out());
    }

    /* set plane-wave coefficients of the charge density */
    std::copy(v.begin(), v.end(), &rho().rg().f_pw_local(0));
    /* convert charge density to real space mesh */
    rho().rg().fft_transform(1);

    if (env::print_checksum()) {
        auto cs = rho().rg().checksum_rg();
        print_checksum("rho_rg", cs, ctx_.out());
    }

    /* remove possible negative noise */
    for (int ir = 0; ir < ctx_.spfft<double>().local_slice_size(); ir++) {
        rho().rg().value(ir) = std::max(0.0, rho().rg().value(ir));
    }

    /* set Y00 component of charge density */
    for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
        int nmtp = ctx_.unit_cell().atom(ia).num_mt_points();

        for (int ir = 0; ir < nmtp; ir++) {
            double x              = ctx_.unit_cell().atom(ia).radial_grid(ir);
            rho().mt()[ia](0, ir) = unit_cell_.atom(ia).type().free_atom_density(x) / y00;
        }
    }

    int lmax  = ctx_.lmax_rho();
    int lmmax = sf::lmmax(lmax);

    auto l_by_lm = sf::l_by_lm(lmax);

    std::vector<std::complex<double>> zil(lmax + 1);
    for (int l = 0; l <= lmax; l++) {
        zil[l] = std::pow(std::complex<double>(0, 1), l);
    }

    /* compute boundary value at MT sphere from the plane-wave expansion */
    auto gvec_ylm = generate_gvec_ylm(ctx_, lmax);

    auto sbessel_mt = generate_sbessel_mt(ctx_, lmax);

    auto flm = sum_fg_fl_yg(ctx_, lmax, v.at(memory_t::host), sbessel_mt, gvec_ylm);

    /* this is the difference between the value of periodic charge density at MT boundary and
       a value of the atom's free density at the boundary */
    for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
        double R = ctx_.unit_cell().atom(ia).mt_radius();
        double c = unit_cell_.atom(ia).type().free_atom_density(R) / y00;
        flm(0, ia) -= c;
    }

    /* match density at MT */
    for (int iat = 0; iat < ctx_.unit_cell().num_atom_types(); iat++) {
        mdarray<double, 2> rRl({ctx_.unit_cell().max_num_mt_points(), lmax + 1});
        double R = ctx_.unit_cell().atom_type(iat).mt_radius();
        int nmtp = ctx_.unit_cell().atom_type(iat).num_mt_points();

        #pragma omp parallel for default(shared)
        for (int l = 0; l <= lmax; l++) {
            for (int ir = 0; ir < nmtp; ir++) {
                rRl(ir, l) = std::pow(ctx_.unit_cell().atom_type(iat).radial_grid(ir) / R, 2);
            }
        }
        #pragma omp parallel for default(shared)
        for (int i = 0; i < unit_cell_.atom_type(iat).num_atoms(); i++) {
            int ia = unit_cell_.atom_type(iat).atom_id(i);
            std::vector<double> glm(lmmax);
            SHT::convert(lmax, &flm(0, ia), &glm[0]);
            for (int lm = 0; lm < lmmax; lm++) {
                int l = l_by_lm[lm];
                for (int ir = 0; ir < nmtp; ir++) {
                    rho().mt()[ia](lm, ir) += glm[lm] * rRl(ir, l);
                }
            }
        }
    }

    /* normalize charge density */
    normalize();

    check_num_electrons();

    // FILE* fout = fopen("rho.dat", "w");
    // for (int i = 0; i <= 10000; i++) {
    //    r3::vector<double> v = (i / 10000.0) * r3::vector<double>({10.26, 10.26, 10.26});
    //    double val = rho().value(v);
    //    fprintf(fout, "%18.12f %18.12f\n", v.length(), val);
    //}
    // fclose(fout);

    // FILE* fout2 = fopen("rho_rg.dat", "w");
    // for (int i = 0; i <= 10000; i++) {
    //    r3::vector<double> v = (i / 10000.0) * r3::vector<double>({10.26, 10.26, 10.26});
    //    double val = rho().value_rg(v);
    //    fprintf(fout2, "%18.12f %18.12f\n", v.length(), val);
    //}
    // fclose(fout2);

    /* initialize the magnetization */
    if (ctx_.num_mag_dims()) {
        for (auto it : unit_cell_.spl_num_atoms()) {
            int ia   = it.i;
            auto v   = unit_cell_.atom(ia).vector_field();
            auto len = v.length();

            int nmtp = unit_cell_.atom(ia).num_mt_points();
            Spline<double> rho_s(unit_cell_.atom(ia).type().radial_grid());
            double R = unit_cell_.atom(ia).mt_radius();
            for (int ir = 0; ir < nmtp; ir++) {
                double x  = unit_cell_.atom(ia).type().radial_grid(ir);
                rho_s(ir) = this->rho().mt()[ia](0, ir) * y00 * (1 - 3 * std::pow(x / R, 2) + 2 * std::pow(x / R, 3));
            }

            /* maximum magnetization which can be achieved if we smooth density towards MT boundary */
            double q = fourpi * rho_s.interpolate().integrate(2);

            /* if very strong initial magnetization is given */
            if (q < len) {
                /* renormalize starting magnetization */
                for (int x : {0, 1, 2}) {
                    v[x] *= (q / len);
                }
                len = q;
            }

            if (len > 1e-8) {
                for (int ir = 0; ir < nmtp; ir++) {
                    mag(0).mt()[ia](0, ir) = rho_s(ir) * v[2] / q / y00;
                }
                if (ctx_.num_mag_dims() == 3) {
                    for (int ir = 0; ir < nmtp; ir++) {
                        mag(1).mt()[ia](0, ir) = rho_s(ir) * v[0] / q / y00;
                        mag(2).mt()[ia](0, ir) = rho_s(ir) * v[1] / q / y00;
                    }
                }
            }
        }
    }
}

void
Density::init_density_matrix_for_paw()
{
    for (int ipaw = 0; ipaw < unit_cell_.num_paw_atoms(); ipaw++) {
        int ia = unit_cell_.paw_atom_index(paw_atom_index_t::global(ipaw));

        auto& dm = density_matrix(ia);
        dm.zero();

        auto& atom      = unit_cell_.atom(ia);
        auto& atom_type = atom.type();

        int nbf = atom_type.mt_basis_size();

        auto& occupations = atom_type.paw_wf_occ();

        /* magnetization vector */
        auto magn = atom.vector_field();

        for (int xi = 0; xi < nbf; xi++) {
            auto& basis_func_index_dsc = atom_type.indexb()[xi];

            int rad_func_index = basis_func_index_dsc.idxrf;

            double occ = occupations[rad_func_index];

            int l = basis_func_index_dsc.am.l();

            switch (ctx_.num_mag_dims()) {
                case 0: {
                    dm(xi, xi, 0) = occ / double(2 * l + 1);
                    break;
                }

                case 3:
                case 1: {
                    double nm     = (std::abs(magn[2]) < 1.0) ? magn[2] : std::copysign(1, magn[2]);
                    dm(xi, xi, 0) = 0.5 * (1.0 + nm) * occ / double(2 * l + 1);
                    dm(xi, xi, 1) = 0.5 * (1.0 - nm) * occ / double(2 * l + 1);
                    break;
                }
            }
        }
    }
}

void
Density::generate_paw_density(paw_atom_index_t::local ialoc__)
{
    auto ia_paw = ctx_.unit_cell().spl_num_paw_atoms(ialoc__);
    auto ia     = ctx_.unit_cell().paw_atom_index(ia_paw);

    auto& atom_type = ctx_.unit_cell().atom(ia).type();

    auto lmax = atom_type.indexr().lmax();

    auto l_by_lm = sf::l_by_lm(2 * lmax);

    /* get gaunt coefficients */
    Gaunt_coefficients<double> GC(lmax, 2 * lmax, lmax, SHT::gaunt_rrr);

    paw_density_->zero(ia);

    /* get radial grid to divide density over r^2 */
    auto& grid = atom_type.radial_grid();

    auto& paw_ae_wfs = atom_type.ae_paw_wfs_array();
    auto& paw_ps_wfs = atom_type.ps_paw_wfs_array();

    auto dm = this->density_matrix_aux(atom_index_t::global(ia));

    for (int imagn = 0; imagn < ctx_.num_mag_dims() + 1; imagn++) {
        auto& ae_dens = paw_density_->ae_density(imagn, ia);
        auto& ps_dens = paw_density_->ps_density(imagn, ia);

        /* iterate over local basis functions (or over lm1 and lm2) */
        for (int xi2 = 0; xi2 < atom_type.indexb().size(); xi2++) {
            int lm2  = atom_type.indexb(xi2).lm;
            int irb2 = atom_type.indexb(xi2).idxrf;

            for (int xi1 = 0; xi1 <= xi2; xi1++) {
                int lm1  = atom_type.indexb(xi1).lm;
                int irb1 = atom_type.indexb(xi1).idxrf;

                /* get num of non-zero GC */
                int num_non_zero_gc = GC.num_gaunt(lm1, lm2);

                double diag_coef = (xi1 == xi2) ? 1.0 : 2.0;

                auto idx = packed_index(xi1, xi2);

                /* add nonzero coefficients */
                for (int inz = 0; inz < num_non_zero_gc; inz++) {
                    auto& lm3coef = GC.gaunt(lm1, lm2, inz);

                    /* iterate over radial points */
                    for (int irad = 0; irad < grid.num_points(); irad++) {

                        /* we need to divide density over r^2 since wave functions are stored multiplied by r */
                        double inv_r2 = diag_coef / (grid[irad] * grid[irad]);

                        /* calculate unified density/magnetization
                         * dm_ij * GauntCoef * ( phi_i phi_j  +  Q_ij) */
                        ae_dens(lm3coef.lm3, irad) += dm(idx, imagn) * inv_r2 * lm3coef.coef * paw_ae_wfs(irad, irb1) *
                                                      paw_ae_wfs(irad, irb2);
                        ps_dens(lm3coef.lm3, irad) +=
                                dm(idx, imagn) * inv_r2 * lm3coef.coef *
                                (paw_ps_wfs(irad, irb1) * paw_ps_wfs(irad, irb2) +
                                 atom_type.q_radial_function(irb1, irb2, l_by_lm[lm3coef.lm3])(irad));
                    }
                }
            }
        }
    }
}

void
Density::generate_paw_density()
{
    if (!unit_cell_.num_paw_atoms()) {
        return;
    }

    PROFILE("sirius::Density::generate_paw_density");

    #pragma omp parallel for
    for (auto it : unit_cell_.spl_num_paw_atoms()) {
        generate_paw_density(it.li);
    }
}

/// Compute non-magnetic or up- or dn- contribution of the wave-functions to the charge density.
template <typename T>
static void
add_k_point_contribution_rg_collinear(fft::spfft_transform_type<T>& fft__, int ispn__, T w__, T const* inp_wf__,
                                      int nr__, bool gamma__, mdarray<T, 2>& density_rg__)
{
    /* transform to real space */
    fft__.backward(inp_wf__, fft__.processing_unit());

    /* location of the real-space wave-functions psi(r) */
    auto data_ptr = fft__.space_domain_data(fft__.processing_unit());

    switch (fft__.processing_unit()) {
        case SPFFT_PU_HOST: {
            if (gamma__) {
                #pragma omp parallel for
                for (int ir = 0; ir < nr__; ir++) {
                    density_rg__(ir, ispn__) += w__ * std::pow(data_ptr[ir], 2);
                }
            } else {
                auto data = reinterpret_cast<std::complex<T>*>(data_ptr);
                #pragma omp parallel for
                for (int ir = 0; ir < nr__; ir++) {
                    auto z = data[ir];
                    density_rg__(ir, ispn__) += w__ * (std::pow(z.real(), 2) + std::pow(z.imag(), 2));
                }
            }
            break;
        }
        case SPFFT_PU_GPU: {
#if defined(SIRIUS_GPU)
            if (gamma__) {
                update_density_rg_1_real_gpu(nr__, data_ptr, w__, density_rg__.at(memory_t::device, 0, ispn__));
            } else {
                auto data = reinterpret_cast<std::complex<T>*>(data_ptr);
                update_density_rg_1_complex_gpu(nr__, data, w__, density_rg__.at(memory_t::device, 0, ispn__));
            }
#endif
            break;
        }
    }
}

/// Compute contribution to density and megnetisation from the 2-component spinor wave-functions.
template <typename T>
static void
add_k_point_contribution_rg_noncollinear(fft::spfft_transform_type<T>& fft__, T w__, T const* inp_wf_up__,
                                         T const* inp_wf_dn__, int nr__, mdarray<std::complex<T>, 1>& psi_r_up__,
                                         mdarray<T, 2>& density_rg__)
{
    /* location of the real-space wave-functions psi(r) */
    auto data_ptr = fft__.space_domain_data(fft__.processing_unit());

    /* transform up- component to real space */
    fft__.backward(inp_wf_up__, fft__.processing_unit());

    /* this is a non-collinear case, so the wave-functions and FFT buffer are complex */
    switch (fft__.processing_unit()) {
        case SPFFT_PU_HOST: {
            auto inp = reinterpret_cast<std::complex<T>*>(data_ptr);
            std::copy(inp, inp + nr__, psi_r_up__.at(memory_t::host));
            break;
        }
        case SPFFT_PU_GPU: {
            acc::copy(psi_r_up__.at(memory_t::device), reinterpret_cast<std::complex<T>*>(data_ptr), nr__);
            break;
        }
    }

    /* transform to real space */
    fft__.backward(inp_wf_dn__, fft__.processing_unit());

    /* alias for dn- component of wave-functions */
    auto psi_r_dn  = reinterpret_cast<std::complex<T>*>(data_ptr);
    auto& psi_r_up = psi_r_up__;

    switch (fft__.processing_unit()) {
        case SPFFT_PU_HOST: {
            #pragma omp parallel for
            for (int ir = 0; ir < nr__; ir++) {
                auto r0 = (std::pow(psi_r_up[ir].real(), 2) + std::pow(psi_r_up[ir].imag(), 2)) * w__;
                auto r1 = (std::pow(psi_r_dn[ir].real(), 2) + std::pow(psi_r_dn[ir].imag(), 2)) * w__;

                auto z2 = psi_r_up[ir] * std::conj(psi_r_dn[ir]) * std::complex<T>(w__, 0);

                density_rg__(ir, 0) += r0;
                density_rg__(ir, 1) += r1;
                density_rg__(ir, 2) += 2.0 * std::real(z2);
                density_rg__(ir, 3) -= 2.0 * std::imag(z2);
            }
            break;
        }
        case SPFFT_PU_GPU: {
#ifdef SIRIUS_GPU
            /* add up-up contribution */
            update_density_rg_1_complex_gpu(nr__, psi_r_up.at(memory_t::device), w__,
                                            density_rg__.at(memory_t::device, 0, 0));
            /* add dn-dn contribution */
            update_density_rg_1_complex_gpu(nr__, psi_r_dn, w__, density_rg__.at(memory_t::device, 0, 1));
            /* add off-diagonal contribution */
            update_density_rg_2_gpu(nr__, psi_r_up.at(memory_t::device), psi_r_dn, w__,
                                    density_rg__.at(memory_t::device, 0, 2), density_rg__.at(memory_t::device, 0, 3));
#endif
            break;
        }
    }
}

template <typename T>
void
Density::add_k_point_contribution_rg(K_point<T>* kp__, std::array<wf::Wave_functions_fft<T>, 2>& wf_fft__)
{
    PROFILE("sirius::Density::add_k_point_contribution_rg");

    double omega = unit_cell_.omega();

    auto& fft = ctx_.spfft_coarse<T>();

    /* local number of real-space points */
    int nr = fft.local_slice_size();

    /* get preallocated memory */
    mdarray<T, 2> density_rg({nr, ctx_.num_mag_dims() + 1}, get_memory_pool(memory_t::host),
                             mdarray_label("density_rg"));
    density_rg.zero();

    if (fft.processing_unit() == SPFFT_PU_GPU) {
        density_rg.allocate(get_memory_pool(memory_t::device)).zero(memory_t::device);
    }

    /* non-magnetic or collinear case */
    if (ctx_.num_mag_dims() != 3) {
        /* loop over pure spinor components */
        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            /* where fft-transformed wave-functions are stored */
            auto wf_mem = wf_fft__[ispn].on_device() ? memory_t::device : memory_t::host;
            /* local number of bands for a fft-distribution */
            int nbnd = wf_fft__[ispn].num_wf_local();
            for (int i = 0; i < nbnd; i++) {
                auto j = wf_fft__[ispn].spl_num_wf().global_index(i);
                T w    = kp__->band_occupancy(j, ispn) * kp__->weight() / omega;

                auto inp_wf = wf_fft__[ispn].pw_coeffs_spfft(wf_mem, wf::band_index(i));

                add_k_point_contribution_rg_collinear(kp__->spfft_transform(), ispn, w, inp_wf, nr, ctx_.gamma_point(),
                                                      density_rg);
            }
        }    // ispn
    } else { /* non-collinear case */
        /* allocate on CPU or GPU */
        mdarray<std::complex<T>, 1> psi_r_up({nr}, get_memory_pool(memory_t::host));
        if (fft.processing_unit() == SPFFT_PU_GPU) {
            psi_r_up.allocate(get_memory_pool(memory_t::device));
        }

        RTE_ASSERT(wf_fft__[0].num_wf_local() == wf_fft__[1].num_wf_local());

        int nbnd = wf_fft__[0].num_wf_local();
        for (int i = 0; i < nbnd; i++) {
            auto j = wf_fft__[0].spl_num_wf().global_index(i);
            T w    = kp__->band_occupancy(j, 0) * kp__->weight() / omega;

            auto wf_mem_up = wf_fft__[0].on_device() ? memory_t::device : memory_t::host;
            auto wf_mem_dn = wf_fft__[1].on_device() ? memory_t::device : memory_t::host;

            /* up- and dn- components */
            auto inp_wf_up = wf_fft__[0].pw_coeffs_spfft(wf_mem_up, wf::band_index(i));
            auto inp_wf_dn = wf_fft__[1].pw_coeffs_spfft(wf_mem_dn, wf::band_index(i));

            add_k_point_contribution_rg_noncollinear(kp__->spfft_transform(), w, inp_wf_up, inp_wf_dn, nr, psi_r_up,
                                                     density_rg);
        }
    }

    if (fft.processing_unit() == SPFFT_PU_GPU) {
        density_rg.copy_to(memory_t::host);
    }

    /* switch from real density matrix to density and magnetization */
    switch (ctx_.num_mag_dims()) {
        case 3: {
            #pragma omp parallel for
            for (int ir = 0; ir < nr; ir++) {
                rho_mag_coarse_[2]->value(ir) += density_rg(ir, 2); // Mx
                rho_mag_coarse_[3]->value(ir) += density_rg(ir, 3); // My
            }
        }
        case 1: {
            #pragma omp parallel for
            for (int ir = 0; ir < nr; ir++) {
                rho_mag_coarse_[0]->value(ir) += (density_rg(ir, 0) + density_rg(ir, 1)); // rho
                rho_mag_coarse_[1]->value(ir) += (density_rg(ir, 0) - density_rg(ir, 1)); // Mz
            }
            break;
        }
        case 0: {
            #pragma omp parallel for
            for (int ir = 0; ir < nr; ir++) {
                rho_mag_coarse_[0]->value(ir) += density_rg(ir, 0); // rho
            }
        }
    }
}

template <typename T>
static void
add_k_point_contribution_dm_fplapw(Simulation_context const& ctx__, K_point<T> const& kp__,
                                   density_matrix_t& density_matrix__)
{
    PROFILE("sirius::add_k_point_contribution_dm_fplapw");

    auto& uc = ctx__.unit_cell();

    auto one = la::constant<std::complex<double>>::one();

    /* add |psi_j> n_j <psi_j| to density matrix */
    #pragma omp parallel
    {
        mdarray<std::complex<double>, 3> wf1({uc.max_mt_basis_size(), ctx__.num_bands(), ctx__.num_spins()});
        mdarray<std::complex<double>, 3> wf2({uc.max_mt_basis_size(), ctx__.num_bands(), ctx__.num_spins()});
        #pragma omp for
        for (auto it : kp__.spinor_wave_functions().spl_num_atoms()) {
            int ia            = it.i;
            int mt_basis_size = uc.atom(ia).type().mt_basis_size();

            for (int ispn = 0; ispn < ctx__.num_spins(); ispn++) {
                for (int j = 0; j < kp__.num_occupied_bands(ispn); j++) {
                    for (int xi = 0; xi < mt_basis_size; xi++) {
                        auto z           = kp__.spinor_wave_functions().mt_coeffs(xi, it.li, wf::spin_index(ispn),
                                                                                  wf::band_index(j));
                        wf1(xi, j, ispn) = std::conj(z);
                        wf2(xi, j, ispn) =
                                static_cast<std::complex<double>>(z) * kp__.band_occupancy(j, ispn) * kp__.weight();
                    }
                }
            }

            /* compute diagonal terms */
            for (int ispn = 0; ispn < ctx__.num_spins(); ispn++) {
                la::wrap(la::lib_t::blas)
                        .gemm('N', 'T', mt_basis_size, mt_basis_size, kp__.num_occupied_bands(ispn), &one,
                              &wf1(0, 0, ispn), wf1.ld(), &wf2(0, 0, ispn), wf2.ld(), &one,
                              density_matrix__[ia].at(memory_t::host, 0, 0, ispn), density_matrix__[ia].ld());
            }
            /* offdiagonal term */
            if (ctx__.num_mag_dims() == 3) {
                la::wrap(la::lib_t::blas)
                        .gemm('N', 'T', mt_basis_size, mt_basis_size, kp__.num_occupied_bands(), &one, &wf1(0, 0, 0),
                              wf1.ld(), &wf2(0, 0, 1), wf2.ld(), &one, density_matrix__[ia].at(memory_t::host, 0, 0, 2),
                              density_matrix__[ia].ld());
            }
        }
    }
}

template <typename T, typename F>
static void
add_k_point_contribution_dm_pwpp_collinear(Simulation_context& ctx__, K_point<T>& kp__,
                                           beta_projectors_coeffs_t<T>& bp_coeffs__, density_matrix_t& density_matrix__)
{
    /* number of beta projectors */
    int nbeta = bp_coeffs__.beta_chunk_->num_beta_;
    auto mt   = ctx__.processing_unit_memory_t();

    for (int ispn = 0; ispn < ctx__.num_spins(); ispn++) {
        /* total number of occupied bands for this spin */
        int nbnd = kp__.num_occupied_bands(ispn);
        /* compute <beta|psi> */
        auto beta_psi =
                inner_prod_beta<F>(ctx__.spla_context(), mt, ctx__.host_memory_t(), is_device_memory(mt), bp_coeffs__,
                                   kp__.spinor_wave_functions(), wf::spin_index(ispn), wf::band_range(0, nbnd));

        /* use communicator of the k-point to split band index */
        splindex_block<> spl_nbnd(nbnd, n_blocks(kp__.comm().size()), block_id(kp__.comm().rank()));

        int nbnd_loc = spl_nbnd.local_size();
        #pragma omp parallel
        if (nbnd_loc) { // TODO: this part can also be moved to GPU
            /* auxiliary arrays */
            mdarray<std::complex<double>, 2> bp1({nbeta, nbnd_loc});
            mdarray<std::complex<double>, 2> bp2({nbeta, nbnd_loc});
            #pragma omp for
            for (int ia = 0; ia < bp_coeffs__.beta_chunk_->num_atoms_; ia++) {
                int nbf = bp_coeffs__.beta_chunk_->desc_(beta_desc_idx::nbf, ia);
                if (!nbf) {
                    continue;
                }
                int offs = bp_coeffs__.beta_chunk_->desc_(beta_desc_idx::offset, ia);
                int ja   = bp_coeffs__.beta_chunk_->desc_(beta_desc_idx::ia, ia);

                for (int i = 0; i < nbnd_loc; i++) {
                    /* global index of band */
                    auto j = spl_nbnd.global_index(i);

                    for (int xi = 0; xi < nbf; xi++) {
                        bp1(xi, i) = beta_psi(offs + xi, j);
                        bp2(xi, i) = std::conj(bp1(xi, i)) * kp__.weight() * kp__.band_occupancy(j, ispn);
                    }
                }

                la::wrap(la::lib_t::blas)
                        .gemm('N', 'T', nbf, nbf, nbnd_loc, &la::constant<std::complex<double>>::one(), &bp1(0, 0),
                              bp1.ld(), &bp2(0, 0), bp2.ld(), &la::constant<std::complex<double>>::one(),
                              &density_matrix__[ja](0, 0, ispn), density_matrix__[ja].ld());
            }
        }
    } // ispn
}

template <typename T, typename F>
static void
add_k_point_contribution_dm_pwpp_noncollinear(Simulation_context& ctx__, K_point<T>& kp__,
                                              beta_projectors_coeffs_t<T>& bp_coeffs__,
                                              density_matrix_t& density_matrix__)
{
    /* number of beta projectors */
    int nbeta = bp_coeffs__.beta_chunk_->num_beta_;

    /* total number of occupied bands */
    int nbnd = kp__.num_occupied_bands();

    splindex_block<> spl_nbnd(nbnd, n_blocks(kp__.comm().size()), block_id(kp__.comm().rank()));
    int nbnd_loc = spl_nbnd.local_size();

    /* auxiliary arrays */
    mdarray<std::complex<double>, 3> bp1({nbeta, nbnd_loc, ctx__.num_spins()});
    mdarray<std::complex<double>, 3> bp2({nbeta, nbnd_loc, ctx__.num_spins()});

    auto& uc = ctx__.unit_cell();

    auto mt = ctx__.processing_unit_memory_t();
    for (int ispn = 0; ispn < ctx__.num_spins(); ispn++) {
        /* compute <beta|psi> */
        auto beta_psi =
                inner_prod_beta<F>(ctx__.spla_context(), mt, ctx__.host_memory_t(), is_device_memory(mt), bp_coeffs__,
                                   kp__.spinor_wave_functions(), wf::spin_index(ispn), wf::band_range(0, nbnd));

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < nbnd_loc; i++) {
            auto j = spl_nbnd.global_index(i);

            for (int m = 0; m < nbeta; m++) {
                bp1(m, i, ispn) = beta_psi(m, j);
                bp2(m, i, ispn) = std::conj(beta_psi(m, j));
                bp2(m, i, ispn) *= kp__.weight() * kp__.band_occupancy(j, 0);
            }
        }
    }
    for (int ia = 0; ia < bp_coeffs__.beta_chunk_->num_atoms_; ia++) {
        int nbf = bp_coeffs__.beta_chunk_->desc_(beta_desc_idx::nbf, ia);
        if (!nbf) {
            continue;
        }
        int offs = bp_coeffs__.beta_chunk_->desc_(beta_desc_idx::offset, ia);
        int ja   = bp_coeffs__.beta_chunk_->desc_(beta_desc_idx::ia, ia);
        if (uc.atom(ja).type().spin_orbit_coupling()) {
            mdarray<std::complex<double>, 3> bp3({nbf, nbnd_loc, 2});
            bp3.zero();
            /* We already have the <beta|psi> but we need to rotate
             *  them when the spin orbit interaction is included in the
             *  pseudo potential.
             *
             *  We rotate \f[\langle\beta|\psi\rangle\f] accordingly by multiplying it with
             *  the \f[f^{\sigma\sigma^{'}}_{\xi,\xi^'}\f]
             */

            for (int xi1 = 0; xi1 < nbf; xi1++) {
                for (int i = 0; i < nbnd_loc; i++) {
                    for (int xi1p = 0; xi1p < nbf; xi1p++) {
                        if (uc.atom(ja).type().compare_index_beta_functions(xi1, xi1p)) {
                            bp3(xi1, i, 0) +=
                                    bp1(offs + xi1p, i, 0) * uc.atom(ja).type().f_coefficients(xi1, xi1p, 0, 0) +
                                    bp1(offs + xi1p, i, 1) * uc.atom(ja).type().f_coefficients(xi1, xi1p, 0, 1);
                            bp3(xi1, i, 1) +=
                                    bp1(offs + xi1p, i, 0) * uc.atom(ja).type().f_coefficients(xi1, xi1p, 1, 0) +
                                    bp1(offs + xi1p, i, 1) * uc.atom(ja).type().f_coefficients(xi1, xi1p, 1, 1);
                        }
                    }
                }
            }

            for (int xi1 = 0; xi1 < nbf; xi1++) {
                for (int i = 0; i < nbnd_loc; i++) {
                    bp1(offs + xi1, i, 0) = bp3(xi1, i, 0);
                    bp1(offs + xi1, i, 1) = bp3(xi1, i, 1);
                }
            }

            bp3.zero();

            for (int xi1 = 0; xi1 < nbf; xi1++) {
                for (int i = 0; i < nbnd_loc; i++) {
                    for (int xi1p = 0; xi1p < nbf; xi1p++) {
                        if (uc.atom(ja).type().compare_index_beta_functions(xi1, xi1p)) {
                            bp3(xi1, i, 0) +=
                                    bp2(offs + xi1p, i, 0) * uc.atom(ja).type().f_coefficients(xi1p, xi1, 0, 0) +
                                    bp2(offs + xi1p, i, 1) * uc.atom(ja).type().f_coefficients(xi1p, xi1, 1, 0);
                            bp3(xi1, i, 1) +=
                                    bp2(offs + xi1p, i, 0) * uc.atom(ja).type().f_coefficients(xi1p, xi1, 0, 1) +
                                    bp2(offs + xi1p, i, 1) * uc.atom(ja).type().f_coefficients(xi1p, xi1, 1, 1);
                        }
                    }
                }
            }

            for (int xi1 = 0; xi1 < nbf; xi1++) {
                for (int i = 0; i < nbnd_loc; i++) {
                    bp2(offs + xi1, i, 0) = bp3(xi1, i, 0);
                    bp2(offs + xi1, i, 1) = bp3(xi1, i, 1);
                }
            }
        }
    }

    if (nbnd_loc) {
        #pragma omp parallel for
        for (int ia = 0; ia < bp_coeffs__.beta_chunk_->num_atoms_; ia++) {
            int nbf  = bp_coeffs__.beta_chunk_->desc_(beta_desc_idx::nbf, ia);
            int offs = bp_coeffs__.beta_chunk_->desc_(beta_desc_idx::offset, ia);
            int ja   = bp_coeffs__.beta_chunk_->desc_(beta_desc_idx::ia, ia);
            /* compute diagonal spin blocks */
            for (int ispn = 0; ispn < 2; ispn++) {
                la::wrap(la::lib_t::blas)
                        .gemm('N', 'T', nbf, nbf, nbnd_loc, &la::constant<std::complex<double>>::one(),
                              &bp1(offs, 0, ispn), bp1.ld(), &bp2(offs, 0, ispn), bp2.ld(),
                              &la::constant<std::complex<double>>::one(), &density_matrix__[ja](0, 0, ispn),
                              density_matrix__[ja].ld());
            }
            /* off-diagonal spin block */
            la::wrap(la::lib_t::blas)
                    .gemm('N', 'T', nbf, nbf, nbnd_loc, &la::constant<std::complex<double>>::one(), &bp1(offs, 0, 0),
                          bp1.ld(), &bp2(offs, 0, 1), bp2.ld(), &la::constant<std::complex<double>>::one(),
                          &density_matrix__[ja](0, 0, 2), density_matrix__[ja].ld());
        }
    }
}

template <typename T, typename F>
static void
add_k_point_contribution_dm_pwpp(Simulation_context& ctx__, K_point<T>& kp__, density_matrix_t& density_matrix__)
{
    PROFILE("sirius::add_k_point_contribution_dm_pwpp");

    if (!ctx__.unit_cell().max_mt_basis_size()) {
        return;
    }

    auto bp_gen    = kp__.beta_projectors().make_generator();
    auto bp_coeffs = bp_gen.prepare();

    for (int ichunk = 0; ichunk < kp__.beta_projectors().num_chunks(); ichunk++) {
        // kp__.beta_projectors().generate(ctx__.processing_unit_memory_t(), ichunk);
        bp_gen.generate(bp_coeffs, ichunk);

        if (ctx__.num_mag_dims() != 3) {
            add_k_point_contribution_dm_pwpp_collinear<T, F>(ctx__, kp__, bp_coeffs, density_matrix__);
        } else {
            add_k_point_contribution_dm_pwpp_noncollinear<T, F>(ctx__, kp__, bp_coeffs, density_matrix__);
        }
    }
}

void
Density::normalize()
{
    double nel   = rho().integrate().total;
    double scale = unit_cell_.num_electrons() / nel;

    /* renormalize interstitial part */
    for (int ir = 0; ir < ctx_.spfft<double>().local_slice_size(); ir++) {
        rho().rg().value(ir) *= scale;
    }
    if (ctx_.full_potential()) {
        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
            for (int ir = 0; ir < unit_cell_.atom(ia).num_mt_points(); ir++) {
                for (int lm = 0; lm < ctx_.lmmax_rho(); lm++) {
                    rho().mt()[ia](lm, ir) *= scale;
                }
            }
        }
    }
}

/// Check total density for the correct number of electrons.
bool
Density::check_num_electrons() const
{
    double nel{0};
    if (ctx_.full_potential()) {
        nel = rho().integrate().total;
    } else {
        nel = rho().rg().f_0().real() * unit_cell_.omega();
    }

    /* check the number of electrons */
    if (std::abs(nel - unit_cell_.num_electrons()) > 1e-5 && ctx_.comm().rank() == 0) {
        std::stringstream s;
        s << "wrong number of electrons" << std::endl
          << "  obtained value : " << nel << std::endl
          << "  target value : " << unit_cell_.num_electrons() << std::endl
          << "  difference : " << std::abs(nel - unit_cell_.num_electrons()) << std::endl;
        if (ctx_.full_potential()) {
            s << "  total core leakage : " << core_leakage();
            for (int ic = 0; ic < unit_cell_.num_atom_symmetry_classes(); ic++) {
                s << std::endl << "    atom class : " << ic << ", core leakage : " << core_leakage(ic);
            }
        }
        RTE_WARNING(s);
        return false;
    } else {
        return true;
    }
}

template <typename T>
void
Density::generate(K_point_set const& ks__, bool symmetrize__, bool add_core__, bool transform_to_rg__)
{
    PROFILE("sirius::Density::generate");

    generate_valence<T>(ks__);

    if (ctx_.full_potential()) {
        if (add_core__) {
            /* find the core states */
            generate_core_charge_density();
            /* add core contribution */
            for (auto it : unit_cell_.spl_num_atoms()) {
                for (int ir = 0; ir < unit_cell_.atom(it.i).num_mt_points(); ir++) {
                    rho().mt()[it.i](0, ir) += unit_cell_.atom(it.i).symmetry_class().ae_core_charge_density(ir) / y00;
                }
            }
        }
        /* synchronize muffin-tin part */
        for (int iv = 0; iv < ctx_.num_mag_dims() + 1; iv++) {
            this->component(iv).mt().sync(ctx_.unit_cell().spl_num_atoms());
        }
    }
    if (symmetrize__) {
        symmetrize_field4d(*this);
        if (ctx_.electronic_structure_method() == electronic_structure_method_t::pseudopotential) {
            std::unique_ptr<density_matrix_t> dm_ref;
            std::unique_ptr<Occupation_matrix> om_ref;

            /* copy density matrix for future comparison */
            if (ctx_.cfg().control().verification() >= 1 && ctx_.cfg().parameters().use_ibz() == false) {
                dm_ref = std::make_unique<density_matrix_t>(unit_cell_, ctx_.num_mag_comp());
                copy(*density_matrix_, *dm_ref);
            }
            if (ctx_.cfg().control().verification() >= 1 && ctx_.cfg().parameters().use_ibz() == false &&
                occupation_matrix_) {
                om_ref = std::make_unique<Occupation_matrix>(ctx_);
                copy(*occupation_matrix_, *om_ref);
            }

            /* symmetrize density matrix (used in standard uspp case) */
            if (unit_cell_.max_mt_basis_size() != 0) {
                sirius::symmetrize_density_matrix(unit_cell_, *density_matrix_, ctx_.num_mag_comp());
            }

            if (occupation_matrix_) {
                /* all symmetrization is done in the occupation_matrix class */
                symmetrize_occupation_matrix(*occupation_matrix_);
            }

            /* compare with reference density matrix */
            if (ctx_.cfg().control().verification() >= 1 && ctx_.cfg().parameters().use_ibz() == false) {
                double diff{0};
                for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
                    for (size_t i = 0; i < (*density_matrix_)[ia].size(); i++) {
                        diff = std::max(diff, std::abs((*dm_ref)[ia][i] - (*density_matrix_)[ia][i]));
                    }
                }
                std::string status = (diff > 1e-8) ? "Fail" : "OK";
                if (ctx_.verbosity() >= 1) {
                    RTE_OUT(ctx_.out()) << "error of density matrix symmetrization: " << diff << " " << status
                                        << std::endl;
                }
            }
            /* compare with reference occupation matrix */
            if (ctx_.cfg().control().verification() >= 1 && ctx_.cfg().parameters().use_ibz() == false &&
                occupation_matrix_) {
                double diff1{0};
                for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
                    if (ctx_.unit_cell().atom(ia).type().hubbard_correction()) {
                        for (size_t i = 0; i < occupation_matrix_->local(ia).size(); i++) {
                            diff1 = std::max(diff1, std::abs(om_ref->local(ia)[i] - occupation_matrix_->local(ia)[i]));
                        }
                    }
                }
                std::string status = (diff1 > 1e-8) ? "Fail" : "OK";
                if (ctx_.verbosity() >= 1) {
                    RTE_OUT(ctx_.out()) << "error of LDA+U local occupation matrix symmetrization: " << diff1 << " "
                                        << status << std::endl;
                }

                om_ref->update_nonlocal();

                double diff2{0};
                for (size_t i = 0; i < occupation_matrix_->nonlocal().size(); i++) {
                    for (size_t j = 0; j < occupation_matrix_->nonlocal(i).size(); j++) {
                        diff2 = std::max(diff2, std::abs(om_ref->nonlocal(i)[j] - occupation_matrix_->nonlocal(i)[j]));
                    }
                }
                status = (diff2 > 1e-8) ? "Fail" : "OK";
                if (ctx_.verbosity() >= 1) {
                    RTE_OUT(ctx_.out()) << "error of LDA+U nonlocal occupation matrix symmetrization: " << diff2 << " "
                                        << status << std::endl;
                }
            }
        }
    } else { /* if we don't symmetrize, we still need to copy nonlocal part of occupation matrix */
        if (occupation_matrix_) {
            occupation_matrix_->update_nonlocal();
        }
    }

    if (occupation_matrix_) {
        occupation_matrix_->print_occupancies(2);
        // calculate the lagrange multiplier and resulting error
        if (ctx_.cfg().hubbard().constrained_calculation()) {
            occupation_matrix_->calculate_constraints_and_error();
        }
    }

    generate_paw_density();

    if (transform_to_rg__) {
        this->fft_transform(1);
    }
}

template void
Density::generate<double>(K_point_set const& ks__, bool symmetrize__, bool add_core__, bool transform_to_rg__);
#if defined(SIRIUS_USE_FP32)
template void
Density::generate<float>(K_point_set const& ks__, bool symmetrize__, bool add_core__, bool transform_to_rg__);
#endif

void
Density::augment()
{
    PROFILE("sirius::Density::augment");

    /* check if we need to augment charge density and magnetization */
    if (!unit_cell_.augment()) {
        return;
    }

    auto rho_aug = generate_rho_aug();

    for (int iv = 0; iv < ctx_.num_mag_dims() + 1; iv++) {
        #pragma omp parallel for schedule(static)
        for (int igloc = 0; igloc < ctx_.gvec().count(); igloc++) {
            this->component(iv).rg().f_pw_local(igloc) += rho_aug(igloc, iv);
        }
    }
}

template <typename T>
void
Density::generate_valence(K_point_set const& ks__)
{
    PROFILE("sirius::Density::generate_valence");

    /* check weights */
    double wt{0};
    double occ_val{0};
    for (int ik = 0; ik < ks__.num_kpoints(); ik++) {
        wt += ks__.get<T>(ik)->weight();
        for (int ispn = 0; ispn < ctx_.num_spinors(); ispn++) {
            for (int j = 0; j < ctx_.num_bands(); j++) {
                occ_val += ks__.get<T>(ik)->weight() * ks__.get<T>(ik)->band_occupancy(j, ispn);
            }
        }
    }

    if (std::abs(wt - 1.0) > 1e-12) {
        std::stringstream s;
        s << "K_point weights don't sum to one" << std::endl << "  obtained sum: " << wt;
        RTE_THROW(s);
    }

    if (std::abs(occ_val - unit_cell_.num_valence_electrons() + ctx_.cfg().parameters().extra_charge()) > 1e-8 &&
        ctx_.comm().rank() == 0) {
        std::stringstream s;
        s << "wrong band occupancies" << std::endl
          << "  computed : " << occ_val << std::endl
          << "  required : " << unit_cell_.num_valence_electrons() - ctx_.cfg().parameters().extra_charge() << std::endl
          << "  difference : "
          << std::abs(occ_val - unit_cell_.num_valence_electrons() + ctx_.cfg().parameters().extra_charge());
        RTE_WARNING(s);
    }

    density_matrix_->zero();

    if (occupation_matrix_) {
        occupation_matrix_->zero();
    }

    /* zero density and magnetization */
    zero();
    for (int i = 0; i < ctx_.num_mag_dims() + 1; i++) {
        rho_mag_coarse_[i]->zero();
    }

    auto mem = ctx_.processing_unit() == device_t::CPU ? memory_t::host : memory_t::device;

    /* start the main loop over k-points */
    for (auto it : ks__.spl_num_kpoints()) {
        auto kp = ks__.get<T>(it.i);

        std::array<wf::Wave_functions_fft<T>, 2> wf_fft;

        std::vector<wf::device_memory_guard> mg;

        mg.emplace_back(kp->spinor_wave_functions().memory_guard(mem, wf::copy_to::device));
        if (ctx_.hubbard_correction()) {
            mg.emplace_back(kp->hubbard_wave_functions_S().memory_guard(mem, wf::copy_to::device));
        }

        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            int nbnd = kp->num_occupied_bands(ispn);
            /* swap wave functions for the FFT transformation */
            wf_fft[ispn] =
                    wf::Wave_functions_fft<T>(kp->gkvec_fft_sptr(), kp->spinor_wave_functions(), wf::spin_index(ispn),
                                              wf::band_range(0, nbnd), wf::shuffle_to::fft_layout);
        }

        if (ctx_.full_potential()) {
            add_k_point_contribution_dm_fplapw<T>(ctx_, *kp, *density_matrix_);
        } else {
            if (ctx_.gamma_point() && (ctx_.so_correction() == false)) {
                add_k_point_contribution_dm_pwpp<T, T>(ctx_, *kp, *density_matrix_);
            } else {
                add_k_point_contribution_dm_pwpp<T, std::complex<T>>(ctx_, *kp, *density_matrix_);
            }
            if (occupation_matrix_) {
                occupation_matrix_->add_k_point_contribution(*kp);
            }
        }

        /* add contribution from regular space grid */
        add_k_point_contribution_rg(kp, wf_fft);
    }

    for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
        ctx_.comm().allreduce(density_matrix(ia).at(memory_t::host), static_cast<int>(density_matrix(ia).size()));
    }

    if (occupation_matrix_) {
        occupation_matrix_->reduce();
    }

    auto& comm = ctx_.gvec_coarse_fft_sptr()->comm_ortho_fft();
    for (int j = 0; j < ctx_.num_mag_dims() + 1; j++) {
        auto ptr = (ctx_.spfft_coarse<double>().local_slice_size() == 0) ? nullptr : &rho_mag_coarse_[j]->value(0);
        /* reduce arrays; assume that each rank did its own fraction of the density */
        /* comm_ortho_fft is identical to a product of column communicator inside k-point with k-point communicator */
        comm.allreduce(ptr, ctx_.spfft_coarse<double>().local_slice_size());
        /* print checksum if needed */
        if (env::print_checksum()) {
            auto cs = mdarray<double, 1>({ctx_.spfft_coarse<double>().local_slice_size()}, ptr).checksum();
            mpi::Communicator(ctx_.spfft_coarse<double>().communicator()).allreduce(&cs, 1);
            print_checksum("rho_mag_coarse_rg", cs, ctx_.out());
        }
        /* transform to PW domain */
        rho_mag_coarse_[j]->fft_transform(-1);
        /* map to fine G-vector grid */
        for (int igloc = 0; igloc < ctx_.gvec_coarse().count(); igloc++) {
            component(j).rg().f_pw_local(ctx_.gvec().gvec_base_mapping(igloc)) = rho_mag_coarse_[j]->f_pw_local(igloc);
        }
    }

    if (!ctx_.full_potential()) {
        augment();

        /* remove extra chanrge */
        if (ctx_.gvec().comm().rank() == 0) {
            rho().rg().f_pw_local(0) += ctx_.cfg().parameters().extra_charge() / ctx_.unit_cell().omega();
        }

        if (env::print_hash()) {
            auto h = mdarray<std::complex<double>, 1>({ctx_.gvec().count()}, &rho().rg().f_pw_local(0)).hash();
            print_hash("rho", h, ctx_.out());
        }

        double nel = rho().rg().f_0().real() * unit_cell_.omega();
        /* check the number of electrons */
        if (std::abs(nel - unit_cell_.num_electrons()) > 1e-8 && ctx_.comm().rank() == 0) {
            std::stringstream s;
            s << "wrong unsymmetrized density" << std::endl
              << "  obtained value : " << std::scientific << nel << std::endl
              << "  target value : " << std::scientific << unit_cell_.num_electrons() << std::endl
              << "  difference : " << std::scientific << std::abs(nel - unit_cell_.num_electrons()) << std::endl;
            RTE_WARNING(s);
        }
    }

    /* for muffin-tin part */
    if (ctx_.full_potential()) {
        generate_valence_mt();
    }
}

mdarray<std::complex<double>, 2>
Density::generate_rho_aug() const
{
    PROFILE("sirius::Density::generate_rho_aug");

    /* local number of G-vectors */
    int gvec_count   = ctx_.gvec().count();
    auto spl_ngv_loc = split_in_blocks(gvec_count, ctx_.cfg().control().gvec_chunk_size());

    auto& mph = get_memory_pool(memory_t::host);
    memory_pool* mpd{nullptr};

    mdarray<std::complex<double>, 2> rho_aug({gvec_count, ctx_.num_mag_dims() + 1}, mph);

    switch (ctx_.processing_unit()) {
        case device_t::CPU: {
            rho_aug.zero(memory_t::host);
            break;
        }
        case device_t::GPU: {
            mpd = &get_memory_pool(memory_t::device);
            rho_aug.allocate(*mpd).zero(memory_t::device);
            break;
        }
    }

    /* add contribution to Q(G) from atoms of all types */
    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
        auto& atom_type = unit_cell_.atom_type(iat);

        if (!atom_type.augment() || atom_type.num_atoms() == 0) {
            continue;
        }

        /* number of beta-projector functions */
        int nbf = atom_type.mt_basis_size();
        /* number of Q_{xi,xi'} components for each G */
        int nqlm = nbf * (nbf + 1) / 2;

        /* convert to real matrix */
        auto dm = density_matrix_aux(atom_type);

        if (env::print_checksum()) {
            auto cs = dm.checksum();
            print_checksum("density_matrix_aux", cs, ctx_.out());
        }

        int ndm_pw = (ctx_.processing_unit() == device_t::CPU) ? 1 : ctx_.num_mag_dims() + 1;
        /* treat auxiliary array as real with x2 size */
        mdarray<double, 3> dm_pw({nqlm, spl_ngv_loc[0] * 2, ndm_pw}, mph);
        mdarray<double, 2> phase_factors({atom_type.num_atoms(), spl_ngv_loc[0] * 2}, mph);

        print_memory_usage(ctx_.out(), FILE_LINE);

        switch (ctx_.processing_unit()) {
            case device_t::CPU: {
                break;
            }
            case device_t::GPU: {
                phase_factors.allocate(*mpd);
                dm_pw.allocate(*mpd);
                dm.allocate(*mpd).copy_to(memory_t::device);
                break;
            }
        }

        print_memory_usage(ctx_.out(), FILE_LINE);

        auto qpw = (ctx_.processing_unit() == device_t::CPU)
                           ? mdarray<double, 2>()
                           : mdarray<double, 2>({nqlm, 2 * spl_ngv_loc[0]}, *mpd, mdarray_label("qpw"));

        int g_begin{0};
        /* loop over blocks of G-vectors */
        for (auto ng : spl_ngv_loc) {

            /* work on the block of the local G-vectors */
            switch (ctx_.processing_unit()) {
                case device_t::CPU: {
                    /* generate phase factors */
                    #pragma omp parallel for
                    for (int g = 0; g < ng; g++) {
                        int ig = ctx_.gvec().offset() + g_begin + g;
                        for (int i = 0; i < atom_type.num_atoms(); i++) {
                            int ia                      = atom_type.atom_id(i);
                            auto z                      = std::conj(ctx_.gvec_phase_factor(ig, ia));
                            phase_factors(i, 2 * g)     = z.real();
                            phase_factors(i, 2 * g + 1) = z.imag();
                        }
                    }
                    for (int iv = 0; iv < ctx_.num_mag_dims() + 1; iv++) {
                        PROFILE_START("sirius::Density::generate_rho_aug|gemm");
                        la::wrap(la::lib_t::blas)
                                .gemm('N', 'N', nqlm, 2 * ng, atom_type.num_atoms(), &la::constant<double>::one(),
                                      dm.at(memory_t::host, 0, 0, iv), dm.ld(), phase_factors.at(memory_t::host),
                                      phase_factors.ld(), &la::constant<double>::zero(),
                                      dm_pw.at(memory_t::host, 0, 0, 0), dm_pw.ld());
                        PROFILE_STOP("sirius::Density::generate_rho_aug|gemm");
                        PROFILE_START("sirius::Density::generate_rho_aug|sum");
                        #pragma omp parallel for
                        for (int g = 0; g < ng; g++) {
                            int igloc = g_begin + g;
                            std::complex<double> zsum(0, 0);
                            /* get contribution from non-diagonal terms */
                            for (int i = 0; i < nqlm; i++) {
                                std::complex<double> z1(ctx_.augmentation_op(iat).q_pw(i, 2 * igloc),
                                                        ctx_.augmentation_op(iat).q_pw(i, 2 * igloc + 1));
                                std::complex<double> z2(dm_pw(i, 2 * g, 0), dm_pw(i, 2 * g + 1, 0));

                                zsum += z1 * z2 * ctx_.augmentation_op(iat).sym_weight(i);
                            }
                            /* add contribution from atoms of a given type */
                            rho_aug(igloc, iv) += zsum;
                        }
                        PROFILE_STOP("sirius::Density::generate_rho_aug|sum");
                    }
                    break;
                }
                case device_t::GPU: {
#if defined(SIRIUS_GPU)
                    acc::copyin(qpw.at(memory_t::device),
                                ctx_.augmentation_op(iat).q_pw().at(memory_t::host, 0, 2 * g_begin), 2 * ng * nqlm);

                    for (int iv = 0; iv < ctx_.num_mag_dims() + 1; iv++) {
                        generate_dm_pw_gpu(
                                atom_type.num_atoms(), ng, nbf, ctx_.unit_cell().atom_coord(iat).at(memory_t::device),
                                ctx_.gvec_coord().at(memory_t::device, g_begin, 0),
                                ctx_.gvec_coord().at(memory_t::device, g_begin, 1),
                                ctx_.gvec_coord().at(memory_t::device, g_begin, 2), phase_factors.at(memory_t::device),
                                dm.at(memory_t::device, 0, 0, iv), dm_pw.at(memory_t::device, 0, 0, iv), 1 + iv);
                        sum_q_pw_dm_pw_gpu(ng, nbf, qpw.at(memory_t::device), qpw.ld(),
                                           dm_pw.at(memory_t::device, 0, 0, iv), dm_pw.ld(),
                                           ctx_.augmentation_op(iat).sym_weight().at(memory_t::device),
                                           rho_aug.at(memory_t::device, g_begin, iv), 1 + iv);
                    }
                    for (int iv = 0; iv < ctx_.num_mag_dims() + 1; iv++) {
                        acc::sync_stream(acc::stream_id(1 + iv));
                    }
#endif
                    break;
                }
            } // switch (pu)

            g_begin += ng;
        }
    }

    if (ctx_.processing_unit() == device_t::GPU) {
        rho_aug.copy_to(memory_t::host);
    }

    if (env::print_checksum()) {
        auto cs = rho_aug.checksum();
        ctx_.comm().allreduce(&cs, 1);
        print_checksum("rho_aug", cs, ctx_.out());
    }

    if (env::print_hash()) {
        auto h = rho_aug.hash();
        print_hash("rho_aug", h, ctx_.out());
    }

    return rho_aug;
}

template <int num_mag_dims>
void
Density::reduce_density_matrix(Atom_type const& atom_type__, mdarray<std::complex<double>, 3> const& zdens__,
                               mdarray<double, 3>& mt_density_matrix__)
{
    mt_density_matrix__.zero();

    #pragma omp parallel for default(shared)
    for (int idxrf2 = 0; idxrf2 < atom_type__.mt_radial_basis_size(); idxrf2++) {
        int l2 = atom_type__.indexr(idxrf2).am.l();
        for (int idxrf1 = 0; idxrf1 <= idxrf2; idxrf1++) {
            int offs = idxrf2 * (idxrf2 + 1) / 2 + idxrf1;
            int l1   = atom_type__.indexr(idxrf1).am.l();

            int xi2 = atom_type__.indexb().index_of(rf_index(idxrf2));
            for (int lm2 = sf::lm(l2, -l2); lm2 <= sf::lm(l2, l2); lm2++, xi2++) {
                int xi1 = atom_type__.indexb().index_of(rf_index(idxrf1));
                for (int lm1 = sf::lm(l1, -l1); lm1 <= sf::lm(l1, l1); lm1++, xi1++) {
                    for (int k = 0; k < atom_type__.gaunt_coefs().num_gaunt(lm1, lm2); k++) {
                        int lm3 = atom_type__.gaunt_coefs().gaunt(lm1, lm2, k).lm3;
                        auto gc = atom_type__.gaunt_coefs().gaunt(lm1, lm2, k).coef;
                        switch (num_mag_dims) {
                            case 3: {
                                mt_density_matrix__(lm3, offs, 2) += 2.0 * std::real(zdens__(xi1, xi2, 2) * gc);
                                mt_density_matrix__(lm3, offs, 3) -= 2.0 * std::imag(zdens__(xi1, xi2, 2) * gc);
                            }
                            case 1: {
                                mt_density_matrix__(lm3, offs, 1) += std::real(zdens__(xi1, xi2, 1) * gc);
                            }
                            case 0: {
                                mt_density_matrix__(lm3, offs, 0) += std::real(zdens__(xi1, xi2, 0) * gc);
                            }
                        }
                    }
                }
            }
        }
    }
}

void
Density::generate_valence_mt()
{
    PROFILE("sirius::Density::generate_valence_mt");

    /* compute occupation matrix */
    if (ctx_.hubbard_correction()) {
        RTE_THROW("LDA+U in LAPW is not implemented");

        // TODO: fix the way how occupation matrix is calculated

        // Timer t3("sirius::Density::generate:om");
        //
        // mdarray<std::complex<double>, 4> occupation_matrix(16, 16, 2, 2);
        //
        // for (int ialoc = 0; ialoc < unit_cell_.spl_num_atoms().local_size(); ialoc++)
        //{
        //    int ia = unit_cell_.spl_num_atoms(ialoc);
        //    Atom_type* type = unit_cell_.atom(ia)->type();
        //
        //    occupation_matrix.zero();
        //    for (int l = 0; l <= 3; l++)
        //    {
        //        int num_rf = type->indexr().num_rf(l);

        //        for (int j = 0; j < num_zdmat; j++)
        //        {
        //            for (int order2 = 0; order2 < num_rf; order2++)
        //            {
        //            for (int lm2 = Utils::lm_by_l_m(l, -l); lm2 <= Utils::lm_by_l_m(l, l); lm2++)
        //            {
        //                for (int order1 = 0; order1 < num_rf; order1++)
        //                {
        //                for (int lm1 = Utils::lm_by_l_m(l, -l); lm1 <= Utils::lm_by_l_m(l, l); lm1++)
        //                {
        //                    occupation_matrix(lm1, lm2, dmat_spins_[j].first, dmat_spins_[j].second) +=
        //                        mt_complex_density_matrix_loc(type->indexb_by_lm_order(lm1, order1),
        //                                                      type->indexb_by_lm_order(lm2, order2), j, ialoc) *
        //                        unit_cell_.atom(ia)->symmetry_class()->o_radial_integral(l, order1, order2);
        //                }
        //                }
        //            }
        //            }
        //        }
        //    }
        //
        //    // restore the du block
        //    for (int lm1 = 0; lm1 < 16; lm1++)
        //    {
        //        for (int lm2 = 0; lm2 < 16; lm2++)
        //            occupation_matrix(lm2, lm1, 1, 0) = conj(occupation_matrix(lm1, lm2, 0, 1));
        //    }

        //    unit_cell_.atom(ia)->set_occupation_matrix(&occupation_matrix(0, 0, 0, 0));
        //}

        // for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
        //{
        //    int rank = unit_cell_.spl_num_atoms().local_rank(ia);
        //    unit_cell_.atom(ia)->sync_occupation_matrix(ctx_.comm(), rank);
        //}
    }

    int max_num_rf_pairs = unit_cell_.max_mt_radial_basis_size() * (unit_cell_.max_mt_radial_basis_size() + 1) / 2;

    // real density matrix
    mdarray<double, 3> mt_density_matrix({ctx_.lmmax_rho(), max_num_rf_pairs, ctx_.num_mag_dims() + 1});

    mdarray<double, 2> rf_pairs({unit_cell_.max_num_mt_points(), max_num_rf_pairs});
    mdarray<double, 3> dlm({ctx_.lmmax_rho(), unit_cell_.max_num_mt_points(), ctx_.num_mag_dims() + 1});

    for (auto it : unit_cell_.spl_num_atoms()) {
        auto& atom_type = unit_cell_.atom(it.i).type();

        int nmtp         = atom_type.num_mt_points();
        int num_rf_pairs = atom_type.mt_radial_basis_size() * (atom_type.mt_radial_basis_size() + 1) / 2;

        PROFILE_START("sirius::Density::generate|sum_zdens");
        switch (ctx_.num_mag_dims()) {
            case 3: {
                reduce_density_matrix<3>(atom_type, density_matrix(it.i), mt_density_matrix);
                break;
            }
            case 1: {
                reduce_density_matrix<1>(atom_type, density_matrix(it.i), mt_density_matrix);
                break;
            }
            case 0: {
                reduce_density_matrix<0>(atom_type, density_matrix(it.i), mt_density_matrix);
                break;
            }
        }
        PROFILE_STOP("sirius::Density::generate|sum_zdens");

        PROFILE("sirius::Density::generate|expand_lm");
        /* collect radial functions */
        for (int idxrf2 = 0; idxrf2 < atom_type.mt_radial_basis_size(); idxrf2++) {
            int offs = idxrf2 * (idxrf2 + 1) / 2;
            for (int idxrf1 = 0; idxrf1 <= idxrf2; idxrf1++) {
                /* off-diagonal pairs are taken two times: d_{12}*f_1*f_2 + d_{21}*f_2*f_1 = d_{12}*2*f_1*f_2 */
                int n = (idxrf1 == idxrf2) ? 1 : 2;
                for (int ir = 0; ir < unit_cell_.atom(it.i).num_mt_points(); ir++) {
                    rf_pairs(ir, offs + idxrf1) = n *
                                                  unit_cell_.atom(it.i).symmetry_class().radial_function(ir, idxrf1) *
                                                  unit_cell_.atom(it.i).symmetry_class().radial_function(ir, idxrf2);
                }
            }
        }
        for (int j = 0; j < ctx_.num_mag_dims() + 1; j++) {
            la::wrap(la::lib_t::blas)
                    .gemm('N', 'T', ctx_.lmmax_rho(), nmtp, num_rf_pairs, &la::constant<double>::one(),
                          &mt_density_matrix(0, 0, j), mt_density_matrix.ld(), &rf_pairs(0, 0), rf_pairs.ld(),
                          &la::constant<double>::zero(), &dlm(0, 0, j), dlm.ld());
        }

        auto sz = ctx_.lmmax_rho() * nmtp;
        switch (ctx_.num_mag_dims()) {
            case 3: {
                /* copy x component */
                std::copy(&dlm(0, 0, 2), &dlm(0, 0, 2) + sz, &mag(1).mt()[it.i](0, 0));
                /* copy y component */
                std::copy(&dlm(0, 0, 3), &dlm(0, 0, 3) + sz, &mag(2).mt()[it.i](0, 0));
            }
            case 1: {
                for (int ir = 0; ir < nmtp; ir++) {
                    for (int lm = 0; lm < ctx_.lmmax_rho(); lm++) {
                        rho().mt()[it.i](lm, ir)  = dlm(lm, ir, 0) + dlm(lm, ir, 1);
                        mag(0).mt()[it.i](lm, ir) = dlm(lm, ir, 0) - dlm(lm, ir, 1);
                    }
                }
                break;
            }
            case 0: {
                std::copy(&dlm(0, 0, 0), &dlm(0, 0, 0) + sz, &rho().mt()[it.i](0, 0));
            }
        }
    }
}

std::vector<double>
Density::compute_atomic_mag_mom(int j__) const
{
    PROFILE("sirius::Density::compute_atomic_mag_mom");

    std::vector<double> result(unit_cell_.num_atoms(), 0);

    #pragma omp parallel for
    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {

        auto& atom_to_grid_map = ctx_.atoms_to_grid_idx_map(ia);

        for (auto coord : atom_to_grid_map) {
            int ir = coord.first;
            result[ia] += this->mag(j__).rg().value(ir);
        }

        result[ia] *= (unit_cell_.omega() / fft::spfft_grid_size(ctx_.spfft<double>()));
    }
    mpi::Communicator(ctx_.spfft<double>().communicator()).allreduce(result.data(), unit_cell_.num_atoms());
    return result;
}

std::array<periodic_function_integrate_t<double>, 3>
Density::get_magnetisation() const
{
    PROFILE("sirius::Density::get_magnetisation");

    std::array<periodic_function_integrate_t<double>, 3> result;
    for (int j = 0; j < 3; j++) {
        result[j].mt = std::vector<double>(ctx_.unit_cell().num_atoms(), 0);
    }

    std::vector<int> idx = (ctx_.num_mag_dims() == 1) ? std::vector<int>({2}) : std::vector<int>({2, 0, 1});

    for (int j = 0; j < ctx_.num_mag_dims(); j++) {
        result[idx[j]] = this->mag(j).integrate();
        if (!ctx_.full_potential()) {
            result[idx[j]].mt = this->compute_atomic_mag_mom(j);
        }
    }

    return result;
}

mdarray<double, 2>
Density::density_matrix_aux(typename atom_index_t::global ia__) const
{
    auto nbf = ctx_.unit_cell().atom(ia__).type().mt_basis_size();
    mdarray<double, 2> dm({nbf * (nbf + 1) / 2, ctx_.num_mag_dims() + 1});
    for (int xi2 = 0; xi2 < nbf; xi2++) {
        for (int xi1 = 0; xi1 <= xi2; xi1++) {
            auto idx12 = packed_index(xi1, xi2);
            switch (ctx_.num_mag_dims()) {
                case 3: {
                    dm(idx12, 2) = 2 * std::real(this->density_matrix(ia__)(xi2, xi1, 2));
                    dm(idx12, 3) = -2 * std::imag(this->density_matrix(ia__)(xi2, xi1, 2));
                }
                case 1: {
                    dm(idx12, 0) = std::real(this->density_matrix(ia__)(xi2, xi1, 0) +
                                             this->density_matrix(ia__)(xi2, xi1, 1));
                    dm(idx12, 1) = std::real(this->density_matrix(ia__)(xi2, xi1, 0) -
                                             this->density_matrix(ia__)(xi2, xi1, 1));
                    break;
                }
                case 0: {
                    dm(idx12, 0) = this->density_matrix(ia__)(xi2, xi1, 0).real();
                    break;
                }
            }
        }
    }
    return dm;
}

mdarray<double, 3>
Density::density_matrix_aux(Atom_type const& atom_type__) const
{
    auto nbf = atom_type__.mt_basis_size();

    /* convert to real matrix */
    mdarray<double, 3> dm({nbf * (nbf + 1) / 2, atom_type__.num_atoms(), ctx_.num_mag_dims() + 1});
    #pragma omp parallel for
    for (int i = 0; i < atom_type__.num_atoms(); i++) {
        int ia   = atom_type__.atom_id(i);
        auto dm1 = this->density_matrix_aux(typename atom_index_t::global(ia));
        for (int j = 0; j < ctx_.num_mag_dims() + 1; j++) {
            for (int k = 0; k < nbf * (nbf + 1) / 2; k++) {
                dm(k, i, j) = dm1(k, j);
            }
        }
    }
    return dm;
}

void
Density::mixer_init(config_t::mixer_t const& mixer_cfg__)
{
    auto func_prop    = mixer::periodic_function_property();
    auto func_prop1   = mixer::periodic_function_property_modified(true);
    auto density_prop = mixer::density_function_property();
    auto paw_prop     = mixer::paw_density_function_property();
    auto hubbard_prop = mixer::hubbard_matrix_function_property();

    /* create mixer */
    this->mixer_ =
            mixer::Mixer_factory<Periodic_function<double>, Periodic_function<double>, Periodic_function<double>,
                                 Periodic_function<double>, density_matrix_t, PAW_density<double>, Hubbard_matrix>(
                    mixer_cfg__);

    if (ctx_.full_potential()) {
        this->mixer_->initialize_function<0>(func_prop, component(0), ctx_,
                                             [&](int ia) { return lmax_t(ctx_.lmax_rho()); });
        if (ctx_.num_mag_dims() > 0) {
            this->mixer_->initialize_function<1>(func_prop, component(1), ctx_,
                                                 [&](int ia) { return lmax_t(ctx_.lmax_rho()); });
        }
        if (ctx_.num_mag_dims() > 1) {
            this->mixer_->initialize_function<2>(func_prop, component(2), ctx_,
                                                 [&](int ia) { return lmax_t(ctx_.lmax_rho()); });
            this->mixer_->initialize_function<3>(func_prop, component(3), ctx_,
                                                 [&](int ia) { return lmax_t(ctx_.lmax_rho()); });
        }
    } else {
        /* initialize functions */
        if (mixer_cfg__.use_hartree()) {
            this->mixer_->initialize_function<0>(func_prop1, component(0), ctx_);
        } else {
            this->mixer_->initialize_function<0>(func_prop, component(0), ctx_);
        }
        if (ctx_.num_mag_dims() > 0) {
            this->mixer_->initialize_function<1>(func_prop, component(1), ctx_);
        }
        if (ctx_.num_mag_dims() > 1) {
            this->mixer_->initialize_function<2>(func_prop, component(2), ctx_);
            this->mixer_->initialize_function<3>(func_prop, component(3), ctx_);
        }
    }

    this->mixer_->initialize_function<4>(density_prop, *density_matrix_, unit_cell_, ctx_.num_mag_comp());

    if (ctx_.unit_cell().num_paw_atoms()) {
        this->mixer_->initialize_function<5>(paw_prop, *paw_density_, unit_cell_);
    }
    if (occupation_matrix_) {
        this->mixer_->initialize_function<6>(hubbard_prop, *occupation_matrix_, ctx_);
    }
}

void
Density::mixer_input()
{
    PROFILE("sirius::Density::mixer_input");

    mixer_->set_input<0>(component(0));
    if (ctx_.num_mag_dims() > 0) {
        mixer_->set_input<1>(component(1));
    }
    if (ctx_.num_mag_dims() > 1) {
        mixer_->set_input<2>(component(2));
        mixer_->set_input<3>(component(3));
    }

    mixer_->set_input<4>(*density_matrix_);

    if (ctx_.unit_cell().num_paw_atoms()) {
        mixer_->set_input<5>(*paw_density_);
    }

    if (occupation_matrix_) {
        mixer_->set_input<6>(*occupation_matrix_);
    }
}

void
Density::mixer_output()
{
    PROFILE("sirius::Density::mixer_output");

    mixer_->get_output<0>(component(0));
    if (ctx_.num_mag_dims() > 0) {
        mixer_->get_output<1>(component(1));
    }
    if (ctx_.num_mag_dims() > 1) {
        mixer_->get_output<2>(component(2));
        mixer_->get_output<3>(component(3));
    }

    mixer_->get_output<4>(*density_matrix_);

    if (ctx_.unit_cell().num_paw_atoms()) {
        mixer_->get_output<5>(*paw_density_);
    }

    if (occupation_matrix_) {
        mixer_->get_output<6>(*occupation_matrix_);
    }

    /* transform mixed density to plane-wave domain */
    this->fft_transform(-1);
}

double
Density::mix()
{
    PROFILE("sirius::Density::mix");

    mixer_input();
    double rms = mixer_->mix(ctx_.cfg().mixer().rms_min());
    mixer_output();

    return rms;
}

void
Density::print_info(std::ostream& out__) const
{
    auto result = this->rho().integrate();

    auto total_charge = result.total;
    auto it_charge    = result.rg;
    auto mt_charge    = result.mt;

    auto result_mag = this->get_magnetisation();

    auto write_vector = [&](r3::vector<double> v__) {
        out__ << "[" << std::setw(9) << std::setprecision(5) << std::fixed << v__[0] << ", " << std::setw(9)
              << std::setprecision(5) << std::fixed << v__[1] << ", " << std::setw(9) << std::setprecision(5)
              << std::fixed << v__[2] << "]";
    };

    out__ << "Charges and magnetic moments" << std::endl << hbar(80, '-') << std::endl;
    if (ctx_.full_potential()) {
        double total_core_leakage{0.0};
        out__ << "atom      charge    core leakage";
        if (ctx_.num_mag_dims()) {
            out__ << "                 moment                |moment|";
        }
        out__ << std::endl << hbar(80, '-') << std::endl;

        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
            double core_leakage = unit_cell_.atom(ia).symmetry_class().core_leakage();
            total_core_leakage += core_leakage;
            out__ << std::setw(4) << ia << std::setw(12) << std::setprecision(6) << std::fixed << mt_charge[ia]
                  << std::setw(16) << std::setprecision(6) << std::scientific << core_leakage;
            if (ctx_.num_mag_dims()) {
                r3::vector<double> v({result_mag[0].mt[ia], result_mag[1].mt[ia], result_mag[2].mt[ia]});
                out__ << "  ";
                write_vector(v);
                out__ << std::setw(12) << std::setprecision(6) << std::fixed << v.length();
            }
            out__ << std::endl;
        }
        out__ << std::endl;
        out__ << "total core leakage    : " << std::setprecision(8) << std::scientific << total_core_leakage
              << std::endl
              << "interstitial charge   : " << std::setprecision(6) << std::fixed << it_charge << std::endl;
        if (ctx_.num_mag_dims()) {
            r3::vector<double> v({result_mag[0].rg, result_mag[1].rg, result_mag[2].rg});
            out__ << "interstitial moment   : ";
            write_vector(v);
            out__ << ", magnitude : " << std::setprecision(6) << std::fixed << v.length() << std::endl;
        }
    } else {
        if (ctx_.num_mag_dims()) {
            out__ << "atom                moment                |moment|" << std::endl << hbar(80, '-') << std::endl;

            for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
                r3::vector<double> v({result_mag[0].mt[ia], result_mag[1].mt[ia], result_mag[2].mt[ia]});
                out__ << std::setw(4) << ia << " ";
                write_vector(v);
                out__ << std::setw(12) << std::setprecision(6) << std::fixed << v.length() << std::endl;
            }
            out__ << std::endl;
        }
    }
    out__ << "total charge          : " << std::setprecision(6) << std::fixed << total_charge << std::endl;

    if (ctx_.num_mag_dims()) {
        r3::vector<double> v({result_mag[0].total, result_mag[1].total, result_mag[2].total});
        out__ << "total moment          : ";
        write_vector(v);
        out__ << ", magnitude : " << std::setprecision(6) << std::fixed << v.length() << std::endl;
    }

    /*
     * DEBUG: compute magnetic moments analytically
     */
    // auto Rmt = ctx_.unit_cell().find_mt_radii(1, true);

    // for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
    //     double mom{0};
    //     for (int ig = 0; ig < ctx_.gvec().num_gvec(); ig++) {
    //         auto ff = sirius::unit_step_function_form_factors(Rmt[ctx_.unit_cell().atom(ia).type_id()],
    //         ctx_.gvec().gvec_len(ig)); mom += (ctx_.gvec_phase_factor(ctx_.gvec().gvec(ig), ia) * ff *
    //         this->magnetization(0).f_pw_local(ig)).real();
    //     }
    //     mom *= fourpi;
    //     if (ctx_.gvec().reduced()) {
    //         mom *= 2;
    //     }
    //     out__ << "ia="<<ia<<" mom="<<mom<<std::endl;
    // }
}

} // namespace sirius
