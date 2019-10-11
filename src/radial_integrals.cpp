#include "radial_integrals.hpp"
#include <omp.h>

namespace sirius {

template <bool jl_deriv>
void Radial_integrals_atomic_wf<jl_deriv>::generate()
{
    PROFILE("sirius::Radial_integrals|atomic_wfs");

    /* spherical Bessel functions jl(qx) */
    sddk::mdarray<Spherical_Bessel_functions, 1> jl(nq());

    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {

        auto& atom_type = unit_cell_.atom_type(iat);

        int nwf = (hubbard_) ? atom_type.indexr_hub().size() : atom_type.indexr_wfs().size();
        if (!nwf) {
            continue;
        }

        /* create jl(qx) */
        #pragma omp parallel for
        for (int iq = 0; iq < nq(); iq++) {
            jl(iq) = Spherical_Bessel_functions(atom_type.lmax_ps_atomic_wf(), atom_type.radial_grid(), grid_q_[iq]);
        }

        /* loop over all pseudo wave-functions */
        for (int i = 0; i < nwf; i++) {
            values_(i, iat) = Spline<double>(grid_q_);

            int l = (hubbard_) ? atom_type.indexr_hub(i).l : atom_type.indexr_wfs(i).l;
            auto& rwf = (hubbard_) ? atom_type.hubbard_radial_function(i) : std::get<3>(atom_type.ps_atomic_wf(i));

            #pragma omp parallel for
            for (int iq = 0; iq < nq(); iq++) {
                if (jl_deriv) {
                    auto s              = jl(iq).deriv_q(l);
                    values_(i, iat)(iq) = sirius::inner(s, rwf, 1);
                } else {
                    values_(i, iat)(iq) = sirius::inner(jl(iq)[l], rwf, 1);
                }
            }

            values_(i, iat).interpolate();
        }
    }
}

template<bool jl_deriv>
void Radial_integrals_aug<jl_deriv>::generate()
{
    PROFILE("sirius::Radial_integrals|aug");

    /* interpolate <j_{l_n}(q*x) | Q_{xi,xi'}^{l}(x) > with splines */
    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
        auto& atom_type = unit_cell_.atom_type(iat);

        if (!atom_type.augment()) {
            continue;
        }

        /* number of radial beta-functions */
        int nbrf = atom_type.mt_radial_basis_size();
        /* maximum l of beta-projectors */
        int lmax_beta = atom_type.indexr().lmax();

        for (int l = 0; l <= 2 * lmax_beta; l++) {
            for (int idx = 0; idx < nbrf * (nbrf + 1) / 2; idx++) {
                values_(idx, l, iat) = Spline<double>(grid_q_);
            }
        }

        #pragma omp parallel for
        for (int iq_loc = 0; iq_loc < spl_q_.local_size(); iq_loc++) {
            int iq = spl_q_[iq_loc];

            Spherical_Bessel_functions jl(2 * lmax_beta, atom_type.radial_grid(), grid_q_[iq]);

            for (int l3 = 0; l3 <= 2 * lmax_beta; l3++) {
                for (int idxrf2 = 0; idxrf2 < nbrf; idxrf2++) {
                    int l2 = atom_type.indexr(idxrf2).l;
                    for (int idxrf1 = 0; idxrf1 <= idxrf2; idxrf1++) {
                        int l1 = atom_type.indexr(idxrf1).l;

                        int idx = idxrf2 * (idxrf2 + 1) / 2 + idxrf1;

                        if (l3 >= std::abs(l1 - l2) && l3 <= (l1 + l2) && (l1 + l2 + l3) % 2 == 0) {
                            if (jl_deriv) {
                                auto s = jl.deriv_q(l3);
                                values_(idx, l3, iat)(iq) =
                                    sirius::inner(s, atom_type.q_radial_function(idxrf1, idxrf2, l3), 0);
                            } else {
                                values_(idx, l3, iat)(iq) =
                                    sirius::inner(jl[l3], atom_type.q_radial_function(idxrf1, idxrf2, l3), 0);
                            }
                        }
                    }
                }
            }
        }
        for (int l = 0; l <= 2 * lmax_beta; l++) {
            for (int idx = 0; idx < nbrf * (nbrf + 1) / 2; idx++) {
                unit_cell_.comm().allgather(&values_(idx, l, iat)(0), spl_q_.global_offset(), spl_q_.local_size());
            }
        }

        #pragma omp parallel for
        for (int l = 0; l <= 2 * lmax_beta; l++) {
            for (int idx = 0; idx < nbrf * (nbrf + 1) / 2; idx++) {
                values_(idx, l, iat).interpolate();
            }
        }
    }
}

void Radial_integrals_rho_pseudo::generate()
{
    PROFILE("sirius::Radial_integrals|rho_pseudo");

    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
        auto& atom_type = unit_cell_.atom_type(iat);

        if (atom_type.ps_total_charge_density().empty()) {
            continue;
        }

        values_(iat) = Spline<double>(grid_q_);

        Spline<double> rho(atom_type.radial_grid(), atom_type.ps_total_charge_density());

        #pragma omp parallel for
        for (int iq_loc = 0; iq_loc < spl_q_.local_size(); iq_loc++) {
            int iq = spl_q_[iq_loc];
            Spherical_Bessel_functions jl(0, atom_type.radial_grid(), grid_q_[iq]);

            values_(iat)(iq) = sirius::inner(jl[0], rho, 0, atom_type.num_mt_points()) / fourpi;
        }
        unit_cell_.comm().allgather(&values_(iat)(0), spl_q_.global_offset(), spl_q_.local_size());
        values_(iat).interpolate();
    }
}

template<bool jl_deriv>
void Radial_integrals_rho_core_pseudo<jl_deriv>::generate()
{
    PROFILE("sirius::Radial_integrals|rho_core_pseudo");

    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
        auto& atom_type = unit_cell_.atom_type(iat);

        if (atom_type.ps_core_charge_density().empty()) {
            continue;
        }

        values_(iat) = Spline<double>(grid_q_);

        Spline<double> ps_core(atom_type.radial_grid(), atom_type.ps_core_charge_density());

        #pragma omp parallel for
        for (int iq_loc = 0; iq_loc < spl_q_.local_size(); iq_loc++) {
            int iq = spl_q_[iq_loc];
            Spherical_Bessel_functions jl(0, atom_type.radial_grid(), grid_q_[iq]);

            if (jl_deriv) {
                auto s           = jl.deriv_q(0);
                values_(iat)(iq) = sirius::inner(s, ps_core, 2, atom_type.num_mt_points());
            } else {
                values_(iat)(iq) = sirius::inner(jl[0], ps_core, 2, atom_type.num_mt_points());
            }
        }
        unit_cell_.comm().allgather(&values_(iat)(0), spl_q_.global_offset(), spl_q_.local_size());
        values_(iat).interpolate();
    }
}

template<bool jl_deriv>
void Radial_integrals_beta<jl_deriv>::generate()
{
    PROFILE("sirius::Radial_integrals|beta");

    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
        auto& atom_type = unit_cell_.atom_type(iat);
        int nrb = atom_type.num_beta_radial_functions();

        if (!nrb) {
            continue;
        }

        for (int idxrf = 0; idxrf < nrb; idxrf++) {
            values_(idxrf, iat) = Spline<double>(grid_q_);
        }

        #pragma omp parallel for
        for (int iq_loc = 0; iq_loc < spl_q_.local_size(); iq_loc++) {
            int iq = spl_q_[iq_loc];
            Spherical_Bessel_functions jl(unit_cell_.lmax(), atom_type.radial_grid(), grid_q_[iq]);
            for (int idxrf = 0; idxrf < nrb; idxrf++) {
                int l  = atom_type.indexr(idxrf).l;
                /* compute \int j_l(q * r) beta_l(r) r^2 dr or \int d (j_l(q*r) / dq) beta_l(r) r^2  */
                /* remeber that beta(r) are defined as miltiplied by r */
                if (jl_deriv) {
                    auto s  = jl.deriv_q(l);
                    values_(idxrf, iat)(iq) = sirius::inner(s, atom_type.beta_radial_function(idxrf), 1);
                } else {
                    values_(idxrf, iat)(iq) = sirius::inner(jl[l], atom_type.beta_radial_function(idxrf), 1);
                }
            }
        }

        for (int idxrf = 0; idxrf < nrb; idxrf++) {
            unit_cell_.comm().allgather(&values_(idxrf, iat)(0), spl_q_.global_offset(), spl_q_.local_size());
            values_(idxrf, iat).interpolate();
        }
    }
}

void Radial_integrals_beta_jl::generate()
{
    PROFILE("sirius::Radial_integrals|beta_jl");

    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
        auto& atom_type = unit_cell_.atom_type(iat);
        int nrb         = atom_type.num_beta_radial_functions();

        if (!nrb) {
            continue;
        }

        for (int idxrf = 0; idxrf < nrb; idxrf++) {
            for (int l = 0; l <= lmax_; l++) {
                values_(idxrf, l, iat) = Spline<double>(grid_q_);
            }
        }

        #pragma omp parallel for
        for (int iq = 0; iq < grid_q_.num_points(); iq++) {
            Spherical_Bessel_functions jl(lmax_, atom_type.radial_grid(), grid_q_[iq]);
            for (int idxrf = 0; idxrf < nrb; idxrf++) {
                // int nr = atom_type.pp_desc().num_beta_radial_points[idxrf];
                for (int l = 0; l <= lmax_; l++) {
                    /* compute \int j_{l'}(q * r) beta_l(r) r^2 * r * dr */
                    /* remeber that beta(r) are defined as miltiplied by r */
                    values_(idxrf, l, iat)(iq) = sirius::inner(jl[l], atom_type.beta_radial_function(idxrf), 2);
                }
            }
        }

        #pragma omp parallel for
        for (int idxrf = 0; idxrf < nrb; idxrf++) {
            for (int l = 0; l <= lmax_; l++) {
                values_(idxrf, l, iat).interpolate();
            }
        }
    }
}

template <bool jl_deriv>
void Radial_integrals_vloc<jl_deriv>::generate()
{
    PROFILE("sirius::Radial_integrals|vloc");

    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
        auto& atom_type = unit_cell_.atom_type(iat);

        if (atom_type.local_potential().empty()) {
            continue;
        }

        values_(iat) = Spline<double>(grid_q_);

        auto& vloc = atom_type.local_potential();

        int np = atom_type.num_mt_points();
        // if (std::abs(vloc.back() * atom_type.radial_grid().last() + atom_type.zn()) > 1e-10) {
        //    std::stringstream s;
        //    s << "Wrong asymptotics of local potential for atom type " << iat << std::endl
        //      << "hack with 10 a.u. cutoff is activated";
        //    WARNING(s);
        if (true) {
            int np1 = atom_type.radial_grid().index_of(10);
            if (np1 != -1) {
                np = np1;
            }
        }

        auto rg = atom_type.radial_grid().segment(np);

        #pragma omp parallel for
        for (int iq_loc = 0; iq_loc < spl_q_.local_size(); iq_loc++) {
            int iq = spl_q_[iq_loc];
            Spline<double> s(rg);
            double g = grid_q_[iq];

            if (jl_deriv) { /* integral with derivative of j0(q*r) over q */
                for (int ir = 0; ir < rg.num_points(); ir++) {
                    double x = rg[ir];
                    s(ir) = (x * vloc[ir] + atom_type.zn() * std::erf(x)) * (std::sin(g * x) - g * x * std::cos(g * x));
                }
            } else {           /* integral with j0(q*r) */
                if (iq == 0) { /* q=0 case */
                    if (unit_cell_.parameters().parameters_input().enable_esm_ &&
                        unit_cell_.parameters().parameters_input().esm_bc_ != "pbc") {
                        for (int ir = 0; ir < rg.num_points(); ir++) {
                            double x = rg[ir];
                            s(ir)    = (x * vloc[ir] + atom_type.zn() * std::erf(x)) * x;
                        }
                    } else {
                        for (int ir = 0; ir < rg.num_points(); ir++) {
                            double x = rg[ir];
                            s(ir)    = (x * vloc[ir] + atom_type.zn()) * x;
                        }
                    }
                } else {
                    for (int ir = 0; ir < rg.num_points(); ir++) {
                        double x = rg[ir];
                        s(ir)    = (x * vloc[ir] + atom_type.zn() * std::erf(x)) * std::sin(g * x);
                    }
                }
            }
            values_(iat)(iq) = s.interpolate().integrate(0);
        }
        unit_cell_.comm().allgather(&values_(iat)(0), spl_q_.global_offset(), spl_q_.local_size());
        values_(iat).interpolate();
    }
}

void Radial_integrals_rho_free_atom::generate()
{
    PROFILE("sirius::Radial_integrals|rho_free_atom");

    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
        auto& atom_type = unit_cell_.atom_type(iat);
        values_(iat)    = Spline<double>(grid_q_);

        #pragma omp parallel for
        for (int iq = 0; iq < grid_q_.num_points(); iq++) {
            double g = grid_q_[iq];
            Spline<double> s(unit_cell_.atom_type(iat).free_atom_radial_grid());
            if (iq == 0) {
                for (int ir = 0; ir < s.num_points(); ir++) {
                    s(ir) = atom_type.free_atom_density(ir);
                }
                values_(iat)(iq) = s.interpolate().integrate(2);
            } else {
                for (int ir = 0; ir < s.num_points(); ir++) {
                    s(ir) = atom_type.free_atom_density(ir) * std::sin(g * atom_type.free_atom_radial_grid(ir));
                }
                values_(iat)(iq) = s.interpolate().integrate(1);
            }
        }
        values_(iat).interpolate();
    }
}

template class Radial_integrals_atomic_wf<true>;
template class Radial_integrals_atomic_wf<false>;

template class Radial_integrals_aug<true>;
template class Radial_integrals_aug<false>;

template class Radial_integrals_rho_core_pseudo<true>;
template class Radial_integrals_rho_core_pseudo<false>;

template class Radial_integrals_beta<true>;
template class Radial_integrals_beta<false>;

template class Radial_integrals_vloc<true>;
template class Radial_integrals_vloc<false>;


} // namespace sirius
