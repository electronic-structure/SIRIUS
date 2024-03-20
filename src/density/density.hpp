/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file density.hpp
 *
 *  \brief Contains definition and partial implementation of sirius::Density class.
 */

#ifndef __DENSITY_HPP__
#define __DENSITY_HPP__

#include <iomanip>
#include "function3d/field4d.hpp"
#include "function3d/paw_field4d.hpp"
#include "function3d/periodic_function.hpp"
#include "function3d/spheric_function_set.hpp"
#include "function3d/make_periodic_function.hpp"
#include "k_point/k_point_set.hpp"
#include "mixer/mixer.hpp"
#include "occupation_matrix.hpp"
#include "density_matrix.hpp"
#include "core/math_tools.hpp"

#if defined(SIRIUS_GPU)
extern "C" {

void
update_density_rg_1_real_gpu_float(int size__, float const* psi_rg__, float wt__, float* density_rg__);

void
update_density_rg_1_real_gpu_double(int size__, double const* psi_rg__, double wt__, double* density_rg__);

void
update_density_rg_1_complex_gpu_float(int size__, std::complex<float> const* psi_rg__, float wt__, float* density_rg__);

void
update_density_rg_1_complex_gpu_double(int size__, std::complex<double> const* psi_rg__, double wt__,
                                       double* density_rg__);

void
update_density_rg_2_gpu_float(int size__, std::complex<float> const* psi_rg_up__,
                              std::complex<float> const* psi_rg_dn__, float wt__, float* density_x_rg__,
                              float* density_y_rg__);

void
update_density_rg_2_gpu_double(int size__, std::complex<double> const* psi_rg_up__,
                               std::complex<double> const* psi_rg_dn__, double wt__, double* density_x_rg__,
                               double* density_y_rg__);

void
generate_dm_pw_gpu(int num_atoms__, int num_gvec_loc__, int num_beta__, double const* atom_pos__, int const* gvx__,
                   int const* gvy__, int const* gvz__, double* phase_factors__, double const* dm__, double* dm_pw__,
                   int stream_id__);

void
sum_q_pw_dm_pw_gpu(int num_gvec_loc__, int nbf__, double const* q_pw__, int ldq__, double const* dm_pw__, int ldd__,
                   double const* sym_weight__, std::complex<double>* rho_pw__, int stream_id__);
}
#endif

namespace sirius {

/// Use Kuebler's trick to get rho_up and rho_dn from density and magnetisation.
inline auto
get_rho_up_dn(int num_mag_dims__, double rho__, r3::vector<double> mag__)
{
    if (rho__ < 0.0) {
        return std::pair<double, double>(0.0, 0.0);
    }

    double mag{0};
    if (num_mag_dims__ == 1) { /* collinear case */
        mag = mag__[0];
        /* fix numerical noise at high values of magnetization */
        if (std::abs(mag) > rho__) {
            mag = sign(mag) * rho__;
        }
    } else { /* non-collinear case */
        /* fix numerical noise at high values of magnetization */
        mag = std::min(mag__.length(), rho__);
    }

    return std::make_pair<double, double>(0.5 * (rho__ + mag), 0.5 * (rho__ - mag));
}

/// PAW density storage.
template <typename T>
class PAW_density : public PAW_field4D<T>
{
  public:
    PAW_density(Unit_cell const& uc__)
        : PAW_field4D<T>("PAW density", uc__, false)
    {
    }

    auto&
    ae_density(int i__, int ja__)
    {
        return this->ae_component(i__)[ja__];
    }

    auto const&
    ae_density(int i__, int ja__) const
    {
        return this->ae_component(i__)[ja__];
    }

    auto&
    ps_density(int i__, int ja__)
    {
        return this->ps_component(i__)[ja__];
    }

    auto const&
    ps_density(int i__, int ja__) const
    {
        return this->ps_component(i__)[ja__];
    }
};

/// Generate charge density and magnetization from occupied spinor wave-functions.
/** Let's start from the definition of the complex density matrix:
    \f[
    \rho_{\sigma' \sigma}({\bf r}) =
     \sum_{j{\bf k}} n_{j{\bf k}} \Psi_{j{\bf k}}^{\sigma*}({\bf r}) \Psi_{j{\bf k}}^{\sigma'}({\bf r}) =
     \frac{1}{2} \left( \begin{array}{cc} \rho({\bf r})+m_z({\bf r}) &
            m_x({\bf r})-im_y({\bf r}) \\ m_x({\bf r})+im_y({\bf r}) & \rho({\bf r})-m_z({\bf r}) \end{array} \right)
    \f]
    We notice that the diagonal components of the density matrix are actually real and the off-diagonal components are
    expressed trough two independent functions \f$ m_x({\bf r}) \f$ and \f$ m_y({\bf r}) \f$. Having this in mind we
    will work with a slightly different object, namely a real density matrix, defined as a 1-, 2- or 4-dimensional
    (depending on the number of magnetic components) vector with the following elements:
        - \f$ [ \rho({\bf r}) ] \f$ in case of non-magnetic configuration
        - \f$ [ \rho_{\uparrow \uparrow}({\bf r}), \rho_{\downarrow \downarrow}({\bf r}) ]  =
              [ \frac{\rho({\bf r})+m_z({\bf r})}{2}, \frac{\rho({\bf r})-m_z({\bf r})}{2} ] \f$ in case of collinear
           magnetic configuration
        - \f$ [ \rho_{\uparrow \uparrow}({\bf r}), \rho_{\downarrow \downarrow}({\bf r}),
                2 \Re \rho_{\uparrow \downarrow}({\bf r}), -2 \Im \rho_{\uparrow \downarrow}({\bf r}) ] =
              [ \frac{\rho({\bf r})+m_z({\bf r})}{2}, \frac{\rho({\bf r})-m_z({\bf r})}{2},
                m_x({\bf r}),  m_y({\bf r}) ] \f$ in the general case of non-collinear magnetic configuration

    At this point it is straightforward to compute the density and magnetization in the interstitial
    (see Density::add_k_point_contribution_rg()). The muffin-tin part of the density and magnetization is obtained
    in a slighlty more complicated way. Recall the expansion of spinor wave-functions inside the muffin-tin
    \f$ \alpha \f$:
    \f[
    \Psi_{j{\bf k}}^{\sigma}({\bf r}) = \sum_{\xi}^{N_{\xi}^{\alpha}} {S_{\xi}^{\sigma j {\bf k},\alpha}}
    f_{\ell_{\xi} \lambda_{\xi}}^{\alpha}(r)Y_{\ell_{\xi}m_{\xi}}(\hat {\bf r})
    \f]
    which we insert into expression for the complex density matrix:
    \f[
    \rho_{\sigma' \sigma}({\bf r}) = \sum_{j{\bf k}} n_{j{\bf k}} \sum_{\xi}^{N_{\xi}^{\alpha}}
        S_{\xi}^{\sigma j {\bf k},\alpha*} f_{\ell_{\xi} \lambda_{\xi}}^{\alpha}(r)
        Y_{\ell_{\xi}m_{\xi}}^{*}(\hat {\bf r}) \sum_{\xi'}^{N_{\xi'}^{\alpha}} S_{\xi'}^{\sigma' j{\bf k},\alpha}
        f_{\ell_{\xi'} \lambda_{\xi'}}^{\alpha}(r)Y_{\ell_{\xi'}m_{\xi'}}(\hat {\bf r})
    \f]
    First, we eliminate a sum over bands and k-points by forming an auxiliary density tensor:
    \f[
    D_{\xi \sigma, \xi' \sigma'}^{\alpha} = \sum_{j{\bf k}} n_{j{\bf k}} S_{\xi}^{\sigma j {\bf k},\alpha*}
      S_{\xi'}^{\sigma' j {\bf k},\alpha}
    \f]
    The expression for complex density matrix simplifies to:
    \f[
    \rho_{\sigma' \sigma}({\bf r}) =  \sum_{\xi \xi'} D_{\xi \sigma, \xi' \sigma'}^{\alpha}
        f_{\ell_{\xi} \lambda_{\xi}}^{\alpha}(r)Y_{\ell_{\xi}m_{\xi}}^{*}(\hat {\bf r})
        f_{\ell_{\xi'} \lambda_{\xi'}}^{\alpha}(r)Y_{\ell_{\xi'}m_{\xi'}}(\hat {\bf r})
    \f]
    Now we can switch to the real density matrix and write its' expansion in real spherical harmonics. Let's take
    non-magnetic case as an example:
    \f[
    \rho({\bf r}) = \sum_{\xi \xi'} D_{\xi \xi'}^{\alpha}
        f_{\ell_{\xi} \lambda_{\xi}}^{\alpha}(r)Y_{\ell_{\xi}m_{\xi}}^{*}(\hat {\bf r})
        f_{\ell_{\xi'} \lambda_{\xi'}}^{\alpha}(r)Y_{\ell_{\xi'}m_{\xi'}}(\hat {\bf r}) =
        \sum_{\ell_3 m_3} \rho_{\ell_3 m_3}^{\alpha}(r) R_{\ell_3 m_3}(\hat {\bf r})
    \f]
    where
    \f[
    \rho_{\ell_3 m_3}^{\alpha}(r) = \sum_{\xi \xi'} D_{\xi \xi'}^{\alpha} f_{\ell_{\xi} \lambda_{\xi}}^{\alpha}(r)
        f_{\ell_{\xi'} \lambda_{\xi'}}^{\alpha}(r) \langle Y_{\ell_{\xi}m_{\xi}} | R_{\ell_3 m_3} |
          Y_{\ell_{\xi'}m_{\xi'}} \rangle
    \f]
    We are almost done. Now it is time to switch to the full index notation
    \f$ \xi \rightarrow \{ \ell \lambda m \} \f$ and sum over \a m and \a m' indices:
    \f[
    \rho_{\ell_3 m_3}^{\alpha}(r) = \sum_{\ell \lambda, \ell' \lambda'} f_{\ell \lambda}^{\alpha}(r)
       f_{\ell' \lambda'}^{\alpha}(r) d_{\ell \lambda, \ell' \lambda', \ell_3 m_3}^{\alpha}
    \f]
    where
    \f[
    d_{\ell \lambda, \ell' \lambda', \ell_3 m_3}^{\alpha} =
        \sum_{mm'} D_{\ell \lambda m, \ell' \lambda' m'}^{\alpha}
        \langle Y_{\ell m} | R_{\ell_3 m_3} | Y_{\ell' m'} \rangle
    \f]
    This is our final answer: radial components of density and magnetization are expressed as a linear combination of
    quadratic forms in radial functions.

    \note density and potential are allocated as global function because it's easier to load and save them.
    \note In case of full-potential calculation valence + core electron charge density is computed.
    \note In tcase of pseudopotential valence charge density is computed.
 */
class Density : public Field4D
{
  private:
    /// Alias to ctx_.unit_cell()
    Unit_cell& unit_cell_;

    /// Density matrix for all atoms.
    /** This is a global matrix, meaning that each MPI rank holds the full copy. This simplifies the symmetrization. */
    std::unique_ptr<density_matrix_t> density_matrix_;

    /// Local fraction of atoms with PAW correction.
    std::unique_ptr<PAW_density<double>> paw_density_;

    /// Occupation matrix of the LDA+U method.
    std::unique_ptr<Occupation_matrix> occupation_matrix_;

    /// Density and magnetization on the coarse FFT mesh.
    /** Coarse FFT grid is enough to generate density and magnetization from the wave-functions. The components
        of the <tt>rho_mag_coarse</tt> vector have the following order:
        \f$ \{\rho({\bf r}), m_z({\bf r}), m_x({\bf r}), m_y({\bf r}) \} \f$.
     */
    std::array<std::unique_ptr<Smooth_periodic_function<double>>, 4> rho_mag_coarse_;

    /// Pointer to pseudo core charge density
    /** In the case of pseudopotential we need to know the non-linear core correction to the
        exchange-correlation energy which is introduced trough the pseudo core density:
        \f$ E_{xc}[\rho_{val} + \rho_{core}] \f$. The 'pseudo' reflects the fact that
        this density integrated does not reproduce the total number of core elctrons.
     */
    std::unique_ptr<Smooth_periodic_function<double>> rho_pseudo_core_{nullptr};

    /// Fast mapping between composite lm index and corresponding orbital quantum number.
    std::vector<int> l_by_lm_;

    /// Density mixer.
    /** Mix the following objects: density, x-,y-,z-components of magnetisation, density matrix and
        PAW density of atoms. */
    std::unique_ptr<mixer::Mixer<Periodic_function<double>, Periodic_function<double>, Periodic_function<double>,
                                 Periodic_function<double>, density_matrix_t, PAW_density<double>, Hubbard_matrix>>
            mixer_;

    /// Generate atomic densities in the case of PAW.
    void
    generate_paw_density(paw_atom_index_t::local iapaw__);

    /// Initialize PAW density matrix.
    void
    init_density_matrix_for_paw();

    /// Reduce complex density matrix over magnetic quantum numbers
    /** The following operation is performed:
        \f[
            n_{\ell \lambda, \ell' \lambda', \ell_3 m_3}^{\alpha} =
                \sum_{mm'} D_{\ell \lambda m, \ell' \lambda' m'}^{\alpha}
                \langle Y_{\ell m} | R_{\ell_3 m_3} | Y_{\ell' m'} \rangle
        \f]
     */
    template <int num_mag_dims>
    void
    reduce_density_matrix(Atom_type const& atom_type__, mdarray<std::complex<double>, 3> const& zdens__,
                          mdarray<double, 3>& mt_density_matrix__);

    /// Add k-point contribution to the density matrix in the canonical form.
    /** In case of full-potential LAPW complex density matrix has the following expression:
        \f[
            d_{\xi \sigma, \xi' \sigma'}^{\alpha} = \sum_{j{\bf k}} n_{j{\bf k}}
                S_{\xi}^{\sigma j {\bf k},\alpha*} S_{\xi'}^{\sigma' j {\bf k},\alpha}
        \f]

        where \f$ S_{\xi}^{\sigma j {\bf k},\alpha} \f$ are the expansion coefficients of
        spinor wave functions inside muffin-tin spheres.

        In case of LDA+U the occupation matrix is also computed. It has the following expression:
        \f[
            n_{\ell,mm'}^{\sigma \sigma'} = \sum_{i {\bf k}}^{occ} \int_{0}^{R_{MT}} r^2 dr
                      \Psi_{\ell m}^{i{\bf k}\sigma *}({\bf r}) \Psi_{\ell m'}^{i{\bf k}\sigma'}({\bf r})
        \f]

        In case of ultrasoft pseudopotential the following density matrix has to be computed for each atom:
        \f[
             d_{\xi \xi'}^{\alpha} = \langle \beta_{\xi}^{\alpha} | \hat N | \beta_{\xi'}^{\alpha} \rangle =
               \sum_{j {\bf k}} \langle \beta_{\xi}^{\alpha} | \Psi_{j{\bf k}} \rangle n_{j{\bf k}}
               \langle \Psi_{j{\bf k}} | \beta_{\xi'}^{\alpha} \rangle
        \f]
        Here \f$ \hat N = \sum_{j{\bf k}} | \Psi_{j{\bf k}} \rangle n_{j{\bf k}} \langle \Psi_{j{\bf k}} | \f$ is
        the occupancy operator written in spectral representation.

        \tparam T  Precision type of wave-functions.
        \tparam F  Type of the wave-functions inner product (used in pp-pw).
     */
    template <typename T, typename F>
    void
    add_k_point_contribution_dm(K_point<T>& kp__, mdarray<std::complex<double>, 4>& density_matrix__);

    /// Add k-point contribution to the density and magnetization defined on the regular FFT grid.
    template <typename T>
    void
    add_k_point_contribution_rg(K_point<T>* kp__, std::array<wf::Wave_functions_fft<T>, 2>& wf_fft__);

    /// Generate valence density in the muffin-tins
    void
    generate_valence_mt();

    /// Generate charge density of core states
    void
    generate_core_charge_density()
    {
        PROFILE("sirius::Density::generate_core_charge_density");

        auto& spl_idx = unit_cell_.spl_num_atom_symmetry_classes();

        for (auto it : spl_idx) {
            unit_cell_.atom_symmetry_class(it.i).generate_core_charge_density(ctx_.core_relativity());
        }

        for (auto ic = begin_global(spl_idx); ic != end_global(spl_idx); ic++) {
            auto rank = spl_idx.location(ic).ib;
            unit_cell_.atom_symmetry_class(ic).sync_core_charge_density(ctx_.comm(), rank);
        }
    }

    void
    generate_pseudo_core_charge_density()
    {
        PROFILE("sirius::Density::generate_pseudo_core_charge_density");

        /* get lenghts of all G shells */
        auto q = ctx_.gvec().shells_len();
        /* get form-factors for all G shells */
        auto const ff = ctx_.ri().ps_core_->values(q, ctx_.comm());
        /* make rho_core(G) */
        auto v = make_periodic_function<true>(ctx_.unit_cell(), ctx_.gvec(), ctx_.phase_factors_t(), ff);

        std::copy(v.begin(), v.end(), &rho_pseudo_core_->f_pw_local(0));
        rho_pseudo_core_->fft_transform(1);
    }

  public:
    /// Constructor
    Density(Simulation_context& ctx__);

    /// Update internal parameters after a change of lattice vectors or atomic coordinates.
    void
    update();

    /// Find the total leakage of the core states out of the muffin-tins
    double
    core_leakage() const;

    /// Generate initial charge density and magnetization
    void
    initial_density();

    void
    initial_density_pseudo();

    void
    initial_density_full_pot();

    void
    normalize();

    /// Check total density for the correct number of electrons.
    bool
    check_num_electrons() const;

    /// Generate full charge density (valence + core) and magnetization from the wave functions.
    /** This function calls generate_valence() and then in case of full-potential LAPW method adds a core density
        to get the full charge density of the system. Density is generated in spectral representation, i.e.
        plane-wave coefficients in the interstitial and spherical harmonic components in the muffin-tins.
     */
    template <typename T>
    void
    generate(K_point_set const& ks__, bool symmetrize__, bool add_core__, bool transform_to_rg__);

    /// Generate valence charge density and magnetization from the wave functions.
    /** The interstitial density is generated on the coarse FFT grid and then transformed to the PW domain.
        After symmetrization and mixing and before the generation of the XC potential density is transformted to the
        real-space domain and checked for the number of electrons.
     */
    template <typename T>
    void
    generate_valence(K_point_set const& ks__);

    /// Add augmentation charge Q(r).
    /** Restore valence density by adding the Q-operator constribution.
        The following term is added to the valence density, generated by the pseudo wave-functions:
        \f[
            \tilde \rho({\bf G}) = \sum_{\alpha} \sum_{\xi \xi'} d_{\xi \xi'}^{\alpha} Q_{\xi' \xi}^{\alpha}({\bf G})
        \f]
        Plane-wave coefficients of the Q-operator for a given atom \f$ \alpha \f$ can be obtained from the
        corresponding coefficients of the Q-operator for a given atom \a type A:
        \f[
             Q_{\xi' \xi}^{\alpha(A)}({\bf G}) = e^{-i{\bf G}\tau_{\alpha(A)}} Q_{\xi' \xi}^{A}({\bf G})
        \f]
        We use this property to split the sum over atoms into sum over atom types and inner sum over atoms of the
        same type:
        \f[
             \tilde \rho({\bf G}) = \sum_{A} \sum_{\xi \xi'} Q_{\xi' \xi}^{A}({\bf G}) \sum_{\alpha(A)}
                d_{\xi \xi'}^{\alpha(A)} e^{-i{\bf G}\tau_{\alpha(A)}} =
                \sum_{A} \sum_{\xi \xi'} Q_{\xi' \xi}^{A}({\bf G}) d_{\xi \xi'}^{A}({\bf G})
        \f]
        where
        \f[
            d_{\xi \xi'}^{A}({\bf G}) = \sum_{\alpha(A)} d_{\xi \xi'}^{\alpha(A)} e^{-i{\bf G}\tau_{\alpha(A)}}
        \f]
     */
    void
    augment();

    /// Generate augmentation charge density.
    mdarray<std::complex<double>, 2>
    generate_rho_aug() const;

    /// Return core leakage for a specific atom symmetry class
    inline double
    core_leakage(int ic) const
    {
        return unit_cell_.atom_symmetry_class(ic).core_leakage();
    }

    /// Return const reference to charge density (scalar functions).
    inline auto const&
    rho() const
    {
        return this->scalar();
    }

    /// Return charge density (scalar functions).
    inline auto&
    rho()
    {
        return const_cast<Periodic_function<double>&>(static_cast<Density const&>(*this).rho());
    }

    inline auto&
    mag(int i)
    {
        return this->vector(i);
    }

    inline auto const&
    mag(int i) const
    {
        return this->vector(i);
    }

    inline auto&
    rho_pseudo_core()
    {
        return *rho_pseudo_core_;
    }

    inline auto const&
    rho_pseudo_core() const
    {
        return *rho_pseudo_core_;
    }

    inline auto const&
    density_mt(atom_index_t::local ialoc__) const
    {
        auto ia = ctx_.unit_cell().spl_num_atoms().global_index(ialoc__);
        return rho().mt()[ia];
    }

    /// Generate \f$ n_1 \f$  and \f$ \tilde{n}_1 \f$ in lm components.
    void
    generate_paw_density();

    /// Return list of pointers to all-electron PAW density function for a given local index of atom with PAW potential.
    inline auto
    paw_ae_density(int ia__) const
    {
        std::vector<Flm const*> result(ctx_.num_mag_dims() + 1);
        for (int j = 0; j < ctx_.num_mag_dims() + 1; j++) {
            result[j] = &paw_density_->ae_density(j, ia__);
        }
        return result;
    }

    /// Return list of pointers to pseudo PAW density function for a given local index of atom with PAW potential.
    inline auto
    paw_ps_density(int ia__) const
    {
        std::vector<Flm const*> result(ctx_.num_mag_dims() + 1);
        for (int j = 0; j < ctx_.num_mag_dims() + 1; j++) {
            result[j] = &paw_density_->ps_density(j, ia__);
        }
        return result;
    }

    void
    mixer_input();

    void
    mixer_output();

    /// Initialize density mixer.
    void
    mixer_init(config_t::mixer_t const& mixer_cfg__);

    /// Mix new density.
    double
    mix();

    inline auto const&
    density_matrix(int ia__) const
    {
        return (*density_matrix_)[ia__];
    }

    inline auto&
    density_matrix(int ia__)
    {
        return (*density_matrix_)[ia__];
    }

    /// Return density matrix for all atoms of a given type in auxiliary form.
    mdarray<double, 3>
    density_matrix_aux(Atom_type const& atom_type__) const;

    mdarray<double, 2>
    density_matrix_aux(typename atom_index_t::global ia__) const;

    /// Calculate approximate atomic magnetic moments in case of PP-PW.
    std::vector<double>
    compute_atomic_mag_mom(int j__) const;

    /// Get total magnetization and also contributions from interstitial and muffin-tin parts.
    /** In case of PP-PW there are no real muffin-tins. Instead, a value of magnetization inside atomic
     *  sphere with some chosen radius is returned.
     */
    std::array<periodic_function_integrate_t<double>, 3>
    get_magnetisation() const;

    void
    print_info(std::ostream& out__) const;

    inline auto const&
    occupation_matrix() const
    {
        return *occupation_matrix_;
    }

    inline auto&
    occupation_matrix()
    {
        return *occupation_matrix_;
    }

    inline auto const&
    paw_density() const
    {
        return *paw_density_;
    }

    /// Check density at MT boundary
    void
    check_density_continuity_at_mt()
    {
        //    // generate plane-wave coefficients of the potential in the interstitial region
        //    ctx_.fft().input(&rho_->f_it<global>(0));
        //    ctx_.fft().transform(-1);
        //    ctx_.fft().output(ctx_.num_gvec(), ctx_.fft_index(), &rho_->f_pw(0));
        //
        //    SHT sht(ctx_.lmax_rho());
        //
        //    double diff = 0.0;
        //    for (int ia = 0; ia < ctx_.num_atoms(); ia++)
        //    {
        //        for (int itp = 0; itp < sht.num_points(); itp++)
        //        {
        //            double vc[3];
        //            for (int x = 0; x < 3; x++) vc[x] = sht.coord(x, itp) * ctx_.atom(ia)->mt_radius();
        //
        //            double val_it = 0.0;
        //            for (int ig = 0; ig < ctx_.num_gvec(); ig++)
        //            {
        //                double vgc[3];
        //                ctx_.get_coordinates<cartesian, reciprocal>(ctx_.gvec(ig), vgc);
        //                val_it += real(rho_->f_pw(ig) * exp(std::complex<double>(0.0, Utils::scalar_product(vc,
        //                vgc))));
        //            }
        //
        //            double val_mt = 0.0;
        //            for (int lm = 0; lm < ctx_.lmmax_rho(); lm++)
        //                val_mt += rho_->f_rlm(lm, ctx_.atom(ia)->num_mt_points() - 1, ia) * sht.rlm_backward(lm, itp);
        //
        //            diff += fabs(val_it - val_mt);
        //        }
        //    }
        //    std::printf("Total and average charge difference at MT boundary : %.12f %.12f\n", diff, diff /
        //    ctx_.num_atoms() / sht.num_points());
    }

    void
    save(std::string name__) const
    {
        rho().hdf5_write(name__, "density");
        for (int j = 0; j < ctx_.num_mag_dims(); j++) {
            std::stringstream s;
            s << "magnetization/" << j;
            mag(j).hdf5_write(name__, s.str());
        }
        ctx_.comm().barrier();
    }

    void
    load(std::string name__)
    {
        HDF5_tree fin(name__, hdf5_access_t::read_only);

        int ngv;
        fin.read("/parameters/num_gvec", &ngv, 1);
        if (ngv != ctx_.gvec().num_gvec()) {
            RTE_THROW("wrong number of G-vectors");
        }
        mdarray<int, 2> gv({3, ngv});
        fin.read("/parameters/gvec", gv);

        rho().hdf5_read(name__, "density", gv);
        rho().rg().fft_transform(1);
        for (int j = 0; j < ctx_.num_mag_dims(); j++) {
            mag(j).hdf5_read(name__, "magnetization/" + std::to_string(j), gv);
            mag(j).rg().fft_transform(1);
        }
    }

    void
    save_to_xsf()
    {
        //== FILE* fout = fopen("unit_cell.xsf", "w");
        //== fprintf(fout, "CRYSTAL\n");
        //== fprintf(fout, "PRIMVEC\n");
        //== auto& lv = unit_cell_.lattice_vectors();
        //== for (int i = 0; i < 3; i++)
        //== {
        //==     fprintf(fout, "%18.12f %18.12f %18.12f\n", lv(0, i), lv(1, i), lv(2, i));
        //== }
        //== fprintf(fout, "CONVVEC\n");
        //== for (int i = 0; i < 3; i++)
        //== {
        //==     fprintf(fout, "%18.12f %18.12f %18.12f\n", lv(0, i), lv(1, i), lv(2, i));
        //== }
        //== fprintf(fout, "PRIMCOORD\n");
        //== fprintf(fout, "%i 1\n", unit_cell_.num_atoms());
        //== for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
        //== {
        //==     auto pos = unit_cell_.get_cartesian_coordinates(unit_cell_.atom(ia).position());
        //==     fprintf(fout, "%i %18.12f %18.12f %18.12f\n", unit_cell_.atom(ia).zn(), pos[0], pos[1], pos[2]);
        //== }
        //== fclose(fout);
    }

    void
    save_to_ted()
    {

        //== void write_periodic_function()
        //== {
        //==     //== mdarray<double, 3> vloc_3d_map(&vloc_it[0], fft_->size(0), fft_->size(1), fft_->size(2));
        //==     //== int nx = fft_->size(0);
        //==     //== int ny = fft_->size(1);
        //==     //== int nz = fft_->size(2);

        //==     //== auto p = parameters_.unit_cell()->unit_cell_parameters();

        //==     //== FILE* fout = fopen("potential.ted", "w");
        //==     //== fprintf(fout, "%s\n", parameters_.unit_cell()->chemical_formula().c_str());
        //==     //== fprintf(fout, "%16.10f %16.10f %16.10f  %16.10f %16.10f %16.10f\n", p.a, p.b, p.c, p.alpha,
        // p.beta, p.gamma);
        //==     //== fprintf(fout, "%i %i %i\n", nx + 1, ny + 1, nz + 1);
        //==     //== for (int i0 = 0; i0 <= nx; i0++)
        //==     //== {
        //==     //==     for (int i1 = 0; i1 <= ny; i1++)
        //==     //==     {
        //==     //==         for (int i2 = 0; i2 <= nz; i2++)
        //==     //==         {
        //==     //==             fprintf(fout, "%14.8f\n", vloc_3d_map(i0 % nx, i1 % ny, i2 % nz));
        //==     //==         }
        //==     //==     }
        //==     //== }
        //==     //== fclose(fout);
        //== }
    }

    void
    save_to_xdmf()
    {
        //== mdarray<double, 3> rho_grid(&rho_->f_it<global>(0), fft_->size(0), fft_->size(1), fft_->size(2));
        //== mdarray<double, 4> pos_grid(3, fft_->size(0), fft_->size(1), fft_->size(2));

        //== mdarray<double, 4> mag_grid(3, fft_->size(0), fft_->size(1), fft_->size(2));
        //== mag_grid.zero();

        //== // loop over 3D array (real space)
        //== for (int j0 = 0; j0 < fft_->size(0); j0++)
        //== {
        //==     for (int j1 = 0; j1 < fft_->size(1); j1++)
        //==     {
        //==         for (int j2 = 0; j2 < fft_->size(2); j2++)
        //==         {
        //==             int ir = static_cast<int>(j0 + j1 * fft_->size(0) + j2 * fft_->size(0) * fft_->size(1));
        //==             // get real space fractional coordinate
        //==             double frv[] = {double(j0) / fft_->size(0),
        //==                             double(j1) / fft_->size(1),
        //==                             double(j2) / fft_->size(2)};
        //==             r3::vector<double> rv = ctx_.unit_cell()->get_cartesian_coordinates(r3::vector<double>(frv));
        //==             for (int x = 0; x < 3; x++) pos_grid(x, j0, j1, j2) = rv[x];
        //==             if (ctx_.num_mag_dims() == 1) mag_grid(2, j0, j1, j2) = magnetization_[0]->f_it<global>(ir);
        //==             if (ctx_.num_mag_dims() == 3)
        //==             {
        //==                 mag_grid(0, j0, j1, j2) = magnetization_[1]->f_it<global>(ir);
        //==                 mag_grid(1, j0, j1, j2) = magnetization_[2]->f_it<global>(ir);
        //==             }
        //==         }
        //==     }
        //== }

        //== HDF5_tree h5_rho("rho.hdf5", true);
        //== h5_rho.write("rho", rho_grid);
        //== h5_rho.write("pos", pos_grid);
        //== h5_rho.write("mag", mag_grid);

        //== FILE* fout = fopen("rho.xdmf", "w");
        //== //== fprintf(fout, "<?xml version=\"1.0\" ?>\n"
        //== //==               "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\">\n"
        //== //==               "<Xdmf>\n"
        //== //==               "  <Domain Name=\"name1\">\n"
        //== //==               "    <Grid Name=\"fft_fine_grid\" Collection=\"Unknown\">\n"
        //== //==               "      <Topology TopologyType=\"3DSMesh\" NumberOfElements=\" %i %i %i \"/>\n"
        //== //==               "      <Geometry GeometryType=\"XYZ\">\n"
        //== //==               "        <DataItem Dimensions=\"%i %i %i 3\" NumberType=\"Float\" Precision=\"8\"
        // Format=\"HDF\">rho.hdf5:/pos</DataItem>\n"
        //== //==               "      </Geometry>\n"
        //== //==               "      <Attribute\n"
        //== //==               "           AttributeType=\"Scalar\"\n"
        //== //==               "           Center=\"Node\"\n"
        //== //==               "           Name=\"rho\">\n"
        //== //==               "          <DataItem\n"
        //== //==               "             NumberType=\"Float\"\n"
        //== //==               "             Precision=\"8\"\n"
        //== //==               "             Dimensions=\"%i %i %i\"\n"
        //== //==               "             Format=\"HDF\">\n"
        //== //==               "             rho.hdf5:/rho\n"
        //== //==               "          </DataItem>\n"
        //== //==               "        </Attribute>\n"
        //== //==               "    </Grid>\n"
        //== //==               "  </Domain>\n"
        //== //==               "</Xdmf>\n", fft_->size(0), fft_->size(1), fft_->size(2), fft_->size(0), fft_->size(1),
        // fft_->size(2), fft_->size(0), fft_->size(1), fft_->size(2));
        //== fprintf(fout, "<?xml version=\"1.0\" ?>\n"
        //==               "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\">\n"
        //==               "<Xdmf>\n"
        //==               "  <Domain Name=\"name1\">\n"
        //==               "    <Grid Name=\"fft_fine_grid\" Collection=\"Unknown\">\n"
        //==               "      <Topology TopologyType=\"3DSMesh\" NumberOfElements=\" %i %i %i \"/>\n"
        //==               "      <Geometry GeometryType=\"XYZ\">\n"
        //==               "        <DataItem Dimensions=\"%i %i %i 3\" NumberType=\"Float\" Precision=\"8\"
        // Format=\"HDF\">rho.hdf5:/pos</DataItem>\n"
        //==               "      </Geometry>\n"
        //==               "      <Attribute\n"
        //==               "           AttributeType=\"Vector\"\n"
        //==               "           Center=\"Node\"\n"
        //==               "           Name=\"mag\">\n"
        //==               "          <DataItem\n"
        //==               "             NumberType=\"Float\"\n"
        //==               "             Precision=\"8\"\n"
        //==               "             Dimensions=\"%i %i %i 3\"\n"
        //==               "             Format=\"HDF\">\n"
        //==               "             rho.hdf5:/mag\n"
        //==               "          </DataItem>\n"
        //==               "        </Attribute>\n"
        //==               "    </Grid>\n"
        //==               "  </Domain>\n"
        //==               "</Xdmf>\n", fft_->size(0), fft_->size(1), fft_->size(2), fft_->size(0), fft_->size(1),
        // fft_->size(2), fft_->size(0), fft_->size(1), fft_->size(2));
        //== fclose(fout);
    }
};

inline void
copy(Density const& src__, Density& dest__)
{
    for (int j = 0; j < src__.ctx().num_mag_dims() + 1; j++) {
        copy(src__.component(j).rg(), dest__.component(j).rg());
        if (src__.ctx().full_potential()) {
            copy(src__.component(j).mt(), dest__.component(j).mt());
        }
    }
    for (int ia = 0; ia < src__.ctx().unit_cell().num_atoms(); ia++) {
        copy(src__.density_matrix(ia), dest__.density_matrix(ia));
    }
    if (src__.ctx().hubbard_correction()) {
        copy(src__.occupation_matrix(), dest__.occupation_matrix());
    }
}

template <bool add_pseudo_core__>
inline auto
get_rho_up_dn(Density const& density__, double add_delta_rho_xc__ = 0.0, double add_delta_mag_xc__ = 0.0)
{
    PROFILE("sirius::get_rho_up_dn");

    auto& ctx      = const_cast<Simulation_context&>(density__.ctx());
    int num_points = ctx.spfft<double>().local_slice_size();

    auto rho_up = std::make_unique<Smooth_periodic_function<double>>(ctx.spfft<double>(), ctx.gvec_fft_sptr());
    auto rho_dn = std::make_unique<Smooth_periodic_function<double>>(ctx.spfft<double>(), ctx.gvec_fft_sptr());

    /* compute "up" and "dn" components and also check for negative values of density */
    double rhomin{0};
    #pragma omp parallel for reduction(min:rhomin)
    for (int ir = 0; ir < num_points; ir++) {
        r3::vector<double> m;
        for (int j = 0; j < ctx.num_mag_dims(); j++) {
            m[j] = density__.mag(j).rg().value(ir) * (1 + add_delta_mag_xc__);
        }

        double rho = density__.rho().rg().value(ir);
        if (add_pseudo_core__) {
            rho += density__.rho_pseudo_core().value(ir);
        }
        rho *= (1 + add_delta_rho_xc__);
        rhomin   = std::min(rhomin, rho);
        auto rud = get_rho_up_dn(ctx.num_mag_dims(), rho, m);

        rho_up->value(ir) = rud.first;
        rho_dn->value(ir) = rud.second;
    }

    mpi::Communicator(ctx.spfft<double>().communicator()).allreduce<double, mpi::op_t::min>(&rhomin, 1);
    if (rhomin < 0.0 && ctx.comm().rank() == 0) {
        std::stringstream s;
        s << "Interstitial charge density has negative values" << std::endl << "most negatve value : " << rhomin;
        RTE_WARNING(s);
    }
    std::array<std::unique_ptr<Smooth_periodic_function<double>>, 2> result;
    result[0] = std::move(rho_up);
    result[1] = std::move(rho_dn);
    return result;
}

} // namespace sirius

#endif // __DENSITY_HPP__
