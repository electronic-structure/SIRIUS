/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file atom.hpp
 *
 *  \brief Contains declaration and partial implementation of sirius::Atom class.
 */

#ifndef __ATOM_HPP__
#define __ATOM_HPP__

#include "core/sht/gaunt.hpp"
#include "core/profiler.hpp"
#include "atom_symmetry_class.hpp"
#include "function3d/spheric_function.hpp"

namespace sirius {

/// Data and methods specific to the actual atom in the unit cell.
class Atom
{
  private:
    /// Type of the given atom.
    Atom_type const& type_;

    /// Symmetry class of the given atom.
    std::shared_ptr<Atom_symmetry_class> symmetry_class_;

    /// Position in fractional coordinates.
    r3::vector<double> position_;

    /// Vector field associated with the current site.
    r3::vector<double> vector_field_;

    /// Muffin-tin potential.
    mdarray<double, 2> veff_;

    /// Radial integrals of the Hamiltonian.
    mdarray<double, 3> h_radial_integrals_;

    /// Muffin-tin magnetic field.
    mdarray<double, 2> beff_[3];

    /// Radial integrals of the effective magnetic field.
    mdarray<double, 4> b_radial_integrals_;

    /// Maximum l for potential and magnetic field.
    int lmax_pot_{-1};

    /// Unsymmetrized (sampled over IBZ) occupation matrix of the L(S)DA+U method.
    mdarray<std::complex<double>, 4> occupation_matrix_;

    /// U,J correction matrix of the L(S)DA+U method
    mdarray<std::complex<double>, 4> uj_correction_matrix_;

    /// True if UJ correction is applied for the current atom.
    bool apply_uj_correction_{false};

    /// Orbital quantum number for UJ correction.
    int uj_correction_l_{-1};

    /// Auxiliary form of the D_{ij} operator matrix of the pseudo-potential method.
    /** The matrix is calculated for the scalar and vector effective fields (thus, it is real and symmetric).
     *  \f[
     *      D_{\xi \xi'}^{\alpha} = \int V({\bf r}) Q_{\xi \xi'}^{\alpha}({\bf r}) d{\bf r}
     *  \f]
     *
     *  The ionic part of the D-operator matrix is added in the D_operator class, when it is initialized.
     */
    mdarray<double, 3> d_mtrx_;

  public:
    /// Constructor.
    Atom(Atom_type const& type__, r3::vector<double> position__, r3::vector<double> vector_field__)
        : type_(type__)
        , position_(position__)
        , vector_field_(vector_field__)
    {
    }

    /// Initialize atom.
    inline void
    init()
    {
        lmax_pot_ = type().parameters().lmax_pot();

        if (type().parameters().full_potential()) {
            int lmmax = sf::lmmax(lmax_pot_);
            int nrf   = type().indexr().size();

            h_radial_integrals_ = mdarray<double, 3>({lmmax, nrf, nrf});
            h_radial_integrals_.zero();

            if (type().parameters().num_mag_dims()) {
                b_radial_integrals_ = mdarray<double, 4>({lmmax, nrf, nrf, type().parameters().num_mag_dims()});
                b_radial_integrals_.zero();
            }

            occupation_matrix_ = mdarray<std::complex<double>, 4>({16, 16, 2, 2});

            uj_correction_matrix_ = mdarray<std::complex<double>, 4>({16, 16, 2, 2});
        }

        if (!type().parameters().full_potential()) {
            int nbf = type().mt_basis_size();
            d_mtrx_ = mdarray<double, 3>({nbf, nbf, type().parameters().num_mag_dims() + 1},
                                         mdarray_label("Atom::d_mtrx_"));
            d_mtrx_.zero();
        }
    }

    /// Generate radial Hamiltonian and effective magnetic field integrals
    /** Hamiltonian operator has the following representation inside muffin-tins:
     *  \f[
     *      \hat H = -\frac{1}{2}\nabla^2 + \sum_{\ell m} V_{\ell m}(r) R_{\ell m}(\hat {\bf r}) =
     *        \underbrace{-\frac{1}{2} \nabla^2+V_{00}(r)R_{00}}_{H_{s}(r)} +\sum_{\ell=1} \sum_{m=-\ell}^{\ell}
     *         V_{\ell m}(r) R_{\ell m}(\hat {\bf r}) = \sum_{\ell m} \widetilde V_{\ell m}(r) R_{\ell m}(\hat {\bf r})
     *  \f]
     *  where
     *  \f[
     *      \widetilde V_{\ell m}(r) = \left\{ \begin{array}{ll}
     *        \frac{H_{s}(r)}{R_{00}} & \ell = 0 \\
     *        V_{\ell m}(r) & \ell > 0 \end{array} \right.
     *  \f]
     */
    inline void
    generate_radial_integrals(device_t pu__, mpi::Communicator const& comm__)
    {
        PROFILE("sirius::Atom::generate_radial_integrals");

        int lmmax        = sf::lmmax(lmax_pot_);
        int nmtp         = type().num_mt_points();
        int nrf          = type().indexr().size();
        int num_mag_dims = type().parameters().num_mag_dims();

        if (comm__.size() != 1) {
            RTE_THROW("not yet mpi parallel");
        }

        auto l_by_lm = sf::l_by_lm(lmax_pot_);

        h_radial_integrals_.zero();
        if (num_mag_dims) {
            b_radial_integrals_.zero();
        }

        if (type().parameters().cfg().settings().simple_lapw_ri()) {
            #pragma omp parallel for
            for (int lm = 0; lm < lmmax; lm++) {
                int l = l_by_lm[lm];

                for (int i2 = 0; i2 < type().indexr().size(); i2++) {
                    int l2 = type().indexr(i2).am.l();
                    for (int i1 = 0; i1 <= i2; i1++) {
                        int l1 = type().indexr(i1).am.l();
                        if ((l + l1 + l2) % 2 == 0) {
                            if (lm) {
                                Spline<double> s(type().radial_grid());
                                for (int ir = 0; ir < nmtp; ir++) {
                                    s(ir) = veff_(lm, ir) * symmetry_class().radial_function(ir, i1) *
                                            symmetry_class().radial_function(ir, i2) *
                                            std::pow(type().radial_grid(ir), 2);
                                }
                                h_radial_integrals_(lm, i1, i2) = h_radial_integrals_(lm, i2, i1) =
                                        s.interpolate().integrate(0);
                            } else {
                                h_radial_integrals_(lm, i1, i2) = symmetry_class().h_spherical_integral(i1, i2);
                                h_radial_integrals_(lm, i2, i1) = symmetry_class().h_spherical_integral(i2, i1);
                            }
                            for (int j = 0; j < num_mag_dims; j++) {
                                Spline<double> s(type().radial_grid());
                                for (int ir = 0; ir < nmtp; ir++) {
                                    s(ir) = beff_[j](lm, ir) * symmetry_class().radial_function(ir, i1) *
                                            symmetry_class().radial_function(ir, i2) *
                                            std::pow(type().radial_grid(ir), 2);
                                }
                                b_radial_integrals_(lm, i1, i2, j) = b_radial_integrals_(lm, i2, i1, j) =
                                        s.interpolate().integrate(0);
                            }
                        }
                    }
                }
            }

            return;
        }

        splindex_block<> spl_lm(lmmax, n_blocks(comm__.size()), block_id(comm__.rank()));

        /* copy radial functions to spline objects */
        std::vector<Spline<double>> rf_spline(nrf);
        #pragma omp parallel for
        for (int i = 0; i < nrf; i++) {
            rf_spline[i] = Spline<double>(type().radial_grid());
            for (int ir = 0; ir < nmtp; ir++) {
                rf_spline[i](ir) = symmetry_class().radial_function(ir, i);
            }
        }

        /* copy effective potential components to spline objects */
        std::vector<Spline<double>> v_spline(lmmax * (1 + num_mag_dims));
        #pragma omp parallel for
        for (int lm = 0; lm < lmmax; lm++) {
            v_spline[lm] = Spline<double>(type().radial_grid());
            for (int ir = 0; ir < nmtp; ir++) {
                v_spline[lm](ir) = veff_(lm, ir);
            }

            for (int j = 0; j < num_mag_dims; j++) {
                v_spline[lm + (j + 1) * lmmax] = Spline<double>(type().radial_grid());
                for (int ir = 0; ir < nmtp; ir++) {
                    v_spline[lm + (j + 1) * lmmax](ir) = beff_[j](lm, ir);
                }
            }
        }

        /* interpolate potential multiplied by a radial function */
        std::vector<Spline<double>> vrf_spline(lmmax * nrf * (1 + num_mag_dims));

        auto& idx_ri = type().idx_radial_integrals();

        mdarray<double, 1> result({idx_ri.size(1)});

        if (pu__ == device_t::GPU) {
#ifdef SIRIUS_GPU
            auto& rgrid    = type().radial_grid();
            auto& rf_coef  = type().rf_coef();
            auto& vrf_coef = type().vrf_coef();

            PROFILE_START("sirius::Atom::generate_radial_integrals|interp");
            #pragma omp parallel
            {
                #pragma omp for
                for (int i = 0; i < nrf; i++) {
                    rf_spline[i].interpolate();
                    std::copy(rf_spline[i].coeffs().at(memory_t::host),
                              rf_spline[i].coeffs().at(memory_t::host) + nmtp * 4, rf_coef.at(memory_t::host, 0, 0, i));
                    // cuda_async_copy_to_device(rf_coef.at<GPU>(0, 0, i), rf_coef.at<CPU>(0, 0, i), nmtp * 4 *
                    // sizeof(double), tid);
                }
                #pragma omp for
                for (int i = 0; i < lmmax * (1 + num_mag_dims); i++) {
                    v_spline[i].interpolate();
                }
            }
            rf_coef.copy_to(memory_t::device, acc::stream_id(-1));

            #pragma omp parallel for
            for (int lm = 0; lm < lmmax; lm++) {
                for (int i = 0; i < nrf; i++) {
                    for (int j = 0; j < num_mag_dims + 1; j++) {
                        int idx         = lm + lmmax * i + lmmax * nrf * j;
                        vrf_spline[idx] = rf_spline[i] * v_spline[lm + j * lmmax];
                        std::memcpy(vrf_coef.at(memory_t::host, 0, 0, idx), vrf_spline[idx].coeffs().at(memory_t::host),
                                    nmtp * 4 * sizeof(double));
                        // cuda_async_copy_to_device(vrf_coef.at<GPU>(0, 0, idx), vrf_coef.at<CPU>(0, 0, idx), nmtp * 4
                        // *sizeof(double), tid);
                    }
                }
            }
            vrf_coef.copy_to(memory_t::device);
            PROFILE_STOP("sirius::Atom::generate_radial_integrals|interp");

            result.allocate(memory_t::device);
            spline_inner_product_gpu_v3(idx_ri.at(memory_t::device), (int)idx_ri.size(1), nmtp,
                                        rgrid.x().at(memory_t::device), rgrid.dx().at(memory_t::device),
                                        rf_coef.at(memory_t::device), vrf_coef.at(memory_t::device),
                                        result.at(memory_t::device));
            acc::sync();
            // if (type().parameters().control().print_performance_) {
            //     double tval = t2.stop();
            //     DUMP("spline GPU integration performance: %12.6f GFlops",
            //          1e-9 * double(idx_ri.size(1)) * nmtp * 85 / tval);
            // }
            result.copy_to(memory_t::host);
            result.deallocate(memory_t::device);
#endif
        }
        if (pu__ == device_t::CPU) {
            PROFILE_START("sirius::Atom::generate_radial_integrals|interp");
            #pragma omp parallel
            {
                #pragma omp for
                for (int i = 0; i < nrf; i++) {
                    rf_spline[i].interpolate();
                }
                #pragma omp for
                for (int i = 0; i < lmmax * (1 + num_mag_dims); i++) {
                    v_spline[i].interpolate();
                }

                #pragma omp for
                for (int lm = 0; lm < lmmax; lm++) {
                    for (int i = 0; i < nrf; i++) {
                        for (int j = 0; j < num_mag_dims + 1; j++) {
                            vrf_spline[lm + lmmax * i + lmmax * nrf * j] = rf_spline[i] * v_spline[lm + j * lmmax];
                        }
                    }
                }
            }
            PROFILE_STOP("sirius::Atom::generate_radial_integrals|interp");

            PROFILE("sirius::Atom::generate_radial_integrals|inner");
            #pragma omp parallel for
            for (int j = 0; j < (int)idx_ri.size(1); j++) {
                result(j) = inner(rf_spline[idx_ri(0, j)], vrf_spline[idx_ri(1, j)], 2);
            }
            // if (type().parameters().control().print_performance_) {
            //     double tval = t2.stop();
            //     DUMP("spline CPU integration performance: %12.6f GFlops",
            //          1e-9 * double(idx_ri.size(1)) * nmtp * 85 / tval);
            // }
        }

        int n{0};
        for (int lm = 0; lm < lmmax; lm++) {
            int l = l_by_lm[lm];

            for (int i2 = 0; i2 < type().indexr().size(); i2++) {
                int l2 = type().indexr(i2).am.l();
                for (int i1 = 0; i1 <= i2; i1++) {
                    int l1 = type().indexr(i1).am.l();
                    if ((l + l1 + l2) % 2 == 0) {
                        if (lm) {
                            h_radial_integrals_(lm, i1, i2) = h_radial_integrals_(lm, i2, i1) = result(n++);
                        } else {
                            h_radial_integrals_(0, i1, i2) = symmetry_class().h_spherical_integral(i1, i2);
                            h_radial_integrals_(0, i2, i1) = symmetry_class().h_spherical_integral(i2, i1);
                        }
                        for (int j = 0; j < num_mag_dims; j++) {
                            b_radial_integrals_(lm, i1, i2, j) = b_radial_integrals_(lm, i2, i1, j) = result(n++);
                        }
                    }
                }
            }
        }

        // if (type().parameters().control().print_checksum_) {
        //     DUMP("checksum(h_radial_integrals): %18.10f", h_radial_integrals_.checksum());
        // }
    }

    /// Return const reference to corresponding atom type object.
    inline Atom_type const&
    type() const
    {
        return type_;
    }

    /// Return reference to corresponding atom symmetry class.
    inline Atom_symmetry_class&
    symmetry_class()
    {
        return (*symmetry_class_);
    }

    /// Return const referenced to atom symmetry class.
    inline Atom_symmetry_class const&
    symmetry_class() const
    {
        return (*symmetry_class_);
    }

    /// Return atom type id.
    inline int
    type_id() const
    {
        return type_.id();
    }

    /// Return atom position in fractional coordinates.
    inline r3::vector<double> const&
    position() const
    {
        return position_;
    }

    /// Set atom position in fractional coordinates.
    inline void
    set_position(r3::vector<double> position__)
    {
        position_ = position__;
    }

    /// Return vector field.
    inline auto
    vector_field() const
    {
        return vector_field_;
    }

    /// Set vector field.
    inline void
    set_vector_field(r3::vector<double> vector_field__)
    {
        vector_field_ = vector_field__;
    }

    /// Return id of the symmetry class.
    inline int
    symmetry_class_id() const
    {
        if (symmetry_class_ != nullptr) {
            return symmetry_class_->id();
        }
        return -1;
    }

    /// Set symmetry class of the atom.
    inline void
    set_symmetry_class(std::shared_ptr<Atom_symmetry_class> symmetry_class__)
    {
        symmetry_class_ = std::move(symmetry_class__);
    }

    /// Set muffin-tin potential and magnetic field.
    inline void
    set_nonspherical_potential(double* veff__, double* beff__[3])
    {
        veff_ = mdarray<double, 2>({sf::lmmax(lmax_pot_), type().num_mt_points()}, veff__);
        for (int j = 0; j < 3; j++) {
            beff_[j] = mdarray<double, 2>({sf::lmmax(lmax_pot_), type().num_mt_points()}, beff__[j]);
        }
    }

    inline void
    sync_radial_integrals(mpi::Communicator const& comm__, int const rank__)
    {
        comm__.bcast(h_radial_integrals_.at(memory_t::host), (int)h_radial_integrals_.size(), rank__);
        if (type().parameters().num_mag_dims()) {
            comm__.bcast(b_radial_integrals_.at(memory_t::host), (int)b_radial_integrals_.size(), rank__);
        }
    }

    inline void
    sync_occupation_matrix(mpi::Communicator const& comm__, int const rank__)
    {
        comm__.bcast(occupation_matrix_.at(memory_t::host), (int)occupation_matrix_.size(), rank__);
    }

    inline double const*
    h_radial_integrals(int idxrf1, int idxrf2) const
    {
        return &h_radial_integrals_(0, idxrf1, idxrf2);
    }

    inline double*
    h_radial_integrals(int idxrf1, int idxrf2)
    {
        return &h_radial_integrals_(0, idxrf1, idxrf2);
    }

    inline double const*
    b_radial_integrals(int idxrf1, int idxrf2, int x) const
    {
        return &b_radial_integrals_(0, idxrf1, idxrf2, x);
    }

    /** Compute the following kinds of sums for different spin-blocks of the Hamiltonian:
     *  \f[
     *      \sum_{L_3} \langle Y_{L_1} u_{\ell_1 \nu_1} | R_{L_3} h_{L_3} | Y_{L_2} u_{\ell_2 \nu_2} \rangle =
     *      \sum_{L_3} \langle u_{\ell_1 \nu_1} | h_{L_3} | u_{\ell_2 \nu_2} \rangle
     *                 \langle Y_{L_1} | R_{L_3} | Y_{L_2} \rangle
     *  \f]
     */
    inline auto
    radial_integrals_sum_L3(spin_block_t sblock, int idxrf1__, int idxrf2__,
                            std::vector<gaunt_L3<std::complex<double>>> const& gnt__) const
    {
        auto h_int = [this, idxrf1__, idxrf2__](int lm3) { return this->h_radial_integrals_(lm3, idxrf1__, idxrf2__); };
        auto b_int = [this, idxrf1__, idxrf2__](int lm3, int i) {
            return this->b_radial_integrals_(lm3, idxrf1__, idxrf2__, i);
        };
        /* just the Hamiltonian */
        auto nm = [h_int](const auto& gaunt_l3) { return gaunt_l3.coef * h_int(gaunt_l3.lm3); };
        /* h + Bz */
        auto uu = [h_int, b_int](const auto& gaunt_l3) {
            return gaunt_l3.coef * (h_int(gaunt_l3.lm3) + b_int(gaunt_l3.lm3, 0));
        };
        /* h - Bz */
        auto dd = [h_int, b_int](const auto& gaunt_l3) {
            return gaunt_l3.coef * (h_int(gaunt_l3.lm3) - b_int(gaunt_l3.lm3, 0));
        };
        /* Bx - i By */
        auto ud = [b_int](const auto& gaunt_l3) {
            return gaunt_l3.coef * std::complex<double>(b_int(gaunt_l3.lm3, 1), -b_int(gaunt_l3.lm3, 2));
        };
        /* Bx + i By */
        auto du = [b_int](const auto& gaunt_l3) {
            return gaunt_l3.coef * std::complex<double>(b_int(gaunt_l3.lm3, 1), b_int(gaunt_l3.lm3, 2));
        };

        std::complex<double> res{0};
        switch (sblock) {
            case spin_block_t::nm: {
                res = std::transform_reduce(gnt__.begin(), gnt__.end(), std::complex<double>{0}, std::plus{}, nm);
                break;
            }
            case spin_block_t::uu: {
                res = std::transform_reduce(gnt__.begin(), gnt__.end(), std::complex<double>{0}, std::plus{}, uu);
                break;
            }
            case spin_block_t::dd: {
                res = std::transform_reduce(gnt__.begin(), gnt__.end(), std::complex<double>{0}, std::plus{}, dd);
                break;
            }
            case spin_block_t::ud: {
                res = std::transform_reduce(gnt__.begin(), gnt__.end(), std::complex<double>{0}, std::plus{}, ud);
                break;
            }
            case spin_block_t::du: {
                res = std::transform_reduce(gnt__.begin(), gnt__.end(), std::complex<double>{0}, std::plus{}, du);
                break;
            }
            default: {
                RTE_THROW("unknown value for spin_block_t");
            }
        }

        return res;
    }

    inline int
    num_mt_points() const
    {
        return type_.num_mt_points();
    }

    inline Radial_grid<double> const&
    radial_grid() const
    {
        return type_.radial_grid();
    }

    inline double
    radial_grid(int idx) const
    {
        return type_.radial_grid(idx);
    }

    inline double
    mt_radius() const
    {
        return type_.mt_radius();
    }

    inline int
    zn() const
    {
        return type_.zn();
    }

    inline int
    mt_basis_size() const
    {
        return type_.mt_basis_size();
    }

    inline int
    mt_aw_basis_size() const
    {
        return type_.mt_aw_basis_size();
    }

    inline int
    mt_lo_basis_size() const
    {
        return type_.mt_lo_basis_size();
    }

    inline void
    set_occupation_matrix(const std::complex<double>* source)
    {
        std::memcpy(occupation_matrix_.at(memory_t::host), source, 16 * 16 * 2 * 2 * sizeof(std::complex<double>));
        apply_uj_correction_ = false;
    }

    inline void
    get_occupation_matrix(std::complex<double>* destination)
    {
        std::memcpy(destination, occupation_matrix_.at(memory_t::host), 16 * 16 * 2 * 2 * sizeof(std::complex<double>));
    }

    inline void
    set_uj_correction_matrix(const int l, const std::complex<double>* source)
    {
        uj_correction_l_ = l;
        std::memcpy(uj_correction_matrix_.at(memory_t::host), source, 16 * 16 * 2 * 2 * sizeof(std::complex<double>));
        apply_uj_correction_ = true;
    }

    inline bool
    apply_uj_correction()
    {
        return apply_uj_correction_;
    }

    inline int
    uj_correction_l()
    {
        return uj_correction_l_;
    }

    inline auto
    uj_correction_matrix(int lm1, int lm2, int ispn1, int ispn2)
    {
        return uj_correction_matrix_(lm1, lm2, ispn1, ispn2);
    }

    inline double&
    d_mtrx(int xi1, int xi2, int iv)
    {
        return d_mtrx_(xi1, xi2, iv);
    }

    inline double const&
    d_mtrx(int xi1, int xi2, int iv) const
    {
        return d_mtrx_(xi1, xi2, iv);
    }

    inline auto const&
    d_mtrx() const
    {
        return d_mtrx_;
    }

    inline auto&
    d_mtrx()
    {
        return d_mtrx_;
    }
};

} // namespace sirius

#endif // __ATOM_H__
