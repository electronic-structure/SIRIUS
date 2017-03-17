// Copyright (c) 2013-2017 Anton Kozhevnikov, Thomas Schulthess
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

/** \file stress.h
 *
 *  \brief Stress tensor calculation.
 */

#ifndef __STRESS_H__
#define __STRESS_H__

namespace sirius {

class Stress {
  private:
    Simulation_context& ctx_;
    
    K_point_set& kset_;

    Density& density_;

    matrix3d<double> stress_kin_;

    matrix3d<double> stress_har_;

    matrix3d<double> stress_ewald_;

    matrix3d<double> stress_vloc_;
    
    inline void calc_stress_kin()
    {
        for (int ikloc = 0; ikloc < kset_.spl_num_kpoints().local_size(); ikloc++) {
            int ik = kset_.spl_num_kpoints(ikloc);
            auto kp = kset_[ik];
            if (kp->gkvec().reduced()) {
                TERMINATE("fix this");
            }

            for (int igloc = 0; igloc < kp->num_gkvec_loc(); igloc++) {
                int ig = kp->idxgk(igloc);
                auto Gk = kp->gkvec().gkvec_cart(ig);
                
                double d{0};
                for (int i = 0; i < ctx_.num_bands(); i++) {
                    double f = kp->band_occupancy(i);
                    if (f > 1e-12) {
                        auto z = kp->spinor_wave_functions(0).pw_coeffs().prime(igloc, i);
                        d += f * (std::pow(z.real(), 2) + std::pow(z.imag(), 2));
                    }
                }
                d *= kp->weight();
                for (int mu: {0, 1, 2}) {
                    for (int nu: {0, 1, 2}) {
                        stress_kin_(mu, nu) += Gk[mu] * Gk[nu] * d;
                    }
                }
            } // igloc
        } // ikloc

        ctx_.comm().allreduce(&stress_kin_(0, 0), 9 * sizeof(double));

        stress_kin_ *= (-1.0 / ctx_.unit_cell().omega());

        symmetrize(stress_kin_);
    }

    inline void calc_stress_har()
    {
        for (int igloc = 0; igloc < ctx_.gvec_count(); igloc++) {
            int ig = ctx_.gvec_offset() + igloc;
            if (!ig) {
                continue;
            }

            auto G = ctx_.gvec().gvec_cart(ig);
            double g2 = std::pow(G.length(), 2);
            auto z = density_.rho()->f_pw(ig);
            double d = twopi * (std::pow(z.real(), 2) + std::pow(z.imag(), 2)) / g2;

            for (int mu: {0, 1, 2}) {
                for (int nu: {0, 1, 2}) {
                    stress_har_(mu, nu) += d * 2 * G[mu] * G[nu] / g2;
                }
            }
            for (int mu: {0, 1, 2}) {
                stress_har_(mu, mu) -= d;
            }
        }

        if (ctx_.gvec().reduced()) {
            stress_har_ *= 2;
        } 

        ctx_.comm().allreduce(&stress_har_(0, 0), 9 * sizeof(double));

        symmetrize(stress_har_);
    }

    inline void calc_stress_ewald()
    {
        double lambda = 1.5;

        auto& uc = ctx_.unit_cell();

        for (int igloc = 0; igloc < ctx_.gvec_count(); igloc++) {
            int ig = ctx_.gvec_offset() + igloc;
            if (!ig) {
                continue;
            }

            auto G = ctx_.gvec().gvec_cart(ig);
            double g2 = std::pow(G.length(), 2);
            double g2lambda = g2 / 4.0 / lambda;

            double_complex rho(0, 0);

            for (int ia = 0; ia < uc.num_atoms(); ia++) {
                rho += ctx_.gvec_phase_factor(ig, ia) * static_cast<double>(uc.atom(ia).zn());
            }

            double a1 = twopi * std::pow(std::abs(rho) / uc.omega(), 2) * std::exp(-g2lambda) / g2;
            
            for (int mu: {0, 1, 2}) {
                for (int nu: {0, 1, 2}) {
                    stress_ewald_(mu, nu) += a1 * G[mu] * G[nu] * 2 * (g2lambda + 1) / g2;
                }
            }

            for (int mu: {0, 1, 2}) {
                stress_ewald_(mu, mu) -= a1;
            }
        }

        if (ctx_.gvec().reduced()) {
            stress_ewald_ *= 2;
        } 

        ctx_.comm().allreduce(&stress_ewald_(0, 0), 9 * sizeof(double));
        
        for (int mu: {0, 1, 2}) {
            stress_ewald_(mu, mu) += twopi * std::pow(uc.num_electrons() / uc.omega(), 2) / 4 / lambda;
        }

        for (int ia = 0; ia < uc.num_atoms(); ia++) {
            for (int i = 1; i < uc.num_nearest_neighbours(ia); i++) {
                int ja = uc.nearest_neighbour(i, ia).atom_id;
                double d = uc.nearest_neighbour(i, ia).distance;

                vector3d<double> v1 = uc.atom(ja).position() - uc.atom(ia).position() + uc.nearest_neighbour(i, ia).translation;
                auto r1 = uc.get_cartesian_coordinates(v1);
                double len = r1.length();

                if (std::abs(d - len) > 1e-12) {
                    STOP(); 
                }

                double a1 = (0.5 * uc.atom(ia).zn() * uc.atom(ja).zn() / uc.omega() / std::pow(len, 3)) *
                            (-2 * std::exp(-lambda * std::pow(len, 2)) * std::sqrt(lambda / pi) * len - gsl_sf_erfc(std::sqrt(lambda) * len)); 

                for (int mu: {0, 1, 2}) {
                    for (int nu: {0, 1, 2}) {
                        stress_ewald_(mu, nu) += a1 * r1[mu] * r1[nu];
                    }
                }
            }
        }

        symmetrize(stress_ewald_);
    }

    inline void calc_stress_vloc()
    {
        Radial_integrals_vloc ri_vloc(ctx_.unit_cell(), ctx_.pw_cutoff(), 100);
        Radial_integrals_vloc_dg ri_vloc_dg(ctx_.unit_cell(), ctx_.pw_cutoff(), 100);

        auto v = ctx_.make_periodic_function<index_domain_t::local>([&ri_vloc](int iat, double g)
                                                                    {
                                                                        return ri_vloc.value(iat, g);
                                                                    });

        auto dv = ctx_.make_periodic_function<index_domain_t::local>([&ri_vloc_dg](int iat, double g)
                                                                     {
                                                                         return ri_vloc_dg.value(iat, g);
                                                                     });
        
        double sdiag{0};

        for (int igloc = 0; igloc < ctx_.gvec_count(); igloc++) {
            int ig = ctx_.gvec_offset() + igloc;
            if (!ig) {
                continue;
            }

            auto G = ctx_.gvec().gvec_cart(ig);

            for (int mu: {0, 1, 2}) {
                for (int nu: {0, 1, 2}) {
                    stress_vloc_(mu, nu) += std::real(std::conj(density_.rho()->f_pw(ig)) * dv[igloc]) * G[mu] * G[nu];
                }
            }
            sdiag += std::real(std::conj(density_.rho()->f_pw(ig)) * v[igloc]);
        }

        if (ctx_.gvec().reduced()) {
            stress_vloc_ *= 2;
            sdiag *= 2;
        }

        std::cout << "sdiag="<<sdiag<<std::endl;

        for (int mu: {0, 1, 2}) {
            stress_vloc_(mu, mu) -= sdiag;
        }

        ctx_.comm().allreduce(&stress_vloc_(0, 0), 9 * sizeof(double));

        symmetrize(stress_vloc_);
    }

    inline void symmetrize(matrix3d<double>& mtrx__)
    {
        if (!ctx_.use_symmetry()) {
            return;
        }

        matrix3d<double> result;

        for (int i = 0; i < ctx_.unit_cell().symmetry().num_mag_sym(); i++) {
            auto R = ctx_.unit_cell().symmetry().magnetic_group_symmetry(i).spg_op.rotation;
            result = result + transpose(R) * mtrx__ * R;
        }

        mtrx__ = result * (1.0 / ctx_.unit_cell().symmetry().num_mag_sym());
    }

  public:
    Stress(Simulation_context& ctx__,
           K_point_set& kset__,
           Density& density__)
        : ctx_(ctx__)
        , kset_(kset__)
        , density_(density__)
    {
        calc_stress_kin();
        calc_stress_har();
        calc_stress_ewald();
        calc_stress_vloc();
        
        const double au2kbar = 2.94210119E5;
        stress_kin_ *= au2kbar;
        stress_har_ *= au2kbar;
        stress_ewald_ *= au2kbar;
        stress_vloc_ *= au2kbar;

        printf("== stress_kin ==\n");
        for (int mu: {0, 1, 2}) {
            printf("%12.6f %12.6f %12.6f\n", stress_kin_(mu, 0), stress_kin_(mu, 1), stress_kin_(mu, 2));
        }
        printf("== stress_har ==\n");
        for (int mu: {0, 1, 2}) {
            printf("%12.6f %12.6f %12.6f\n", stress_har_(mu, 0), stress_har_(mu, 1), stress_har_(mu, 2));
        }
        printf("== stress_ewald ==\n");
        for (int mu: {0, 1, 2}) {
            printf("%12.6f %12.6f %12.6f\n", stress_ewald_(mu, 0), stress_ewald_(mu, 1), stress_ewald_(mu, 2));
        }
        printf("== stress_vloc ==\n");
        for (int mu: {0, 1, 2}) {
            printf("%12.6f %12.6f %12.6f\n", stress_vloc_(mu, 0), stress_vloc_(mu, 1), stress_vloc_(mu, 2));
        }
    }

};

}

#endif

/** 
\page stress Stress tensor
\section stress_section1 Preliminary notes
Derivative of the G-vector in Cartesian coordinates over the lattice vector components:
\f[
  \frac{\partial G_{\tau}}{\partial a_{\mu\nu}} + ({\bf a}^{-1})_{\nu \tau} G_{\mu} = 0
\f]
Mathematica proof script:
\verbatim
A = Table[Subscript[a, i, j], {i, 1, 3}, {j, 1, 3}];
invA = Inverse[A];
B = 2*Pi*Transpose[Inverse[A]];
G = Table[Subscript[g, i], {i, 1, 3}];
gvec = B.G;
Do[
  Print[FullSimplify[
   D[gvec[[tau]], Subscript[a, mu, nu]] + invA[[nu]][[tau]]*gvec[[mu]]]],
{beta, 1, 3}, {mu, 1, 3}, {nu, 1,3}]
\endverbatim
Another relation:
\f[
  \frac{\partial}{\partial a_{\mu \nu}} \frac{1}{\sqrt{\Omega}} + \frac{1}{2} \frac{1}{\sqrt{\Omega}} ({\bf a}^{-1})_{\nu \mu} = 0
\f]
Mathematica proof script:
\verbatim
A = Table[Subscript[a, i, j], {i, 1, 3}, {j, 1, 3}];
invA = Inverse[A];
Do[
 Print[FullSimplify[
   D[1/Sqrt[Det[A]], Subscript[a, mu, nu]] + (1/2)*(1/Sqrt[Det[A]]) * invA[[nu]][[mu]]
   ]
  ],
{mu, 1, 3}, {nu, 1, 3}]
\endverbatim

Strain tensor describes the deformation of a crystal:
\f[
  r_{\mu} \rightarrow \sum_{\nu} (\delta_{\mu \nu} + \varepsilon_{\mu \nu}) r_{\nu}
\f]
Strain derivative of general position vector component:
\f[
  \frac{\partial r_{\tau}}{\partial \varepsilon_{\mu \nu}} = \delta_{\tau \mu} r_{\nu}
\f]
Strain derivative of the lengths of general position vector:
\f[
  \frac{\partial |{\bf r}|}{\partial \varepsilon_{\mu \nu}} = \sum_{\tau} \frac{\partial |{\bf r}|}{\partial r_{\tau}} 
    \frac{\partial r_{\tau}}{\partial \varepsilon_{\mu \nu}} = \sum_{\tau} \frac{ r_{\tau}}{|{\bf r}|}
    \delta_{\tau \mu} r_{\nu} = \frac{r_{\mu} r_{\nu}}{|{\bf r}|} 
\f]
Strain derivative of unit cell volume:
\f[
  \frac{\partial \Omega}{\partial \varepsilon_{\mu \nu}} = \delta_{\mu \nu} \Omega
\f]
Strain derivative of reciprocal vector:
\f[
  \frac{\partial G_{\tau}}{\partial \varepsilon_{\mu \nu}} = -\delta_{\tau \nu} G_{\mu}
\f]
Strain derivative of the length of reciprocal vector:
\f[
  \frac{\partial |{\bf G}|}{\partial \varepsilon_{\mu \nu}} = \sum_{\tau} \frac{\partial |{\bf G}|}{\partial G_{\tau}}
    \frac{\partial G_{\tau}}{\partial \varepsilon_{\mu \nu}} = -\frac{1}{|{\bf G}|}G_{\nu}G_{\mu}
\f]
Stress tensor is a reaction to strain:
\f[
  \sigma_{\mu \nu} = \frac{1}{\Omega} \frac{\partial E_{tot}}{\partial \varepsilon_{\mu \nu}} =
    \frac{1}{\Omega} \sum_{\mu' \nu'} \frac{\partial E_{tot}}{\partial a_{\mu' \nu'}} \frac{\partial a_{\mu' \nu'}}{\partial  \varepsilon_{\mu \nu}} = 
    \frac{1}{\Omega} \sum_{\nu'} \frac{\partial E_{tot}}{\partial a_{\mu \nu'}} a_{\nu \nu'} 
\f]

\section stress_section2 Derivative of beta-projectors.
We need to compute derivative of beta-projectors
\f[
  \frac{\partial}{\partial a_{\mu \nu}} \langle {\bf G+k} | \beta_{\ell m} \rangle
\f]

Derivative of the G-vector real spherical harmonics over the lattice vector components:
\f[
  \frac{\partial R_{\ell m}(\theta, \phi)}{\partial a_{\mu \nu}} = 
    \frac{\partial R_{\ell m}(\theta, \phi)}{\partial \theta} \frac{\partial \theta} {\partial a_{\mu \nu}} + 
    \frac{\partial R_{\ell m}(\theta, \phi)}{\partial \phi} \frac{\partial \phi} {\partial a_{\mu \nu}}
\f]
Derivatives of the \f$ R_{\ell m} \f$ with respect to the \f$ \theta,\, \phi\f$ angles can be tabulated up to a given \f$ \ell_{max} \f$.
The derivatives of angles are computed as following:
\f[
 \frac{\partial \theta} {\partial a_{\mu \nu}} = \sum_{\beta=1}^{3} \frac{\partial \theta}{\partial G_{\beta}} \frac{\partial G_{\beta}} {\partial a_{\mu \nu}}
\f]
\f[
 \frac{\partial \phi} {\partial a_{\mu \nu}} = \sum_{\beta=1}^{3} \frac{\partial \phi}{\partial G_{\beta}} \frac{\partial G_{\beta}} {\partial a_{\mu \nu}}
\f]
where
\f[
   \frac{\partial \theta}{\partial G_{x}} = \frac{\cos(\phi) \cos(\theta)}{G} \\
   \frac{\partial \theta}{\partial G_{y}} = \frac{\cos(\theta) \sin(\phi)}{G} \\
   \frac{\partial \theta}{\partial G_{z}} = -\frac{\sin(\theta)}{G}
\f]
and
\f[
   \frac{\partial \phi}{\partial G_{x}} = -\frac{\sin(\phi)}{\sin(\theta) G} \\
   \frac{\partial \phi}{\partial G_{y}} = \frac{\cos(\phi)}{\sin(\theta) G} \\
   \frac{\partial \phi}{\partial G_{z}} = 0
\f]
The derivative of \f$ phi \f$ has discontinuities at \f$ \theta = 0, \theta=\pi \f$. This, however, is not a problem, because
multiplication by the the derivative of \f$ R_{\ell m} \f$ removes it. The following functions have to be hardcoded:
\f[
  \frac{\partial R_{\ell m}(\theta, \phi)}{\partial \theta} \\
  \frac{\partial R_{\ell m}(\theta, \phi)}{\partial \phi} \frac{1}{\sin(\theta)} 
\f]

Derivatives of the spherical Bessel functions are computed in the same fashion:
\f[
  \frac{\partial j_{\ell}(Gx)}{\partial a_{\mu \nu}} =
  \frac{\partial j_{\ell}(Gx)}{\partial G} \frac{\partial G} {\partial a_{\mu \nu}} = 
  \frac{\partial j_{\ell}(Gx)}{\partial G} \sum_{\beta=1}^{3}\frac{\partial G}{\partial G_{\beta}} \frac{\partial G_{\beta}} {\partial a_{\mu \nu}} 
\f]
The derivatives of \f$ G \f$ are:
\f[
  \frac{\partial G}{\partial G_{x}} = \sin(\theta)\cos(\phi) \\
  \frac{\partial G}{\partial G_{y}} = \sin(\theta)\sin(\phi) \\
  \frac{\partial G}{\partial G_{z}} = \cos(\theta)
\f]

Let's write the full expression for the derivative of beta-projector matrix elements with respect to lattice vector 
components:
\f[
  \frac{\partial \langle {\bf G+k}|\beta_{\ell m} \rangle} {\partial a_{\mu \nu}} = 
    \frac{\partial} {\partial a_{\mu \nu}} \frac{4\pi}{\sqrt{\Omega}}(-i)^{\ell} R_{\ell m}(\theta_{G+k}, \phi_{G+k}) \int \beta_{\ell}(r) j_{\ell}(Gr) r^2 dr =\\
    \frac{4\pi}{\sqrt{\Omega}} (-i)^{\ell} \Bigg[ \int \beta_{\ell}(r) j_{\ell}(Gr) r^2 dr 
      \Big( \frac{\partial R_{\ell m}(\theta, \phi)}{\partial a_{\mu \nu}} - \frac{1}{2} R_{\ell m}(\theta, \phi) ({\bf a}^{-1})_{\nu \mu}  \Big) + 
       R_{\ell m}(\theta, \phi) \int \beta_{\ell}(r) \frac{\partial j_{\ell}(Gr)}{\partial a_{\mu \nu}} r^2 dr  \Bigg]
\f]

\section stress_section3 Kinetic energy contribution to stress.
Kinetic contribution to stress tensor:
\f[
  E^{kin} = \sum_{{\bf k}} w_{\bf k} \sum_i f_i \frac{1}{2} |{\bf G+k}|^2 |\psi_i({\bf G + k})|^2
\f]
Derivative over lattice vectors
\f[
  \frac{\partial  E^{kin}}{\partial a_{\mu \nu}} =
    \sum_{{\bf k}} w_{\bf k} \sum_i f_i \sum_{\tau} (G+k)_{\tau} (-{\bf a}^{-1})_{\nu \tau} (G+k)_{\mu}  |\psi_i({\bf G + k})|^2
\f]
Contribution to the stress tensor
\f[
  \sigma_{\mu \nu}^{kin} = \frac{1}{\Omega} \sum_{\nu'} \frac{\partial E^{kin}}{\partial a_{\mu \nu'}} a_{\nu \nu'} = 
    \frac{1}{\Omega} \sum_{\nu'}  \sum_{{\bf k}} w_{\bf k} \sum_i f_i \sum_{\tau} (G+k)_{\tau} (-{\bf a}^{-1})_{\nu' \tau} (G+k)_{\mu}  |\psi_i({\bf G + k})|^2 a_{\nu \nu'}  = 
    -\frac{1}{\Omega} \sum_{{\bf k}} w_{\bf k} \sum_i (G+k)_{\nu} (G+k)_{\mu} f_i |\psi_i({\bf G + k})|^2 
\f]

\section stress_section4 Hartree energy contribution to stress.
Hartree energy:
\f[
  E^{H} = \frac{1}{2} \int_{\Omega} \rho({\bf r}) V^{H}({\bf r}) d{\bf r} = 
    \frac{1}{2} \frac{1}{\Omega} \sum_{\bf G} \langle \rho | {\bf G} \rangle \langle {\bf G}| V^{H} \rangle = 
    \frac{2 \pi}{\Omega} \sum_{\bf G} \frac{|\tilde \rho({\bf G})|^2}{G^2}
\f]
where 
\f[
  \langle {\bf G} | \rho \rangle = \int e^{-i{\bf Gr}}\rho({\bf r}) d {\bf r} = \tilde \rho({\bf G})
\f]
and
\f[
  \langle {\bf G} | V^{H} \rangle = \int e^{-i{\bf Gr}}V^{H}({\bf r}) d {\bf r} = \frac{4\pi}{G^2} \tilde \rho({\bf G})
\f]

Hartree energy contribution to stress tensor:
\f[
   \sigma_{\mu \nu}^{H} = \frac{1}{\Omega} \frac{\partial  E^{H}}{\partial \varepsilon_{\mu \nu}} = 
      \frac{1}{\Omega} 2\pi \Big( \big( \frac{\partial}{\partial \varepsilon_{\mu \nu}} \frac{1}{\Omega} \big) 
        \sum_{{\bf G}} \frac{|\tilde \rho({\bf G})|^2}{G^2} + 
        \frac{1}{\Omega} \sum_{{\bf G}} |\tilde \rho({\bf G})|^2 \frac{\partial}{\partial \varepsilon_{\mu \nu}} \frac{1}{G^2} \Big) = \\
   \frac{1}{\Omega} 2\pi \Big( -\frac{1}{\Omega} \delta_{\mu \nu} \sum_{{\bf G}} \frac{|\tilde \rho({\bf G})|^2}{G^2} + 
     \frac{1}{\Omega} \sum_{\bf G} |\tilde \rho({\bf G})|^2 \sum_{\tau} \frac{-2 G_{\tau}}{G^4} \frac{\partial G_{\tau}}{\partial \varepsilon_{\mu \nu}} \Big) = \\
   2\pi \sum_{\bf G} \frac{|\rho({\bf G})|^2}{G^2} \Big( -\delta_{\mu \nu} + \frac{2}{G^2} G_{\nu} G_{\mu} \Big)
\f]

\section stress_section5 Ewald energy contribution to stress.
Ewald energy:
\f[
  E^{ion-ion} = \frac{1}{2} \sideset{}{'} \sum_{\alpha \beta {\bf T}} Z_{\alpha} Z_{\beta} 
    \frac{{\rm erfc}(\sqrt{\lambda} |{\bf r}_{\alpha} - {\bf r}_{\beta} + {\bf T}|)}{|{\bf r}_{\alpha} - {\bf r}_{\beta} + {\bf T}|} +
    \frac{2 \pi}{\Omega} \sum_{{\bf G}} \frac{e^{-\frac{G^2}{4 \lambda}}}{G^2} \Big| \sum_{\alpha} Z_{\alpha} e^{-i{\bf r}_{\alpha}{\bf G}} \Big|^2 -
    \sum_{\alpha} Z_{\alpha}^2 \sqrt{\frac{\lambda}{\pi}} - \frac{2\pi}{\Omega}\frac{N_{el}^2}{4 \lambda}
\f]
Contribution to stress tensor:
\f[
  \sigma_{\mu \nu}^{ion-ion} = \frac{1}{\Omega} \frac{\partial  E^{ion-ion}}{\partial \varepsilon_{\mu \nu}} 
\f]
Derivative of the first part:
\f[
  \frac{1}{\Omega}\frac{\partial}{\partial \varepsilon_{\mu \nu}} \frac{1}{2} \sideset{}{'} \sum_{\alpha \beta {\bf T}} Z_{\alpha} Z_{\beta} 
    \frac{{\rm erfc}(\sqrt{\lambda} |{\bf r}_{\alpha} - {\bf r}_{\beta} + {\bf T}|)}{|{\bf r}_{\alpha} - {\bf r}_{\beta} + {\bf T}|}  = 
    \frac{1}{2\Omega} \sideset{}{'} \sum_{\alpha \beta {\bf T}} Z_{\alpha} Z_{\beta} \Big( -2e^{-\lambda |{\bf r'}|^2} 
    \sqrt{\frac{\lambda}{\pi}} \frac{1}{|{\bf r'}|^2} - {\rm erfc}(\sqrt{\lambda} |{\bf r'}|) \frac{1}{|{\bf r'}|^3} \Big) r'_{\mu} r'_{\nu}
\f]
where \f$ {\bf r'} = {\bf r}_{\alpha} - {\bf r}_{\beta} + {\bf T} \f$.

Derivative of the second part:
\f[
  \frac{1}{\Omega}\frac{\partial}{\partial \varepsilon_{\mu \nu}} \frac{2\pi}{\Omega} \sum_{{\bf G}} \frac{e^{-\frac{G^2}{4 \lambda}}}{G^2} 
  \Big| \sum_{\alpha} Z_{\alpha} e^{-i{\bf r}_{\alpha}{\bf G}} \Big|^2 = 
  -\frac{2\pi}{\Omega^2} \delta_{\mu \nu} \sum_{{\bf G}} \frac{e^{-\frac{G^2}{4 \lambda}}}{G^2} \Big| 
  \sum_{\alpha} Z_{\alpha} e^{-i{\bf r}_{\alpha}{\bf G}} \Big|^2 + 
  \frac{2\pi}{\Omega^2} \sum_{\bf G} G_{\mu} G_{\nu} \frac{e^{-\frac{G^2}{4 \lambda}}}{G^2} 2 \frac{\frac{G^2}{4\lambda} + 1}{G^2}  
  \Big| \sum_{\alpha} Z_{\alpha} e^{-i{\bf r}_{\alpha}{\bf G}} \Big|^2 
\f]

Derivative of the fourth part:
\f[
  -\frac{1}{\Omega}\frac{\partial}{\partial \varepsilon_{\mu \nu}} \frac{2\pi}{\Omega}\frac{N_{el}^2}{4 \lambda} = 
     \frac{2\pi}{\Omega^2}\frac{N_{el}^2}{4 \lambda} \delta_{\mu \nu}
\f]

\section stress_section6 Local potential contribution to stress.
Energy contribution from the local part of pseudopotential:
\f[
  E^{loc} = \int \rho({\bf r}) V^{loc}({\bf r}) d{\bf r} = \frac{1}{\Omega} \sum_{\bf G} \langle \rho | {\bf G} \rangle \langle {\bf G}| V^{loc} \rangle = 
    \frac{1}{\Omega} \sum_{\bf G} \tilde \rho^{*}({\bf G}) \tilde V^{loc}({\bf G})
\f]
where
\f[
  \tilde \rho({\bf G}) = \langle {\bf G} | \rho \rangle = \int e^{-i{\bf Gr}}\rho({\bf r}) d {\bf r}
\f]
and
\f[
  \tilde V^{loc}({\bf G}) = \langle {\bf G} | V^{loc} \rangle = \int e^{-i{\bf Gr}}V^{loc}({\bf r}) d {\bf r} 
\f]
Using the expression for \f$ \tilde V^{loc}({\bf G}) \f$, the local contribution to the total energy is rewritten as
\f[
  E^{loc} = \frac{1}{\Omega} \sum_{\bf G} \tilde \rho^{*}({\bf G}) \sum_{\alpha} e^{-{\bf G\tau}_{\alpha}} 4 \pi 
    \int V_{\alpha}^{loc}(r)\frac{\sin(Gr)}{Gr} r^2 dr = 
    \frac{4\pi}{\Omega}\sum_{\bf G}\tilde \rho^{*}({\bf G})\sum_{\alpha} e^{-{\bf G\tau}_{\alpha}} 
    \Bigg( \int \Big(V_{\alpha}(r) r + Z_{\alpha}^p {\rm erf}(r) \Big) \frac{\sin(Gr)}{G} dr -  Z_{\alpha}^p \frac{e^{-\frac{G^2}{4}}}{G^2} \Bigg)
\f]
(see \link sirius::Potential::generate_local_potential \endlink for details).

Contribution to stress tensor:
\f[
   \sigma_{\mu \nu}^{loc} = \frac{1}{\Omega} \frac{\partial  E^{loc}}{\partial \varepsilon_{\mu \nu}} =
     \frac{1}{\Omega} \frac{-1}{\Omega} \delta_{\mu \nu} \sum_{\bf G}\tilde \rho^{*}({\bf G}) \tilde V^{loc}({\bf G}) + 
     \frac{4\pi}{\Omega^2} \sum_{\bf G}\tilde \rho^{*}({\bf G}) \sum_{\alpha} e^{-{\bf G\tau}_{\alpha}}
     \Bigg( \int \Big(V_{\alpha}(r) r + Z_{\alpha}^p {\rm erf}(r) \Big) \Big( \frac{r \cos (G r)}{G}-\frac{\sin (G r)}{G^2} \Big)
     \Big( -\frac{G_{\mu}G_{\nu}}{G} \Big) dr -  Z_{\alpha}^p \Big( -\frac{e^{-\frac{G^2}{4}}}{2 G}-\frac{2 e^{-\frac{G^2}{4}}}{G^3} \Big) 
     \Big( -\frac{G_{\mu}G_{\nu}}{G} \Big)  \Bigg) = \\
     -\delta_{\mu \nu} \sum_{\bf G}\rho^{*}({\bf G}) V^{loc}({\bf G}) + \sum_{\bf G} \rho^{*}({\bf G}) \Delta V^{loc}({\bf G}) G_{\mu}G_{\nu}
\f]
where \f$ \Delta V^{loc}({\bf G}) \f$ is built from the following radial integrals:
\f[
  \int \Big(V_{\alpha}(r) r + Z_{\alpha}^p {\rm erf}(r) \Big) \Big( \frac{\sin (G r)}{G^3} - \frac{r \cos (G r)}{G^2}\Big) dr - 
    Z_{\alpha}^p \Big( \frac{e^{-\frac{G^2}{4}}}{2 G^2} + \frac{2 e^{-\frac{G^2}{4}}}{G^4} \Big)  
\f]

 */
