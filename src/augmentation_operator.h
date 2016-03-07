#ifndef __AUGMENTATION_OPERATOR_H__
#define __AUGMENTATION_OPERATOR_H__

#include "sbessel.h"

namespace sirius {

class Augmentation_operator
{
    private:

        Communicator const& comm_;

        Atom_type const& atom_type_;

        mdarray<double_complex, 2> q_mtrx_;

        mdarray<double_complex, 2> q_pw_;

        mdarray<double, 2> q_pw_real_t_;
        
        //== /// Get Q-operator radial functions.
        //== mdarray<double, 3> get_radial_functions()
        //== {
        //==     /* number of radial beta-functions */
        //==     int nbrf = atom_type_.mt_radial_basis_size();
        //==     /* maximum l of beta-projectors */
        //==     int lmax_beta = atom_type_.indexr().lmax();

        //==     mdarray<double, 3> qrf(atom_type_.num_mt_points(), 2 * lmax_beta + 1, nbrf * (nbrf + 1) / 2);

        //==     for (int l3 = 0; l3 <= 2 * lmax_beta; l3++)
        //==     {
        //==         for (int idxrf2 = 0; idxrf2 < nbrf; idxrf2++)
        //==         {
        //==             for (int idxrf1 = 0; idxrf1 <= idxrf2; idxrf1++)
        //==             {
        //==                 int idx = idxrf2 * (idxrf2 + 1) / 2 + idxrf1;
        //==                 /* take initial radial function */
        //==                 std::memcpy(&qrf(0, l3, idx), &atom_type_.uspp().q_radial_functions_l(0, idx, l3), 
        //==                             atom_type_.num_mt_points() * sizeof(double));

        //==                 ///* apply polynomial approximation */ 
        //==                 //if (atom_type_.uspp().num_q_coefs)
        //==                 //    atom_type_.fix_q_radial_function(l3, idxrf1, idxrf2, &qrf(0, l3, idx));
        //==             }
        //==         }
        //==     }
        //==     return std::move(qrf);
        //== }
        
        /// Get radial integrals of Q-operator with spherical Bessel functions.
        mdarray<double, 3> get_radial_integrals(Gvec const& gvec__)
        {
            // TODO: this can be distributed over G-shells (each mpi rank holds radial integrals only for
            //       G-shells of local fraction of G-vectors

            /* number of radial beta-functions */
            int nbrf = atom_type_.mt_radial_basis_size();
            /* maximum l of beta-projectors */
            int lmax_beta = atom_type_.indexr().lmax();

            /* get radial functions */
            //== auto qrf = get_radial_functions();
            
            /* interpolate Q-operator radial functions */
            mdarray<Spline<double>, 2> qrf_spline(2 * lmax_beta + 1, nbrf * (nbrf + 1) / 2);
            
            for (int l3 = 0; l3 <= 2 * lmax_beta; l3++)
            {
                #pragma omp parallel for
                for (int idx = 0; idx < nbrf * (nbrf + 1) / 2; idx++)
                {
                    qrf_spline(l3, idx) = Spline<double>(atom_type_.radial_grid());

                    for (int ir = 0; ir < atom_type_.num_mt_points(); ir++)
                        qrf_spline(l3, idx)[ir] = atom_type_.uspp().q_radial_functions_l(ir, idx, l3); //= qrf(ir, l3, idx);

                    qrf_spline(l3, idx).interpolate();
                }
            }

            /* allocate radial integrals */
            mdarray<double, 3> qri(nbrf * (nbrf + 1) / 2, 2 * lmax_beta + 1, gvec__.num_shells());
            qri.zero();

            splindex<block> spl_num_gvec_shells(gvec__.num_shells(), comm_.size(), comm_.rank());
        
            #pragma omp parallel for
            for (int ishloc = 0; ishloc < spl_num_gvec_shells.local_size(); ishloc++)
            {
                int igs = spl_num_gvec_shells[ishloc];
                Spherical_Bessel_functions jl(2 * lmax_beta, atom_type_.radial_grid(), gvec__.shell_len(igs));

                for (int l3 = 0; l3 <= 2 * lmax_beta; l3++)
                {
                    for (int idxrf2 = 0; idxrf2 < nbrf; idxrf2++)
                    {
                        int l2 = atom_type_.indexr(idxrf2).l;
                        for (int idxrf1 = 0; idxrf1 <= idxrf2; idxrf1++)
                        {
                            int l1 = atom_type_.indexr(idxrf1).l;

                            int idx = idxrf2 * (idxrf2 + 1) / 2 + idxrf1;
                            
                            if (l3 >= std::abs(l1 - l2) && l3 <= (l1 + l2) && (l1 + l2 + l3) % 2 == 0)
                            {
                                qri(idx, l3, igs) = inner(jl(l3), qrf_spline(l3, idx), 0, atom_type_.num_mt_points());
                            }
                        }
                    }
                }
            }

            int ld = (int)(qri.size(0) * qri.size(1));
            comm_.allgather(&qri(0, 0, 0), ld * (int)spl_num_gvec_shells.global_offset(), ld * (int)spl_num_gvec_shells.local_size());

            return std::move(qri);
        }

        void generate_pw_coeffs(double omega__, Gvec const& gvec__)
        {
            PROFILE_WITH_TIMER("sirius::Augmentation_operator::generate_pw_coeffs");
        
            auto qri = get_radial_integrals(gvec__);

            double fourpi_omega = fourpi / omega__;

            /* maximum l of beta-projectors */
            int lmax_beta = atom_type_.indexr().lmax();
            int lmmax = Utils::lmmax(2 * lmax_beta);

            std::vector<int> l_by_lm = Utils::l_by_lm(2 * lmax_beta);
        
            std::vector<double_complex> zilm(lmmax);
            for (int l = 0, lm = 0; l <= 2 * lmax_beta; l++)
            {
                for (int m = -l; m <= l; m++, lm++) zilm[lm] = std::pow(double_complex(0, 1), l);
            }

            /* Gaunt coefficients of three real spherical harmonics */
            Gaunt_coefficients<double> gaunt_coefs(lmax_beta, 2 * lmax_beta, lmax_beta, SHT::gaunt_rlm);
            
            /* split G-vectors between ranks */
            splindex<block> spl_num_gvec(gvec__.num_gvec(), comm_.size(), comm_.rank());
            
            /* array of real spherical harmonics for each G-vector */
            mdarray<double, 2> gvec_rlm(Utils::lmmax(2 * lmax_beta), spl_num_gvec.local_size());
            for (int igloc = 0; igloc < spl_num_gvec.local_size(); igloc++)
            {
                int ig = spl_num_gvec[igloc];
                auto rtp = SHT::spherical_coordinates(gvec__.cart(ig));
                SHT::spherical_harmonics(2 * lmax_beta, rtp[1], rtp[2], &gvec_rlm(0, igloc));
            }
        
            /* number of beta-projectors */
            int nbf = atom_type_.mt_basis_size();
            
            q_mtrx_ = mdarray<double_complex, 2>(nbf, nbf);
            q_mtrx_.zero();

            /* array of plane-wave coefficients */
            q_pw_ = mdarray<double_complex, 2>(nbf * (nbf + 1) / 2, spl_num_gvec.local_size());
            q_pw_real_t_ = mdarray<double, 2>(2 * spl_num_gvec.local_size(), nbf * (nbf + 1) / 2);
            #pragma omp parallel for
            for (int igloc = 0; igloc < spl_num_gvec.local_size(); igloc++)
            {
                int ig = spl_num_gvec[igloc];
                int igs = gvec__.shell(ig);

                std::vector<double_complex> v(lmmax);

                for (int xi2 = 0; xi2 < nbf; xi2++)
                {
                    int lm2 = atom_type_.indexb(xi2).lm;
                    int idxrf2 = atom_type_.indexb(xi2).idxrf;
        
                    for (int xi1 = 0; xi1 <= xi2; xi1++)
                    {
                        int lm1 = atom_type_.indexb(xi1).lm;
                        int idxrf1 = atom_type_.indexb(xi1).idxrf;
                        
                        /* packed orbital index */
                        int idx12 = xi2 * (xi2 + 1) / 2 + xi1;
                        /* packed radial-function index */
                        int idxrf12 = idxrf2 * (idxrf2 + 1) / 2 + idxrf1;
                        
                        for (int lm3 = 0; lm3 < lmmax; lm3++)
                            v[lm3] = std::conj(zilm[lm3]) * gvec_rlm(lm3, igloc) * qri(idxrf12, l_by_lm[lm3], igs);
        
                        q_pw_(idx12, igloc) = fourpi_omega * gaunt_coefs.sum_L3_gaunt(lm2, lm1, &v[0]);
                        q_pw_real_t_(2 * igloc, idx12)     = q_pw_(idx12, igloc).real();
                        q_pw_real_t_(2 * igloc + 1, idx12) = q_pw_(idx12, igloc).imag();
                    }
                }
            }
    
            if (comm_.rank() == 0)
            {
                for (int xi2 = 0; xi2 < nbf; xi2++)
                {
                    for (int xi1 = 0; xi1 <= xi2; xi1++)
                    {
                        /* packed orbital index */
                        int idx12 = xi2 * (xi2 + 1) / 2 + xi1;
                        q_mtrx_(xi1, xi2) = omega__ * q_pw_(idx12, 0);
                        q_mtrx_(xi2, xi1) = std::conj(q_mtrx_(xi1, xi2));
                    }
                }
            }
            /* broadcast from rank#0 */
            comm_.bcast(&q_mtrx_(0, 0), nbf * nbf , 0);

            #ifdef __PRINT_OBJECT_CHECKSUM
            auto z = q_pw_.checksum();
            comm_.allreduce(&z, 1);
            DUMP("checksum(q_pw) : %18.10f %18.10f", z.real(), z.imag());
            #endif
        }

    public:
       
        Augmentation_operator(Communicator const& comm__,
                              Atom_type const& atom_type__,
                              Gvec const& gvec__,
                              double omega__)
            : comm_(comm__),
              atom_type_(atom_type__)
        {
            generate_pw_coeffs(omega__, gvec__);
        }

        void prepare() const
        {
            #ifdef __GPU
            if (atom_type_.parameters().processing_unit() == GPU)
            {
                q_pw_.allocate_on_device();
                q_pw_.copy_to_device();
            }
            #endif
        }

        void dismiss() const
        {
            #ifdef __GPU
            if (atom_type_.parameters().processing_unit() == GPU)
            {
                q_pw_.deallocate_on_device();
            }
            #endif
        }

        mdarray<double_complex, 2> const& q_pw() const
        {
            return q_pw_;
        }

        mdarray<double, 2> const& q_pw_real_t() const
        {
            return q_pw_real_t_;
        }

        double_complex const& q_pw(int idx__, int ig__) const
        { 
            return q_pw_(idx__, ig__);
        }

        double_complex const& q_mtrx(int xi1__, int xi2__) const
        {
            return q_mtrx_(xi1__, xi2__);
        }
};

};

#endif // __AUGMENTATION_OPERATOR_H__
