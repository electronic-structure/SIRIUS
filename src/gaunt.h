#ifndef __GAUNT_H__
#define __GAUNT_H__

namespace sirius
{

/// Used in the {lm1, lm2} : {lm3, coefficient} way of grouping non-zero Gaunt coefficients
template <typename T>
struct gaunt_L3
{
    int lm3;
    T coef;
};

/// Used in the {lm1, lm2, coefficient} : {lm3} way of grouping non-zero Gaunt coefficients
template <typename T>
struct gaunt_L1_L2
{
    int lm1;
    int lm2;
    T coef;
};

/// Compact storage of non-zero Gaunt coefficients \f$ \langle \ell_1 m_1 | \ell_3 m_3 | \ell_2 m_2 \rangle \f$.
/** Very important! The following notation is adopted and used everywhere: lm1 and lm2 represent 'bra' and 'ket' 
    complex spherical harmonics of the Gaunt integral and lm3 represent the inner real or complex spherical harmonic. */
template <typename T>
class Gaunt_coefficients
{
    private:

        /// lmax of <lm1|
        int lmax1_;
        /// lmmax of <lm1|
        int lmmax1_;
        
        /// lmax of inner real or complex spherical harmonic
        int lmax3_;
        /// lmmax of inner real or complex spherical harmonic
        int lmmax3_;

        /// lmax of |lm2>
        int lmax2_;
        /// lmmax of |lm2>
        int lmmax2_;

        /// list of non-zero Gaunt coefficients for each lm3
        std::vector<std::vector<gaunt_L1_L2<T> > > gaunt_packed_L1_L2_;

        /// list of non-zero Gaunt coefficients for each combination of lm1, lm2
        mdarray<std::vector<gaunt_L3<T> >, 2> gaunt_packed_L3_;

    public:
        
        /// Class constructor.
        Gaunt_coefficients(int lmax1__, int lmax3__, int lmax2__) : lmax1_(lmax1__), lmax3_(lmax3__), lmax2_(lmax2__)
        {
            lmmax1_ = Utils::lmmax(lmax1_);
            lmmax3_ = Utils::lmmax(lmax3_);
            lmmax2_ = Utils::lmmax(lmax2_);

            gaunt_packed_L1_L2_.resize(lmmax3_);
            gaunt_L1_L2<T> g12;
            
            gaunt_packed_L3_.set_dimensions(lmmax1_, lmmax2_);
            gaunt_packed_L3_.allocate();
            gaunt_L3<T> g3;

            for (int l1 = 0; l1 <= lmax1_; l1++) 
            {
            for (int m1 = -l1; m1 <= l1; m1++)
            {
                int lm1 = Utils::lm_by_l_m(l1, m1);
                for (int l2 = 0; l2 <= lmax2_; l2++)
                {
                for (int m2 = -l2; m2 <= l2; m2++)
                {
                    int lm2 = Utils::lm_by_l_m(l2, m2);
                    for (int l3 = 0; l3 <= lmax3_; l3++)
                    {
                    for (int m3 = -l3; m3 <= l3; m3++)
                    {
                        int lm3 = Utils::lm_by_l_m(l3, m3);
                        T gc = SHT::gaunt<T>(l1, l3, l2, m1, m3, m2);
                        if (type_wrapper<T>::abs(gc) > 1e-12) 
                        {
                            g12.lm1 = lm1;
                            g12.lm2 = lm2;
                            g12.coef = gc;
                            gaunt_packed_L1_L2_[lm3].push_back(g12);

                            g3.lm3 = lm3;
                            g3.coef = gc;
                            gaunt_packed_L3_(lm1, lm2).push_back(g3);
                        }
                    }
                    }
                }
                }
            }
            }
        }

        /// Return number of non-zero Gaunt coefficients for a given lm3.
        inline int num_gaunt(int lm3)
        {
            assert(lm3 >= 0 && lm3 < lmmax3_);
            return (int)gaunt_packed_L1_L2_[lm3].size();
        }

        /// Return a structure containing {lm1, lm2, coef} for a given lm3 and index.
        /** Example:
            \code{.cpp}
                for (int lm3 = 0; lm3 < lmmax3; lm3++)
                {
                    for (int i = 0; i < gaunt_coefs.num_gaunt(); i++)
                    {
                        int lm1 = gaunt_coefs.gaunt(lm3, i).lm1;
                        int lm2 = gaunt_coefs.gaunt(lm3, i).lm2;
                        double coef = gaunt_coefs.gaunt(lm3, i).coef;
                        
                        // do something with lm1,lm2,lm3 and coef
                    }
                }
            \endcode
        */
        inline gaunt_L1_L2<T>& gaunt(int lm3, int idx)
        {
            assert(lm3 >= 0 && lm3 < lmmax3_);
            assert(idx >= 0 && idx < (int)gaunt_packed_L1_L2_[lm3].size());
            return gaunt_packed_L1_L2_[lm3][idx];
        }

        /// Return number of non-zero Gaunt coefficients for a combination of lm1 and lm2.
        inline int num_gaunt(int lm1, int lm2)
        {
            return (int)gaunt_packed_L3_(lm1, lm2).size();
        }
        
        /// Return a structure containing {lm3, coef} for a given lm1, lm2 and index
        inline gaunt_L3<T>& gaunt(int lm1, int lm2, int idx)
        {
            return gaunt_packed_L3_(lm1, lm2)[idx];
        }

        /// Return a sum over L3 (lm3) index of Gaunt coefficients and a complex vector.
        /** The following operation is performed:
            \f[
                \sum_{\ell_3 m_3} \langle \ell_1 m_1 | \ell_3 m_3 | \ell_2 m_2 \rangle v_{\ell_3 m_3}
            \f]
            Result is assumed to be complex.
        */
        inline complex16 sum_L3_gaunt(int lm1, int lm2, complex16* v)
        {
            complex16 zsum(0, 0);
            for (int k = 0; k < (int)gaunt_packed_L3_(lm1, lm2).size(); k++)
                zsum += gaunt_packed_L3_(lm1, lm2)[k].coef * v[gaunt_packed_L3_(lm1, lm2)[k].lm3];
            return zsum;
        }
        
        /// Return a sum over L3 (lm3) index of Gaunt coefficients and a real vector.
        /** The following operation is performed:
            \f[
                \sum_{\ell_3 m_3} \langle \ell_1 m_1 | \ell_3 m_3 | \ell_2 m_2 \rangle v_{\ell_3 m_3}
            \f]
            Result is assumed to be of the same type as Gaunt coefficients.
        */
        inline T sum_L3_gaunt(int lm1, int lm2, double* v)
        {
            T sum = 0;
            for (int k = 0; k < (int)gaunt_packed_L3_(lm1, lm2).size(); k++)
                sum += gaunt_packed_L3_(lm1, lm2)[k].coef * v[gaunt_packed_L3_(lm1, lm2)[k].lm3];
            return sum;
        }
    
        /// Return vector of non-zero Gaunt coefficients for a given combination of lm1 and lm2
        inline std::vector<gaunt_L3<T> >& gaunt_vector(int lm1, int lm2)
        {
            return gaunt_packed_L3_(lm1, lm2);
        }
};

};

#endif

