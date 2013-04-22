#ifndef __GAUNT_H__
#define __GAUNT_H__

namespace sirius
{

struct complex_gaunt_L3
{
    int lm3;
    complex16 cg;
};

struct complex_gaunt_L1_L2
{
    int lm1;
    int lm2;
    complex16 cg;
};

class GauntCoefficients
{
    private:

        int lmax1_;
        int lmmax1_;

        int lmax2_;
        int lmmax2_;

        int lmax3_;
        int lmmax3_;

        // for each lm3 -> vector of {lm1, lm2, <Ylm1|Rlm3|Ylm2>}
        std::vector< std::vector<complex_gaunt_L1_L2> > complex_gaunt_packed_L1_L2_;

        mdarray<std::vector<complex_gaunt_L3>, 2> complex_gaunt_packed_L3_;

    public:

        GauntCoefficients()
        {
        }

        void set_lmax(int lmax1__, int lmax2__, int lmax3__)
        {
            lmax1_ = lmax1__;
            lmmax1_ = Utils::lmmax_by_lmax(lmax1_);
            lmax2_ = lmax2__;
            lmmax2_ = Utils::lmmax_by_lmax(lmax2_);
            lmax3_ = lmax3__;
            lmmax3_ = Utils::lmmax_by_lmax(lmax3_);
       
            complex_gaunt_packed_L1_L2_.resize(lmmax3_);
            complex_gaunt_L1_L2 g12;

            complex_gaunt_packed_L3_.set_dimensions(lmmax1_, lmmax2_);
            complex_gaunt_packed_L3_.allocate();
            complex_gaunt_L3 g3;
            
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
                        complex16 z = SHT::complex_gaunt(l1, l3, l2, m1, m3, m2);
                        if (abs(z) > 1e-12) 
                        {
                            g12.lm1 = lm1;
                            g12.lm2 = lm2;
                            g12.cg = z;
                            complex_gaunt_packed_L1_L2_[lm3].push_back(g12);

                            g3.lm3 = lm3;
                            g3.cg = z;
                            complex_gaunt_packed_L3_(lm1, lm2).push_back(g3);
                        }
                    }
                    }
                }
                }
            }
            }
        }

        inline int complex_gaunt_packed_L1_L2_size(int lm3)
        {
            assert(lm3 >= 0 && lm3 < lmmax3_);
            return (int)complex_gaunt_packed_L1_L2_[lm3].size();
        }
        
        inline complex_gaunt_L1_L2& complex_gaunt_packed_L1_L2(int lm3, int idx)
        {
            assert(lm3 >= 0 && lm3 < lmmax3_);
            assert(idx >= 0 && idx < (int)complex_gaunt_packed_L1_L2_[lm3].size());
            return complex_gaunt_packed_L1_L2_[lm3][idx];
        }
        
        template <typename T>
        inline complex16 sum_L3_complex_gaunt(int lm1, int lm2, T* v)
        {
            complex16 zsum(0, 0);
            for (int k = 0; k < (int)complex_gaunt_packed_L3_(lm1, lm2).size(); k++)
                zsum += complex_gaunt_packed_L3_(lm1, lm2)[k].cg * v[complex_gaunt_packed_L3_(lm1, lm2)[k].lm3];
            return zsum;
        }
        
        inline int complex_gaunt_packed_L3_size(int lm1, int lm2)
        {
            return (int)complex_gaunt_packed_L3_(lm1, lm2).size();
        }

        inline complex_gaunt_L3& complex_gaunt_packed_L3(int lm1, int lm2, int k)
        {
            return complex_gaunt_packed_L3_(lm1, lm2)[k];
        }
};

};

#endif

