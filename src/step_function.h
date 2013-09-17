
namespace sirius {

class Step_function : public Reciprocal_lattice
{
    private:
    
        /// plane wave expansion coefficients of the step function
        std::vector<complex16> step_function_pw_;
        
        /// step function on the real-space grid
        std::vector<double> step_function_;
        
        /// volume of muffin tin spheres
        double volume_mt_;
        
        /// volume of interstitial region
        double volume_it_;

    protected:

        void init();

     public:

        void print_info();

        void get_step_function_form_factors(mdarray<double, 2>& ffac);
        
        inline double volume_mt()
        {
            return volume_mt_;
        }

        inline double volume_it()
        {
            return volume_it_;
        }
        
        inline complex16 step_function_pw(int ig)
        {
            return step_function_pw_[ig];
        }

        inline double step_function(int ir)
        {
            return step_function_[ir];
        }

        inline double* step_function()
        {
            return &step_function_[0];
        }
};

#include "step_function.hpp"

};
