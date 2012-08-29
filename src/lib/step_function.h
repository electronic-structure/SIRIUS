
namespace sirius {

class step_function : public reciprocal_lattice
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
    
    public:

        inline double volume_mt()
        {
            return volume_mt_;
        }

        inline double volume_it()
        {
            return volume_it_;
        }
        
        inline void get_step_function(double* ptr)
        {
            memcpy(ptr, &step_function_[0], fft().size() * sizeof(double));
        }

        //TODO: candidate for parallelization 
        void init()
        {
            Timer t("sirius::sirius_step_func::init");
            
            step_function_pw_.resize(fft().size());
            step_function_.resize(fft().size());
            
            memset(&step_function_pw_[0], 0, fft().size() * sizeof(complex16));
            
            step_function_pw_[0] = omega();

            for (int ia = 0; ia < num_atoms(); ia++)
            {
                double R = atom(ia)->type()->mt_radius();

                step_function_pw_[0] -= fourpi * conj(gvec_phase_factor(0, ia)) * pow(R, 3) / 3.0;
                for (int ig = 1; ig < fft().size(); ig++)
                {
                    double g = gvec_shell_len(gvec_shell(ig));
                    double gR = g * R;
                    step_function_pw_[ig] -= fourpi * conj(gvec_phase_factor(ig, ia)) * (sin(gR) - gR * cos(gR)) / pow(g, 3);
                }
            }

            fft().transform(&step_function_pw_[0], &step_function_[0]);
            
            volume_mt_ = 0.0;
            for (int ia = 0; ia < num_atoms(); ia++)
                volume_mt_ += fourpi * pow(atom(ia)->type()->mt_radius(), 3) / 3.0; 
            
            volume_it_ = omega() - volume_mt_;
            double vit = 0.0;
            for (int i = 0; i < fft().size(); i++)
                vit += step_function_[i] * omega() / fft().size();
                
            if (fabs(vit - volume_it_) > 1e-8)
            {
                std::stringstream s;
                s << "step function gives a wrong volume for IT region" << std::endl
                  << "  difference with exact value : " << vit - volume_it_;
                error(__FILE__, __LINE__, s);
            }
        }
};

};
