
namespace sirius {

class step_func : public reciprocal_lattice
{
    private:
    
        /// plane wave expansion coefficients of the step function
        std::vector<complex16> step_func_pw_;
        
        /// step function on the real-space grid
        std::vector<double> step_func_;
        
        /// volume of muffin tin spheres
        double volume_mt_;
        
        /// volume of interstitial region
        double volume_it_;
    
    public:

        //TODO: candidate for parallelization 
        void init()
        {
            Timer t("sirius::sirius_step_func::init");
            
            step_func_pw_.resize(fft().size());
            step_func_.resize(fft().size());
            
            memset(&step_func_pw_[0], 0, fft().size() * sizeof(complex16));
            
            step_func_pw_[0] = omega();

            for (int ig = 0; ig < fft().size(); ig++)
            {
                double vg[3];
                get_coordinates<cartesian,reciprocal>(gvec(ig), vg);
                double vglen = vector_length(vg);
                
                for (int ia = 0; ia < num_atoms(); ia++)
                {
                    complex16 zt = fourpi * exp(complex16(0.0, -twopi * scalar_product(gvec(ig), atom(ia)->position())));
                    
                    if (ig == 0)
                        step_func_pw_[ig] -= zt * pow(atom(ia)->type()->mt_radius(), 3) / 3.0; 
                    else
                    {
                        double gr = vglen * atom(ia)->type()->mt_radius();
                        step_func_pw_[ig] -= zt * (sin(gr) - gr * cos(gr)) / pow(vglen, 3);
                    }
                }
                
                step_func_pw_[ig] /= omega(); // normalization volume of Fourier transform
            }
            
            fft().transform(&step_func_pw_[0], &step_func_[0], 1);
            
            volume_mt_ = 0.0;
            for (int ia = 0; ia < num_atoms(); ia++)
                volume_mt_ += fourpi * pow(atom(ia)->type()->mt_radius(), 3) / 3.0; 
            
            volume_it_ = 0.0;
            for (int i = 0; i < fft().size(); i++)
                volume_it_ += step_func_[i] * omega() / fft().size();
                
            if (fabs(volume_mt_ + volume_it_ - omega()) > 1e-8)
            {
                std::stringstream s;
                s << "volumes of MT and IT regions don't sum up to the total unit cell volume" << std::endl
                  << "  volume difference : " << volume_mt_ + volume_it_ - omega();
                error(__FILE__, __LINE__, s);
            }
        }
};

};
