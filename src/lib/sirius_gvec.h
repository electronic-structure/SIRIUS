
namespace sirius {

class sirius_gvec : public sirius_geometry
{
    private:
        
        /// plane wave cutoff radius (in inverse a.u. of length)
        double pw_cutoff_;
        
        /// fft wrapper
        FFT3D fft_;

        /// list of G-vector fractional coordinates
        mdarray<int,2> gvec_;

        /// number of G-vectors within plane wave cutoff
        int num_gvec_;

        /// mapping between linear G-vector index and G-vector coordinates
        mdarray<int,3> index_by_gvec_;

        /// mapping betwee linear G-vector index and position in FFT buffer
        std::vector<int> fft_index_;
        

    public:
    
        sirius_gvec() : pw_cutoff_(pw_cutoff_default),
                        num_gvec_(0)
        {
        }
  
        void set_pw_cutoff(double _pw_cutoff)
        {
            pw_cutoff_ = _pw_cutoff;
        }

        void init()
        {
            Timer t("sirius::sirius_gvec::init");
            
            int max_frac_coord[] = {0, 0, 0};
            double frac_coord[3];
            // try three directions
            for (int i = 0; i < 3; i++)
            {
                double cart_coord[] = {0.0, 0.0, 0.0};
                cart_coord[i] = pw_cutoff_;
                get_coordinates<fractional, reciprocal>(cart_coord, frac_coord);
                for (int i = 0; i < 3; i++)
                    max_frac_coord[i] = std::max(max_frac_coord[i], 2 * abs(int(frac_coord[i])) + 1);
            }
            
            fft_.init(max_frac_coord);
            
            mdarray<int,2> gvec(NULL, 3, fft_.size());
            gvec.allocate();
            std::vector<double> length(fft_.size());

            int ig = 0;
            for (int i = fft_.grid_limits(0, 0); i <= fft_.grid_limits(0, 1); i++)
                for (int j = fft_.grid_limits(1, 0); j <= fft_.grid_limits(1, 1); j++)
                    for (int k = fft_.grid_limits(2, 0); k <= fft_.grid_limits(2, 1); k++)
                    {
                        gvec(0, ig) = i;
                        gvec(1, ig) = j;
                        gvec(2, ig) = k;

                        int fracc[] = {i, j, k};
                        double cartc[3];
                        get_coordinates<cartesian,reciprocal>(fracc, cartc);
                        length[ig] = vector_length(cartc);
                        ig++;
                    }

            std::vector<size_t> reorder(fft_.size());
            gsl_heapsort_index(&reorder[0], &length[0], fft_.size(), sizeof(double), compare_doubles);
           
            gvec_.set_dimensions(3, fft_.size());
            gvec_.allocate();

            num_gvec_ = 0;
            for (int i = 0; i < fft_.size(); i++)
            {
                for (int j = 0; j < 3; j++)
                    gvec_(j, i) = gvec(j, reorder[i]);
                
                if (length[reorder[i]] <= pw_cutoff_)
                    num_gvec_++;
            }

            index_by_gvec_.set_dimensions(dimension(fft_.grid_limits(0, 0), fft_.grid_limits(0, 1)),
                                          dimension(fft_.grid_limits(1, 0), fft_.grid_limits(1, 1)),
                                          dimension(fft_.grid_limits(2, 0), fft_.grid_limits(2, 1)));
            index_by_gvec_.allocate();

            fft_index_.resize(fft_.size());

            for (int ig = 0; ig < fft_.size(); ig++)
            {
                int i0 = gvec_(0, ig);
                int i1 = gvec_(1, ig);
                int i2 = gvec_(2, ig);

                // mapping from G-vector to it's index
                index_by_gvec_(i0, i1, i2) = ig;

                // mapping of FFT buffer linear index
                fft_index_[ig] = fft_.index(i0, i1, i2);
           }
        }
        
        inline FFT3D& fft()
        {
            return fft_;
        }

        inline int index_by_gvec(int i0, int i1, int i2)
        {
            return index_by_gvec_(i0, i1, i2);
        }

        inline int fft_index(int ig)
        {
            return fft_index_[ig];
        }
};

};
