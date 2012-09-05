
namespace sirius
{

//enum data_type {real_data_type, complex_data_type};

template <typename T> class data_type_wrapper;

template<> class data_type_wrapper<double>
{
    private:
        data_type type_;
    public:
        typedef std::complex<double> complex_type_;
        inline bool real() 
        {
            return true;
        }
        data_type_wrapper() : type_(real_data_type) {}
        inline data_type operator()(void)
        {
            return type_;
        }
};

template<> class data_type_wrapper<float>
{
    private:
        data_type type_;
    public:
        typedef std::complex<float> complex_type_;
        data_type_wrapper() : type_(real_data_type) {}
        inline data_type operator()(void)
        {
            return type_;
        }
};

template<> class data_type_wrapper< std::complex<double> >
{
    private:
        data_type type_;
    public:
        typedef std::complex<double> complex_type_;
        data_type_wrapper() : type_(complex_data_type) {}
        inline data_type operator()(void)
        {
            return type_;
        }
};

template<> class data_type_wrapper< std::complex<float> >
{
    private:
        data_type type_;
    public:
        typedef std::complex<float> complex_type_;
        data_type_wrapper() : type_(complex_data_type) {}
        inline data_type operator()(void)
        {
            return type_;
        }
};


template<typename T> class PeriodicFunction
{  
    private:
        
        typedef typename data_type_wrapper<T>::complex_type_ complex_type_; 
        
        data_type_wrapper<T> data_type_;
        
        int lmax_;
        int lmmax_;

        int max_num_mt_points_;

        int num_atoms_;

        int num_gvec_;

        int num_it_points_;

        /// real spherical harmonic expansion coefficients
        mdarray<T,3> frlm_;
        
        /// complex spherical harmonic expansion coefficients
        mdarray<complex_type_, 3> fylm_;
        
        /// interstitial values defined on the FFT grid
        mdarray<T,1> fit_;
        
        /// plane-wave expansion coefficients
        mdarray<complex16,1> fpw_;
    
    public:

        void convert_to_ylm()
        {
            if (!frlm_.get_ptr()) error(__FILE__, __LINE__, "frlm array is empty");

            fylm_.allocate();
            fylm_.zero();

            for (int ia = 0; ia < num_atoms_; ia++)
                for (int ir = 0; ir < frlm_.size(1); ir++)
                    SHT::convert_frlm_to_fylm(lmax_, &frlm_(0, ir, ia), &fylm_(0, ir, ia));      

        }
        
        void convert_to_rlm()
        {
            if (!fylm_.get_ptr()) error(__FILE__, __LINE__, "fylm array is empty");

            frlm_.allocate();
            frlm_.zero();

            for (int ia = 0; ia < num_atoms_; ia++)
                for (int ir = 0; ir < frlm_.size(1); ir++)
                    SHT::convert_fylm_to_frlm(lmax_, &fylm_(0, ir, ia), &frlm_(0, ir, ia));      

        }

        void allocate(int lmax__, int max_num_mt_points__, int num_atoms__, int num_it_points__, int num_gvec__)
        {
            lmax_ = lmax__;
            lmmax_ = (lmax_ + 1) * (lmax_ + 1);
            max_num_mt_points_ = max_num_mt_points__;
            num_atoms_ = num_atoms__;
            num_it_points_ = num_it_points__;
            num_gvec_ = num_gvec__;

            frlm_.set_dimensions(lmmax_, max_num_mt_points__, num_atoms__);
            fylm_.set_dimensions(lmmax_, max_num_mt_points__, num_atoms__);
            
            if (data_type_() == real_data_type) frlm_.allocate();
            if (data_type_() == complex_data_type) fylm_.allocate();

            fit_.set_dimensions(num_it_points__);
            fit_.allocate();

            fpw_.set_dimensions(num_gvec__);
            fpw_.allocate();
        }

        void allocate_fylm()
        {
            fylm_.allocate();
        }

        void deallocate_fylm()
        {
            fylm_.deallocate();
        }

        void zero()
        {
            if (frlm_.get_ptr()) frlm_.zero();
            if (fylm_.get_ptr()) fylm_.zero();
            if (fit_.get_ptr()) fit_.zero();
            if (fpw_.get_ptr()) fpw_.zero();
        }
        
        inline T& frlm(int lm, int ir, int ia)
        {
            assert(frlm_.get_ptr());

            return frlm_(lm, ir, ia);
        }
        
        inline complex16& fylm(int lm, int ir, int ia)
        {
            assert(fylm_.get_ptr());

            return fylm_(lm, ir, ia);
        }

        inline T& fit(int ir)
        {
            return fit_(ir);
        }

        inline complex16& fpw(int ig)
        {
            return fpw_(ig);
        }

        inline void add(PeriodicFunction<T>& f)
        {
        }

};

};
