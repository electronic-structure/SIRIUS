class Platform
{
    private:

        /// number of OMP threads
        static int num_threads_;
    
    public:

        static void initialize()
        {
            num_threads_ = omp_get_num_threads();
        }

        static int num_threads()
        {
            return num_threads_;
        }
};

int Platform::num_threads_ = 1; 
