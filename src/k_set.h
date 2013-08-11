
namespace sirius 
{

class K_set
{
    private:
    
        Global& parameters_;

        Band* band_;

        std::vector<K_point*> kpoints_;

        splindex<block> spl_num_kpoints_;

    public:

        K_set(Global& parameters__) : parameters_(parameters__)
        {
            band_ = new Band(parameters_);
        }

        ~K_set()
        {
            clear();
            delete band_;
        }
        
        /// Initialize the k-point set
        void initialize();

        void update();
        
        /// Find eigen-states
        void find_eigen_states(Potential* potential, bool precompute);

        /// Find Fermi energy and band occupation numbers
        void find_band_occupancies();

        /// Return sum of valence eigen-values
        double valence_eval_sum();

        void print_info();

        void sync_band_energies();
        
        void save_wave_functions();

        void load_wave_functions();
        
        int max_num_gkvec();

        void force(mdarray<double, 2>& forcek);
        
        void add_kpoint(double* vk__, double weight__)
        {
            kpoints_.push_back(new K_point(parameters_, vk__, weight__));
        }

        void add_kpoints(mdarray<double, 2>& kpoints__, double* weights__)
        {
            for (int ik = 0; ik < kpoints__.size(1); ik++) add_kpoint(&kpoints__(0, ik), weights__[ik]);
        }

        inline K_point* operator[](int i)
        {
            assert(i >= 0 && i < (int)kpoints_.size());
            
            return kpoints_[i];
        }

        void clear()
        {
            for (int ik = 0; ik < (int)kpoints_.size(); ik++) delete kpoints_[ik];
            
            kpoints_.clear();
        }
        
        inline int num_kpoints()
        {
            return (int)kpoints_.size();
        }

        inline splindex<block>& spl_num_kpoints()
        {
            return spl_num_kpoints_;
        }
        
        inline int spl_num_kpoints(int ikloc)
        {
            return spl_num_kpoints_[ikloc];
        }

        void set_band_occupancies(int ik, double* band_occupancies)
        {
            kpoints_[ik]->set_band_occupancies(band_occupancies);
        }
        
        void get_band_energies(int ik, double* band_energies)
        {
            kpoints_[ik]->get_band_energies(band_energies);
        }
        
        void get_band_occupancies(int ik, double* band_occupancies)
        {
            kpoints_[ik]->get_band_occupancies(band_occupancies);
        }

        Band* band()
        {
            return band_;
        }
};

#include "k_set.hpp"

};

