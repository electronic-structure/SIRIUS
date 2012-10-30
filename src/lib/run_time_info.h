namespace sirius
{

class run_time_info
{
    public:

        double pseudo_charge_error;

        double energy_veff;
        
        double energy_vha;
        
        double energy_vxc;
        
        double energy_bxc;

        double energy_exc;

        double energy_enuc;
        
        double core_eval_sum;
        
        double valence_eval_sum;

        double eval_sum;
        
        std::vector<double> core_leakage;

        std::vector<double> mt_charge;
        double it_charge;
        double total_charge;

        std::vector<double> mt_magnetization[3];
        double it_magnetization[3];
        double total_magnetization[3];
        
        double energy_fermi;


};




};
