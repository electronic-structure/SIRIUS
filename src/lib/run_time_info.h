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
        
        double total_charge_ibz;



};




};
