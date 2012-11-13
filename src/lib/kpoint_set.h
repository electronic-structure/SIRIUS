
namespace sirius 
{

class kpoint_set
{
    private:
    
        std::vector<kpoint*> kpoints_;
        
        std::map<int,int> kpoint_index_by_id_;

    public:
        
        void add_kpoint(int kpoint_id, double* vk, double weight, Global& parameters)
        {
            if (kpoint_index_by_id_.count(kpoint_id))
                error(__FILE__, __LINE__, "kpoint is already in list");

            kpoints_.push_back(new kpoint(parameters, vk, weight));
            kpoint_index_by_id_[kpoint_id] = (int)kpoints_.size() - 1;

            std::vector<double> initial_occupancies(parameters.num_bands(), 0.0);

            // in case of non-magnetic, or magnetic non-collinear case occupy first N bands
            if (parameters.num_mag_dims() == 0 || parameters.num_mag_dims() == 3)
            {
                int m = parameters.num_valence_electrons() / parameters.max_occupancy();
                for (int i = 0; i < m; i++)
                    initial_occupancies[i] = double(parameters.max_occupancy());
                initial_occupancies[m] = double(parameters.num_valence_electrons() % parameters.max_occupancy());
            }
            else // otherwise occupy up and down bands
            {
                int m = parameters.num_valence_electrons() / 2;
                for (int i = 0; i < m; i++)
                    initial_occupancies[i] = initial_occupancies[i + parameters.num_fv_states()] = 1.0;
                initial_occupancies[m] = initial_occupancies[m + parameters.num_fv_states()] = 
                    0.5 * parameters.num_valence_electrons() - double(m);
            }

            kpoints_.back()->set_band_occupancies(&initial_occupancies[0]);
        }

   
        inline kpoint* operator[](int i)
        {
            assert(i >= 0 && i < (int)kpoints_.size());
            
            return kpoints_[i];
        }

        inline kpoint* kpoint_by_id(int id)
        {
            assert(kpoint_index_by_id_.count(id) == 1);
            assert(kpoint_index_by_id_[id] >= 0 && kpoint_index_by_id_[id] < (int)kpoints_.size());
            
            return kpoints_[kpoint_index_by_id_[id]];
        }
        
        void clear()
        {
            for (int ik = 0; ik < (int)kpoints_.size(); ik++)
                delete kpoints_[ik];
            
            kpoints_.clear();
            kpoint_index_by_id_.clear();
        }
        
        inline int num_kpoints()
        {
            return (int)kpoints_.size();
        }
};

};

