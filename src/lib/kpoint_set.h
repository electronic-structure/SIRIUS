
namespace sirius 
{

class kpoint_set
{
    private:
    
        std::vector<kpoint*> kpoints_;
        
        std::map<int,int> kpoint_index_by_id_;

    public:
        
        void add_kpoint(int kpoint_id, double* vk, double weight)
        {
            if (kpoint_index_by_id_.count(kpoint_id))
                error(__FILE__, __LINE__, "kpoint is already in list");

            kpoints_.push_back(new kpoint(vk, weight));
            kpoint_index_by_id_[kpoint_id] = kpoints_.size() - 1;

            std::vector<double> initial_occupancies(global.num_bands(), 0.0);

            // in case of non-magnetic, or magnetic non-collinear case occupy first N bands
            if (global.num_dmat() == 1 || global.num_dmat() == 4)
            {
                int m = global.num_valence_electrons() / global.max_occupancy();
                for (int i = 0; i < m; i++)
                    initial_occupancies[i] = double(global.max_occupancy());
                initial_occupancies[m] = double(global.num_valence_electrons() % global.max_occupancy());
            }
            else // otherwise occupy up and down bands
            {
                int m = global.num_valence_electrons() / 2;
                for (int i = 0; i < m; i++)
                    initial_occupancies[i] = initial_occupancies[i + global.num_fv_states()] = 1.0;
                initial_occupancies[m] = initial_occupancies[m + global.num_fv_states()] = 
                    0.5 * global.num_valence_electrons() - double(m);
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
            return kpoints_.size();
        }
};

};

