
namespace sirius {

class global 
{
    private:
    
        std::vector<Atom*> atoms;
        
        std::map<std::string, Atom*> atom_by_label_;
    
        std::vector<Site*> sites;
        
        std::vector<NonequivalentSite*> nonequivalent_sites;
        
        int number_of_nonequivalent_sites_;
        
    public:
    
        global()
        {
            assert(sizeof(int4) == 4);
            assert(sizeof(real8) == 8);
        }
        
        Atom& atom_by_label(std::string& label)
        {
            if (atom_by_label_.count(label) == 0)
            {
                Atom* atom = new Atom(label);
                atoms.push_back(atom);
                atom_by_label_[label] = atom;
                return (*atom);
            }
            else
                return (*atom_by_label_[label]);
        }
        
        void add_site(std::string& label, 
                      std::vector<double>& position, 
                      std::vector<double>& vector_field,
                      int equivalence_id)
        {
            assert(position.size() == 3);
            assert(vector_field.size() == 3);
            
            Atom& atom = atom_by_label(label);
            
            Site* site = new Site(atom);
        
        
        
        }


};

};

extern sirius::global sirius_global;