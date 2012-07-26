
namespace sirius {

class global 
{
    private:

        /// list of atom types with possibly identical items
        std::vector<AtomType*> atom_type_by_id_;
        
        /// unique mapping between label and atom type
        std::map<std::string, AtomType*> atom_type_by_label_;
    
        /*std::vector<Atom*> atoms;
        
        std::map<std::string, Atom*> atom_by_label_;
    
        std::vector<Site*> sites;
        
        std::vector<NonequivalentSite*> nonequivalent_sites;
        
        int number_of_nonequivalent_sites_;*/
        
        /// Bravais lattice vectors in row order
        double lattice_vectors_[3][3];
        
        /// inverse Bravais lattice vectors in column order (used to find lattice coordinates by Cartesian coordinates)
        double inverse_lattice_vectors_[3][3];
        
        /// vectors of the reciprocal lattice in row order (inverse Bravais lattice vectors scaled by 2*Pi)
        double reciprocal_lattice_vectors_[3][3];
        
        
        
        
    public:
    
        global()
        {
            assert(sizeof(int4) == 4);
            assert(sizeof(real8) == 8);
        }

        void print_info()
        {
            std::cout << "lattice vectors" << std::endl;
            for (int i = 0; i < 3; i++)
                printf("  a%1i : %18.10f %18.10f %18.10f \n", i + 1, lattice_vectors_[i][0], 
                                                                     lattice_vectors_[i][1], 
                                                                     lattice_vectors_[i][2]);
            
            std::cout << "reciprocal lattice vectors" << std::endl;
            for (int i = 0; i < 3; i++)
                printf("  b%1i : %18.10f %18.10f %18.10f \n", i + 1, reciprocal_lattice_vectors_[i][0], 
                                                                     reciprocal_lattice_vectors_[i][1], 
                                                                     reciprocal_lattice_vectors_[i][2]);
         }
        
        void set_lattice_vectors(double* a1, 
                                 double* a2, 
                                 double* a3)
        {
            for (int i = 0; i < 3; i++)
            {
                lattice_vectors_[0][i] = a1[i];
                lattice_vectors_[1][i] = a2[i];
                lattice_vectors_[2][i] = a3[i];
            }
            double a[3][3];
            memcpy(&a[0][0], &lattice_vectors_[0][0], 9 * sizeof(double));
            
            double t1;
            t1 = a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]) + 
                 a[0][1] * (a[1][2] * a[2][0] - a[1][0] * a[2][2]) + 
                 a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]);
            
            if (fabs(t1) < 1e-20)
                stop(std::cout << "lattice vectors are linearly dependent");
            
            t1 = 1.0 / t1;

            double b[3][3];
            b[0][0] = t1 * (a[1][1] * a[2][2] - a[1][2] * a[2][1]);
            b[0][1] = t1 * (a[0][2] * a[2][1] - a[0][1] * a[2][2]);
            b[0][2] = t1 * (a[0][1] * a[1][2] - a[0][2] * a[1][1]);
            b[1][0] = t1 * (a[1][2] * a[2][0] - a[1][0] * a[2][2]);
            b[1][1] = t1 * (a[0][0] * a[2][2] - a[0][2] * a[2][0]);
            b[1][2] = t1 * (a[0][2] * a[1][0] - a[0][0] * a[1][2]);
            b[2][0] = t1 * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
            b[2][1] = t1 * (a[0][1] * a[2][0] - a[0][0] * a[2][1]);
            b[2][2] = t1 * (a[0][0] * a[1][1] - a[0][1] * a[1][0]);

            memcpy(&inverse_lattice_vectors_[0][0], &b[0][0], 9 * sizeof(double));

            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    reciprocal_lattice_vectors_[i][j] = twopi * inverse_lattice_vectors_[j][i];
        }

        void add_atom_type(int atom_type_id, const std::string& label)
        {
            if (atom_type_by_label_.count(label) == 0)
                atom_type_by_label_[label] = new AtomType(label);
                
            if (atom_type_id != (int)atom_type_by_id_.size())
            {
                std::stringstream s;
                s << "index of atom type doesn't match provided atom type id" << std::endl
                  << "  atom_type_id : " << atom_type_id << std::endl
                  << "  label : " << label << std::endl 
                  << "  size of atom_type_by_id_ list : " << atom_type_by_id_.size();
                
                error(__FILE__, __LINE__, s.str().c_str());
            }
                
            atom_type_by_id_.push_back(atom_type_by_label_[label]);
        }



        
        /*AtomType& get_atom_type(std::string& label)
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
         

        void set_atom(std::string& label, 
                      std::vector<double>& position, 
                      std::vector<double>& vector_field,
                      int equivalence_id)
        {
            assert(position.size() == 3);
            assert(vector_field.size() == 3);
            
            AtomType& atom_type = get_atom_type(label);
            
            Site* site = new Site(atom);
        
        
        
        }*/


};

};

extern sirius::global sirius_global;
