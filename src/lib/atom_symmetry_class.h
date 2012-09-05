#ifndef __ATOM_SYMMETRY_CLASS_H__
#define __ATOM_SYMMETRY_CLASS_H__

namespace sirius {

class AtomSymmetryClass
{
    private:
        
        int id_;

        mdarray<double,3> radial_integrals;
        
        std::vector<int> atom_id_;
        
        AtomType* atom_type_;

        std::vector<double> spherical_potential_;
        
    public:
    
        AtomSymmetryClass(int id_, 
                          AtomType* atom_type_) : id_(id_),
                                                  atom_type_(atom_type_)
        {
        
        }

        inline int id()
        {
            return id_;
        }

        inline void add_atom_id(int _atom_id)
        {
            atom_id_.push_back(_atom_id);
        }
        
        inline int num_atoms()
        {
            return atom_id_.size();
        }

        inline int atom_id(int idx)
        {
            return atom_id_[idx];
        }

        void set_spherical_potential(std::vector<double>& veff)
        {
            spherical_potential_ = veff;
        }
};

};

#endif // __ATOM_SYMMETRY_CLASS_H__
