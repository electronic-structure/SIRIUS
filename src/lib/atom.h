#ifndef __ATOM_H__
#define __ATOM_H__

namespace sirius {

class Atom
{
    private:
    
        int atom_type_id_;
        
        int atom_class_id_;

        AtomType* atom_type_;
        
        double position_[3];
        
        double vector_field_[3];
    
    
    public:
    
        Atom(int atom_type_id_,
             AtomType* atom_type_,
             double* _position, 
             double* _vector_field) : atom_type_id_(atom_type_id_),
                                      atom_type_(atom_type_)
        {
            for (int i = 0; i < 3; i++)
            {
                position_[i] = _position[i];
                vector_field_[i] = _vector_field[i];
            }
        }

        inline int atom_type_id()
        {
            return atom_type_id_;
        }

        void position(double* pos)
        {
            for (int i = 0; i < 3; i++)
                pos[i] = position_[i];
        }

};

};

#endif // __ATOM_H__
