#ifndef __ATOM_H__
#define __ATOM_H__

namespace sirius {

class Atom
{
    private:
    
        AtomType* type_;

        AtomSymmetryClass* symmetry_class_;
        
        /// position in fractional coordinates
        double position_[3];
        
        double vector_field_[3];
    
    
    public:
    
        Atom(AtomType* _type,
             double* _position, 
             double* _vector_field) : type_(_type),
                                      symmetry_class_(NULL)
        {
            assert(_type != NULL);
                
            for (int i = 0; i < 3; i++)
            {
                position_[i] = _position[i];
                vector_field_[i] = _vector_field[i];
            }
        }

        inline AtomType* type()
        {
            return type_;
        }

        inline int type_id()
        {
            return type_->id();
        }

        inline void get_position(double* position__)
        {
            for (int i = 0; i < 3; i++)
                position__[i] = position_[i];
        }
        
        inline double* position()
        {
            return position_;
        }

        inline int symmetry_class_id()
        {
            if (symmetry_class_) 
                return symmetry_class_->id();
            else
                return -1;
        }

        inline void set_symmetry_class(AtomSymmetryClass* _symmetry_class)
        {
            symmetry_class_ = _symmetry_class;
        }
};

};

#endif // __ATOM_H__
