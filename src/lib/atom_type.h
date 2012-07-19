#ifndef __ATOM_TYPE_H__
#define __ATOM_TYPE_H__

namespace sirius {

class AtomType
{
    private:
    
        std::string label;
        std::string symbol;
        std::string name;
        int zn;
        double mass;
        double r0;
        double rinf;
        double mt_radius;
        int mt_nr;
        
        RadialGrid radial_grid;

        void read_input()
        {
            std::string fname = label + std::string(".json");
            JsonTree parser(fname);
            parser["mass"] >> mass;
        }
    
    public:

        AtomType(std::string& label) : label(label)
        {
            read_input();
        }

};
    
};

#endif // __ATOM_TYPE_H__

