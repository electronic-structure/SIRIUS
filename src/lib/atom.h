#ifndef __ATOM_H__
#define __ATOM_H__

class Atom
{
    private:
    
        std::string symbol;
        std::string name;
        int zn;
        double mass;
        double r0;
        double mt_radius;
        double rinf;
        int mt_nr;
    
    public:
        Atom(std::string& label)
        {
        
        }

};
    

#endif // __ATOM_H__
