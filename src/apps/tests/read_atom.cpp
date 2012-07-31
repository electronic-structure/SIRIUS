#include "sirius.h"

int main(int argn, char **argv)
{
    sirius::AtomType atom_type(1, std::string("Si"));

    atom_type.print_info();

    //JsonTree parser(std::string("H.json"));
    
    //std::vector<double> array = parser["array"].get< std::vector<double> >();
    
    //std::string s = parser["str"].get<std::string>();
    //std::cout << s << std::endl;
    //double mass = parser["mass1"].get<double>();
    //int l = parser["valence"][0]["l1"] >> l;
    //parser["mass"] >> mass;
    
    //double mass = parser["mass"](default_mass);
    
    
    //std::cout << "mass = " << mass << std::endl;
    
    //double f1 = parser["mass"](10.0);
}
