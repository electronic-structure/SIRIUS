namespace sirius
{

class Force
{
    private:

        static void ibs_force(Global& parameters_, Band* band, K_point* kp, mdarray<double, 2>& ffac, mdarray<double, 2>& force);

    public:

        static void total_force(Global& parameters_, Potential* potential, Density* density, K_set* ks, 
                                mdarray<double, 2>& force);

};

#include "force.hpp"

}
