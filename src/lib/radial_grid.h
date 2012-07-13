#ifndef __RADIAL_GRID_H__
#define __RADIAL_GRID_H__

namespace sirius {

enum radial_grid_type {linear_grid, exponential_grid, linear_exponential_grid};

/// \brief radial grid for a muffin-tin or isolated atom
class radial_grid
{
    private:
        
        /// starting point
        double r0;
        
        /// infinity point
        double rinf;

        /// muffin-tin radius
        double mt_radius; 
        
        /// number of muffin-tin radial points
        int mt_nr_;

        /// list of radial points
        std::vector<double> r;
        
        /// intervals between points
        std::vector<double> dr_;
        
        /// type of radial grid
        radial_grid_type grid_type;
        
        // forbid copy constructor
        radial_grid(const radial_grid& src);

        // forbid '=' operator
        radial_grid& operator=(const radial_grid& src);
        
        void init();

    public:

        radial_grid(radial_grid_type grid_type, int mt_nr, double r0, double mt_radius, double rinf) : r0(r0), rinf(rinf),
            mt_radius(mt_radius), mt_nr_(mt_nr), grid_type(grid_type)
        {
            init();
        }
        
        radial_grid(radial_grid_type grid_type, int mt_nr, double r0, double mt_radius) : r0(r0), rinf(mt_radius),
            mt_radius(mt_radius), mt_nr_(mt_nr), grid_type(grid_type)

        {
            init();
        }

        inline double operator [](const int i)
        {
            return r[i];
        }
        
        inline double dr(const int i)
        {
            return dr_[i];
        }
        
        inline int mt_nr()
        {
            return mt_nr_;
        }
        
        inline int size()
        {
            return r.size();
        }
        
        void print_info();
};

void radial_grid::init()
{
    if (grid_type == linear_grid)
    {
        double x = r0;
        double dx = (mt_radius - r0) / (mt_nr_ - 1);
        int i = 1;
        
        while (x <= rinf + 1e-10)
        {
           r.push_back(x);
           x = r0 + dx * (i++);
        }
    }
    
    if (grid_type == exponential_grid)
    {
        double x = r0;
        int i = 1;
        
        while (x <= rinf + 1e-10)
        {
            r.push_back(x);
            x = r0 * pow((mt_radius / r0), double(i++) / (mt_nr_ - 1));
        }
    }
    
    if (grid_type == linear_exponential_grid)
    {
        double x = r0;
        double b = log(mt_radius + 1 - r0);
        int i = 1;
        
        while (x <= rinf + 1e-10)
        {
            r.push_back(x);
            x = r0 + exp(b * (i++) / double (mt_nr_ - 1)) - 1.0;
        }
    }
    
    for (int i = 0; i < (int)r.size() - 1; i++)
    {
        double d = r[i + 1] - r[i];
        dr_.push_back(d); 
    }
    
    if (rinf == mt_radius && mt_nr() != size())
    {
        stop(std::cout << "Rradial grid is wrong");
    }
}

void radial_grid::print_info()
{
    switch(grid_type)
    {
        case linear_grid:
            std::cout << "Linear grid" << std::endl;
            break;
            
        case exponential_grid:
            std::cout << "Exponential grid" << std::endl;
            break;
            
        case linear_exponential_grid:
            std::cout << "Linear exponential grid" << std::endl;
            break;
    }
    std::cout << "  number of muffin-tin points : " << mt_nr() << std::endl;
    std::cout << "  total number of points : " << size() << std::endl;
    std::cout << "  starting point : " << r[0] << std::endl;
    std::cout << "  muffin-tin point : " << r[mt_nr() - 1] << std::endl;
    std::cout << "  effective infinity point : " << r[size() - 1] << std::endl;
}

};

#endif // __RADIAL_GRID_H__
