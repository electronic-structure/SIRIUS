#ifndef __RADIAL_GRID_H__
#define __RADIAL_GRID_H__

#include <cmath>
#include <vector>

namespace sirius {

enum radial_grid_type {linear_grid, exponential_grid, linear_exponential_grid};

/// \brief radial grid for a muffin-tin or isolated atom
class radial_grid
{
    private:
        
        /// number of muffin-tin points
        int nrmt_;

        /// starting point
        double r0;

        /// muffin-tin radius
        double rmt; 

        /// maximum effective infinity point
        double rinf;

        /// list of radial points
        std::vector<double> r;
        
        /// interval between radial points
        std::vector<double> dr;
        
        /// type of radial grid
        radial_grid_type grid_type;
        
        void init()
        {
            if (grid_type == linear_grid)
            {
                double x = r0;
                double dx = (rmt - r0) / (nrmt_ - 1);
                int i = 1;
                
                while (x <= rinf)
                {
                   r.push_back(x);
                   x = r0 + dx * (i++);
                }
            }
            
            if (grid_type == exponential_grid)
            {
                double x = r0;
                int i = 1;
                
                while (x <= rinf)
                {
                    r.push_back(x);
                    x = r0 * pow((rmt / r0), double(i++) / (nrmt_ - 1));
                }
            }
            
            if (grid_type == linear_exponential_grid)
            {
                double x = r0;
                double b = log(rmt + 1 - r0);
                int i = 1;
                
                while (x <= rinf)
                {
                    r.push_back(x);
                    x = r0 + exp(b * (i++) / double (nrmt_ - 1)) - 1.0;
                }
            }
            
            for (int i = 0; i < (int)r.size() - 1; i++)
            {
                double d = r[i + 1] - r[i];
                dr.push_back(d); 
            }
        }

        radial_grid(const radial_grid& src);

        radial_grid& operator=(const radial_grid& src);
        
    public:

        radial_grid(radial_grid_type grid_type, int nrmt_, double r0, double rmt, double rinf) : nrmt_(nrmt_), 
            r0(r0), rmt(rmt), rinf(rinf), grid_type(grid_type)
        {
            init();
        }

        radial_grid(radial_grid_type grid_type, int nrmt_, double r0, double rmt) : nrmt_(nrmt_), 
            r0(r0), rmt(rmt), rinf(rmt), grid_type(grid_type)
        {
            init();
        }
        
        inline double operator [](const int i)
        {
            return r[i];
        }
        
        inline double h(const int i)
        {
            return dr[i];
        }
        
        inline int nrmt()
        {
            return nrmt_;
        }
        
        inline int size()
        {
            return r.size();
        }
        
        void print_info()
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
            std::cout << "  number of muffin-tin points : " << nrmt() << std::endl;
            std::cout << "  total number of points : " << size() << std::endl;
            std::cout << "  starting point : " << r[0] << std::endl;
            std::cout << "  muffin-tin point : " << r[nrmt() - 1] << std::endl;
            std::cout << "  effective infinity point : " << r[size() - 1] << std::endl;
        }
};

};

#endif // __RADIAL_GRID_H__
