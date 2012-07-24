#ifndef __RADIAL_GRID_H__
#define __RADIAL_GRID_H__

namespace sirius {

enum radial_grid_type {linear_grid, exponential_grid, linear_exponential_grid};

/// \brief radial grid for a muffin-tin or isolated atom
class RadialGrid
{
    private:
        
        /// starting (zero) point
        double origin_;
        
        /// infinity point
        double infinity_;

        /// muffin-tin radius
        double mt_radius_; 
        
        /// number of muffin-tin radial points
        int mt_num_points_;

        /// list of radial points
        std::vector<double> points_;
        
        /// intervals between points
        std::vector<double> deltas_;
        
        /// type of radial grid
        radial_grid_type grid_type_;
        
        // forbid copy constructor
        RadialGrid(const RadialGrid& src);

        // forbid '=' operator
        RadialGrid& operator=(const RadialGrid& src);
        
    public:

        RadialGrid()
        {
        
        }

        RadialGrid(radial_grid_type grid_type, 
                   int mt_num_points, 
                   double origin, 
                   double mt_radius, 
                   double infinity) 
        {
            init(grid_type, mt_num_points, origin, mt_radius, infinity);
        }
        
        RadialGrid(radial_grid_type grid_type, 
                   int mt_num_points, 
                   double origin, 
                   double mt_radius) 
        {
            init(grid_type, mt_num_points, origin, mt_radius, mt_radius);
        }
        
        void init(radial_grid_type grid_type, 
                  int _mt_num_points, 
                  double origin, 
                  double mt_radius, 
                  double infinity)
        {
            grid_type_ = grid_type;
            mt_num_points_ = _mt_num_points;
            origin_ = origin;
            mt_radius_ = mt_radius;
            infinity_ = infinity;
        
            points_.clear();
            deltas_.clear();

            double tol = 1e-10;

            if (grid_type_ == linear_grid)
            {
                double x = origin_;
                double dx = (mt_radius_ - origin_) / (mt_num_points_ - 1);
                
                while (x <= infinity_ + tol)
                {
                   points_.push_back(x);
                   x += dx;
                }
            }
            
            if (grid_type_ == exponential_grid)
            {
                double x = origin_;
                int i = 1;
                
                while (x <= infinity_ + tol)
                {
                    points_.push_back(x);
                    x = origin_ * pow((mt_radius_ / origin_), double(i++) / (mt_num_points_ - 1));
                }
            }
            
            if (grid_type_ == linear_exponential_grid)
            {
                double x = origin_;
                double b = log(mt_radius_ + 1 - origin_);
                int i = 1;
                
                while (x <= infinity_ + tol)
                {
                    points_.push_back(x);
                    x = origin_ + exp(b * (i++) / double (mt_num_points_ - 1)) - 1.0;
                }
            }
            
            for (int i = 0; i < (int)points_.size() - 1; i++)
            {
                double d = points_[i + 1] - points_[i];
                deltas_.push_back(d); 
            }
            
            if (infinity_ == mt_radius_ && mt_num_points() != size())
            {
                stop(std::cout << "radial grid is wrong");
            }
        }

        inline double operator [](const int i)
        {
            return points_[i];
        }
        
        inline double dr(const int i)
        {
            return deltas_[i];
        }
        
        inline int mt_num_points()
        {
            return mt_num_points_;
        }
        
        inline int size()
        {
            return points_.size();
        }
                
        void print_info()
        {
            switch(grid_type_)
            {
                case linear_grid:
                    std::cout << "linear grid" << std::endl;
                    break;
                    
                case exponential_grid:
                    std::cout << "exponential grid" << std::endl;
                    break;
                    
                case linear_exponential_grid:
                    std::cout << "linear exponential grid" << std::endl;
                    break;
            }
            std::cout << "  number of muffin-tin points : " << mt_num_points() << std::endl;
            std::cout << "  total number of points      : " << size() << std::endl;
            std::cout << "  starting point              : " << points_[0] << std::endl;
            std::cout << "  muffin-tin point            : " << points_[mt_num_points() - 1] << std::endl;
            std::cout << "  effective infinity point    : " << points_[size() - 1] << std::endl;
        }
};

};

#endif // __RADIAL_GRID_H__
