#ifndef __RADIAL_GRID_H__
#define __RADIAL_GRID_H__

namespace sirius {

enum radial_grid_type {linear_grid, exponential_grid, linear_exponential_grid, pow3_grid};

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
        int num_mt_points_;

        /// list of radial points
        std::vector<double> points_;
        
        /// list of 1/r
        std::vector<double> points_inv_;
        
        /// intervals between points
        std::vector<double> deltas_;
        
        // forbid copy constructor
        RadialGrid(const RadialGrid& src);

        // forbid assignment operator
        RadialGrid& operator=(const RadialGrid& src);
        
        inline void init_dr_ir()
        {
            deltas_.resize(points_.size() - 1);
            for (int i = 0; i < (int)points_.size() - 1; i++) deltas_[i] = points_[i + 1] - points_[i];
            
            points_inv_.resize(points_.size());
            for (int i = 0; i < (int)points_.size(); i++) points_inv_[i] = 1.0 / points_[i];
        }

    public:

        RadialGrid()
        {
        
        }

        RadialGrid(radial_grid_type grid_type, int num_mt_points, double origin, double mt_radius, double infinity) 
        {
            init(grid_type, num_mt_points, origin, mt_radius, infinity);
        }
        
        RadialGrid(radial_grid_type grid_type, int num_mt_points, double origin, double mt_radius) 
        {
            init(grid_type, num_mt_points, origin, mt_radius, mt_radius);
        }
        
        void init(radial_grid_type grid_type, int num_mt_points__, double origin, double mt_radius, double infinity)
        {
            num_mt_points_ = num_mt_points__;
            origin_ = origin;
            mt_radius_ = mt_radius;
            infinity_ = infinity;
        
            points_.clear();
            deltas_.clear();
            points_inv_.clear();

            double tol = 1e-10;

            if (grid_type == linear_grid)
            {
                double x = origin_;
                double dx = (mt_radius_ - origin_) / (num_mt_points_ - 1);
                
                while (x <= infinity_ + tol)
                {
                   points_.push_back(x);
                   x += dx;
                }
            }
            
            if (grid_type == exponential_grid)
            {
                double x = origin_;
                int i = 1;
                
                while (x <= infinity_ + tol)
                {
                    points_.push_back(x);
                    x = origin_ * pow((mt_radius_ / origin_), double(i++) / (num_mt_points_ - 1));
                }
            }
            
            if (grid_type == linear_exponential_grid)
            {
                double x = origin_;
                double b = log(mt_radius_ + 1 - origin_);
                int i = 1;
                
                while (x <= infinity_ + tol)
                {
                    points_.push_back(x);
                    x = origin_ + exp(b * (i++) / double (num_mt_points_ - 1)) - 1.0;
                }
            }
            
            if (grid_type == pow3_grid)
            {
                double x = origin_;
                int i = 1;
                
                while (x <= infinity_ + tol)
                {
                    points_.push_back(x);
                    double t = double(i++) / double(num_mt_points_ - 1);
                    x = origin_ + (mt_radius_ - origin_) * pow(t, 3);
                }

            }
            
            //**if (grid_type == hyperbolic)
            //**{
            //**    double x = origin_;
            //**    int i = 1;
            //**    
            //**    while (x <= infinity_ + tol)
            //**    {
            //**        points_.push_back(x);
            //**        double t = double(i++) / double(num_mt_points_ - 1);
            //**        x = origin_ + 2.0 * (mt_radius_ - origin_) * t / (t + 1);
            //**    }

            //**}
            
            if (mt_radius_ == infinity_ && num_mt_points() != size()) 
                error(__FILE__, __LINE__, "radial grid is wrong");

            init_dr_ir();
        }
        
        inline double* get_ptr()
        {
            return &points_[0];
        }

        inline double operator [](const int i)
        {
            return points_[i];
        }
        
        inline double dr(const int i)
        {
            return deltas_[i];
        }

        inline double rinv(const int i)
        {
            return points_inv_[i];
        }
        
        inline int num_mt_points()
        {
            return num_mt_points_;
        }
        
        inline int size()
        {
            return (int)points_.size();
        }
                
        void print_info()
        {
            printf("  number of muffin-tin points : %i\n", num_mt_points());
            printf("  total number of points      : %i\n", size());
            printf("  starting point              : %f\n", points_[0]);
            printf("  muffin-tin point            : %f\n", points_[num_mt_points() - 1]);
            printf("  effective infinity point    : %f\n", points_[size() - 1]);
        }

        void set_radial_points(int num_radial_points__, int num_mt_points__, double* radial_points__)
        {
            num_mt_points_ = num_mt_points__;

            points_.resize(num_radial_points__);
            for (int i = 0; i < num_radial_points__; i++)
            {
                points_[i] = radial_points__[i];
            }

            init_dr_ir();
        }
};

};

#endif // __RADIAL_GRID_H__
