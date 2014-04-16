#include "radial_grid.h"

namespace sirius
{

std::vector<double> Radial_grid::create_radial_grid(radial_grid_t grid_type, int num_mt_points, double origin, 
                                                    double mt_radius, double infinity)
{
    std::vector<double> grid_points;

    double tol = 1e-10;

    switch (grid_type)
    {
        case linear_grid:
        {
            double x = origin;
            double dx = (mt_radius - origin) / (num_mt_points - 1);
            
            while (x <= infinity + tol)
            {
               grid_points.push_back(x);
               x += dx;
            }
            break;
        }
        case exponential_grid:
        {
            double x = origin;
            int i = 1;
            
            while (x <= infinity + tol)
            {
                grid_points.push_back(x);
                x = origin * pow((mt_radius / origin), double(i++) / (num_mt_points - 1));
            }
            break;
        }
        case linear_exponential_grid:
        {
            double S = 10000.0; // scale - ratio of last and first delta_x inside MT
            double alpha = pow(S, 1.0 / (num_mt_points - 2));
            int i = 0;
            double x = origin;
            while (x <= infinity + tol)
            {
                grid_points.push_back(x);
                x += (mt_radius - origin) * (alpha - 1) * pow(S, double(i) / (num_mt_points - 2)) / (S * alpha - 1);
                if (x <= mt_radius) i++;
            }
            break;
        }
        case pow_grid:
        {
            double x = origin;
            int i = 1;
            
            while (x <= infinity + tol)
            {
                grid_points.push_back(x);
                double t = double(i++) / double(num_mt_points - 1);
                //x = origin + (mt_radius - origin) * pow(t, 1.0 + 1.0 / (t + 1));
                x = origin + (mt_radius - origin) * pow(t, 2);
            }
            break;
        }
        case hyperbolic_grid:
        {
            double x = origin;
            int i = 1;
            
            while (x <= infinity + tol)
            {
                grid_points.push_back(x);
                double t = double(i++) / double(num_mt_points - 1);
                x = origin + 2.0 * (mt_radius - origin) * t / (t + 1);
            }
            break;
        }
        case incremental_grid:
        {
            double D = mt_radius - origin;
            double S = 1000.0;
            double dx0 = 2 * D / (2 * (num_mt_points - 1) + (num_mt_points - 1) * (S - 1));
            double alpha = (S - 1) * dx0 / (num_mt_points - 2);

            int i = 0;
            double x = origin;
            while (x <= infinity + tol)
            {
                grid_points.push_back(x);
                x = origin + (dx0 + dx0 + i * alpha) * (i + 1) / 2.0;
                i++;
            }
            break;
        }
    }
   
    // trivial check
    if (mt_radius == infinity && num_mt_points != (int)grid_points.size()) 
    {
        std::stringstream s;
        s << "Wrong radial grid" << std::endl
          << "  num_mt_points      : " << num_mt_points << std::endl
          << "  grid_points.size() : " <<  grid_points.size() << std::endl
          << "  infinity        : " << infinity << std::endl
          << "  last grid point : " << grid_points[grid_points.size() - 1]; 
        error_local(__FILE__, __LINE__, s);
    }

    return grid_points;
}

void Radial_grid::initialize(radial_grid_t grid_type, double origin, double infinity)
{
    assert(mt_radius_ > 0);
    assert(num_mt_points_ > 0);
    assert(origin > 0);
    assert(infinity > 0 && infinity > origin && infinity >= mt_radius_);

    std::vector<double> grid_points = create_radial_grid(grid_type, num_mt_points_, origin, mt_radius_, infinity);
    set_radial_points((int)grid_points.size(), &grid_points[0]);

    switch (grid_type)
    {
        case linear_grid:
        {
            grid_type_name_ = "linear";
            break;
        }
        case exponential_grid:
        {
            grid_type_name_ = "exponential";
            break;
        }
        case linear_exponential_grid:
        {
            grid_type_name_ = "linear-exponential";
            break;
        }
        case pow_grid:
        {
            grid_type_name_ = "power";
            break;
        }
        case hyperbolic_grid:
        {
            grid_type_name_ = "hyperbolic";
            break;
        }
        case incremental_grid:
        {
            grid_type_name_ = "incremental";
            break;
        }
    }
}

void Radial_grid::set_radial_points(int num_points__, double* points__)
{
    assert(num_points__ > 0);

    points_.resize(num_points__);
    memcpy(&points_[0], points__, num_points__ * sizeof(double));

    // check if the last MT point is equal to the MT radius
    if (fabs(points_[num_mt_points_ - 1] - mt_radius_) > 1e-10) 
    {   
        std::stringstream s;
        s << "Wrong radial grid" << std::endl 
          << "  num_points     : " << num_points__ << std::endl
          << "  num_mt_points  : " << num_mt_points_  << std::endl
          << "  MT radius      : " << mt_radius_ << std::endl
          << "  MT point value : " << points_[num_mt_points_ - 1];
         
        error_local(__FILE__, __LINE__, s);
    }
    
    deltas_.resize(points_.size() - 1);
    for (int i = 0; i < (int)points_.size() - 1; i++) deltas_[i] = points_[i + 1] - points_[i];
    
    points_inv_.resize(points_.size());
    for (int i = 0; i < (int)points_.size(); i++) points_inv_[i] = (points_[i] == 0) ? 0 : 1.0 / points_[i];
}

void Radial_grid::print_info()
{
    printf("number of muffin-tin points : %i\n", num_mt_points());
    printf("total number of points      : %i\n", num_points());
    printf("starting point              : %f\n", points_[0]);
    printf("muffin-tin point            : %f\n", points_[num_mt_points() - 1]);
    printf("effective infinity point    : %f\n", points_[num_points() - 1]);
}

};
