#include "atom_type.h"

namespace sirius {

void Atom_type::init_free_atom(bool smooth)
{
    /* check if atomic file exists */
    if (!Utils::file_exists(file_name_))
    {
        //== std::stringstream s;
        //== s << "file " + file_name_ + " doesn't exist";
        //== error_global(__FILE__, __LINE__, s);
        //std::stringstream s;
        //s << "Free atom density and potential for atom " << label_ << " are not initialized";
        //warning_global(__FILE__, __LINE__, s);
        return;
    }

    JSON_tree parser(file_name_);

    /* create free atom radial grid */
    std::vector<double> fa_r;
    parser["free_atom"]["radial_grid"] >> fa_r;
    free_atom_radial_grid_ = Radial_grid(fa_r);
    
    /* read density and potential */
    std::vector<double> v;
    parser["free_atom"]["density"] >> v;
    free_atom_density_ = Spline<double>(free_atom_radial_grid_, v);
    parser["free_atom"]["potential"] >> v;
    free_atom_potential_ = Spline<double>(free_atom_radial_grid_, v);

    /* smooth free atom density inside the muffin-tin sphere */
    if (smooth)
    {
        /* find point on the grid close to the muffin-tin radius */
        int irmt = idx_rmt_free_atom();
    
        mdarray<double, 1> b(2);
        mdarray<double, 2> A(2, 2);
        double R = free_atom_radial_grid_[irmt];
        A(0, 0) = std::pow(R, 2);
        A(0, 1) = std::pow(R, 3);
        A(1, 0) = 2 * R;
        A(1, 1) = 3 * std::pow(R, 2);
        
        b(0) = free_atom_density_[irmt];
        b(1) = free_atom_density_.deriv(1, irmt);

        linalg<CPU>::gesv<double>(2, 1, A.at<CPU>(), 2, b.at<CPU>(), 2);
       
        //== /* write initial density */
        //== std::stringstream sstr;
        //== sstr << "free_density_" << id_ << ".dat";
        //== FILE* fout = fopen(sstr.str().c_str(), "w");

        //== for (int ir = 0; ir < free_atom_radial_grid().num_points(); ir++)
        //== {
        //==     fprintf(fout, "%f %f \n", free_atom_radial_grid(ir), free_atom_density_[ir]);
        //== }
        //== fclose(fout);
        
        /* make smooth free atom density inside muffin-tin */
        for (int i = 0; i <= irmt; i++)
        {
            free_atom_density_[i] = b(0) * std::pow(free_atom_radial_grid(i), 2) + 
                                    b(1) * std::pow(free_atom_radial_grid(i), 3);
        }

        /* interpolate new smooth density */
        free_atom_density_.interpolate();

        //== /* write smoothed density */
        //== sstr.str("");
        //== sstr << "free_density_modified_" << id_ << ".dat";
        //== fout = fopen(sstr.str().c_str(), "w");

        //== for (int ir = 0; ir < free_atom_radial_grid().num_points(); ir++)
        //== {
        //==     fprintf(fout, "%f %f \n", free_atom_radial_grid(ir), free_atom_density_[ir]);
        //== }
        //== fclose(fout);
   }
}

}
