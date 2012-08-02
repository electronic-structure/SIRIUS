namespace sirius {

struct nearest_neighbour_descriptor
{
    /// id of neighbour atom
    int atom_id;

    /// translation along each lattice vector
    int translation[3];

    /// distance from the central atom
    double distance;
};

class sirius_geometry : public sirius_unit_cell
{
    protected:
    
        /// list of nearest neighbours for each atom
        std::vector< std::vector<nearest_neighbour_descriptor> > nearest_neighbours_;

    public:
    
        void find_nearest_neighbours()
        {
            Timer t("find_nearest_neighbours");

            int max_frac_coord[] = {0, 0, 0};
            double frac_coord[3];
            for (int i = 0; i < 3; i++)
            {
                double cart_coord[] = {0.0, 0.0, 0.0};
                cart_coord[i] = 15.0; // radius of nearest neighbours cluster
                get_fractional_coordinates(cart_coord, frac_coord);
                for (int i = 0; i < 3; i++)
                    max_frac_coord[i] = std::max(max_frac_coord[i], abs(int(frac_coord[i])) + 1);
            }
           
            nearest_neighbours_.resize(atoms_.size());
            for (int ia = 0; ia < (int)atoms_.size(); ia++)
            {
                double iapos[3];
                get_cartesian_coordinates(atoms_[ia]->position(), iapos);
                
                std::vector<nearest_neighbour_descriptor> nn;
                std::vector<double> dist;

                for (int i0 = -max_frac_coord[0]; i0 <= max_frac_coord[0]; i0++)
                    for (int i1 = -max_frac_coord[1]; i1 <= max_frac_coord[1]; i1++)
                        for (int i2 = -max_frac_coord[2]; i2 <= max_frac_coord[2]; i2++)
                        {
                            nearest_neighbour_descriptor nnd;
                            nnd.translation[0] = i0;
                            nnd.translation[1] = i1;
                            nnd.translation[2] = i2;
                            
                            double vt[3];
                            get_cartesian_coordinates(nnd.translation, vt);
                            
                            for (int ja = 0; ja < (int)atoms_.size(); ja++)
                            {
                                nnd.atom_id = ja;

                                double japos[3];
                                get_cartesian_coordinates(atoms_[ja]->position(), japos);

                                double v[3];
                                for (int x = 0; x < 3; x++)
                                    v[x] = japos[x] + vt[x] - iapos[x];

                                nnd.distance = vector_length(v);
                                
                                dist.push_back(nnd.distance);
                                nn.push_back(nnd);
                            }

                        }
                
                std::vector<size_t> reorder(dist.size());
                gsl_heapsort_index(&reorder[0], &dist[0], dist.size(), sizeof(double), compare_doubles);
                nearest_neighbours_[ia].resize(nn.size());
                for (int i = 0; i < (int)nn.size(); i++)
                    nearest_neighbours_[ia][i] = nn[reorder[i]];
            }
        }

        void find_mt_radii()
        {
            std::map<int,double> dr_by_type_id;
            
            std::map<int,AtomType*>::iterator it;
            
            // initialize delta R to huge value
            for (it = atom_type_by_id_.begin(); it != atom_type_by_id_.end(); it++)
            {
                if (dr_by_type_id.count((*it).first) == 0)
                    dr_by_type_id[(*it).first] = 1e10;
            }
             
            for (int ia = 0; ia < (int)atoms_.size(); ia++)
            {
                int ja = nearest_neighbours_[ia][1].atom_id;
                double dist = nearest_neighbours_[ia][1].distance;
                
                // take 95% of remaining distance and assign equal halves to both atoms
                double dr = 0.95 * (dist - atoms_[ia]->type()->mt_radius() - atoms_[ja]->type()->mt_radius()) / 2;
                
                if (dr < 0.0)
                    error(__FILE__, __LINE__, "muffin-tin spheres overlap");
                
                // take minimal delta R for the given atom type
                dr_by_type_id[atoms_[ia]->type_id()] = std::min(dr, dr_by_type_id[atoms_[ia]->type_id()]);
                dr_by_type_id[atoms_[ja]->type_id()] = std::min(dr, dr_by_type_id[atoms_[ja]->type_id()]);
            }
            
            for (int i = 0; i < (int)atoms_.size(); i++)
            {
                std::cout << "atom : " << i << " new r : " << atoms_[i]->type()->mt_radius() + dr_by_type_id[atoms_[i]->type_id()] << std::endl;
            }
        }
};

};
