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
    private:
    
        /// list of nearest neighbours for each atom
        std::vector< std::vector<nearest_neighbour_descriptor> > nearest_neighbours_;
        
        /*! 
            \brief Automatically determine new muffin-tin radii as a half distance between neighbor atoms.
                   
                   In order to guarantee a unique solution muffin-tin radii are dermined as a half distance
                   bethween closest atoms. Initial values of the muffin-tin radii (provided in the input file) 
                   are ignored.
        */
        void find_mt_radii()
        {
            if (nearest_neighbours_.size() == 0)                 
                error(__FILE__, __LINE__, "array of nearest neighbours is empty");

            // initialize Rmt to huge value
            std::vector<double> rmt(num_atom_types(), 1e10);
             
            for (int ia = 0; ia < num_atoms(); ia++)
            {
                int id1 = atom(ia)->type_id();
                if (nearest_neighbours_[ia].size() <= 1) // first atom is always the central one itself
                {
                    std::stringstream s;
                    s << "array of nearest neighbours for atom " << ia << " is empty";
                    error(__FILE__, __LINE__, s);
                }

                int ja = nearest_neighbours_[ia][1].atom_id;
                int id2 = atom(ja)->type_id();
                double dist = nearest_neighbours_[ia][1].distance;
                
                // take a little bit smaller value than half a distance
                double r = 0.95 * (dist / 2);

                // take minimal R for the given atom type
                rmt[atom_type_index_by_id(id1)] = std::min(r, rmt[atom_type_index_by_id(id1)]);
                rmt[atom_type_index_by_id(id2)] = std::min(r, rmt[atom_type_index_by_id(id2)]);
            }

            for (int i = 0; i < num_atom_types(); i++)
            {
                int id = atom_type(i)->id();
                atom_type_by_id(id)->set_mt_radius(std::min(rmt[i], 3.0));
            }
        }

        bool check_mt_overlap(bool stop_if_overlap)
        {
            if (nearest_neighbours_.size() == 0)
                error(__FILE__, __LINE__, "array of nearest neighbours is empty");

            for (int ia = 0; ia < num_atoms(); ia++)
            {
                if (nearest_neighbours_[ia].size() <= 1) // first atom is always the central one itself
                {
                    std::stringstream s;
                    s << "array of nearest neighbours for atom " << ia << " is empty";
                    error(__FILE__, __LINE__, s);
                }

                int ja = nearest_neighbours_[ia][1].atom_id;
                double dist = nearest_neighbours_[ia][1].distance;
                
                if ((atom(ia)->type()->mt_radius() + atom(ja)->type()->mt_radius()) > dist)
                {
                    if (stop_if_overlap)
                    {
                        std::stringstream s;
                        s << "overlaping muffin-tin spheres for atoms " << ia << " and " << ja << std::endl
                          << "  radius of atom " << ia << " : " << atom(ia)->type()->mt_radius() << std::endl
                          << "  radius of atom " << ja << " : " << atom(ja)->type()->mt_radius() << std::endl
                          << "  distance : " << dist;
                        error(__FILE__, __LINE__, s);
                    }
                    
                    return true;
                }
            }
            
            return false;
        }

    public:
        
        void init()
        {
            find_nearest_neighbours(15.0);
            
            if (check_mt_overlap(false))
                find_mt_radii();
            
            check_mt_overlap(true);
        }

        void find_nearest_neighbours(double cluster_radius)
        {
            Timer t("sirius::sirius_geometry::find_nearest_neighbours");

            int max_frac_coord[] = {0, 0, 0};
            double frac_coord[3];
            for (int i = 0; i < 3; i++)
            {
                double cart_coord[] = {0.0, 0.0, 0.0};
                cart_coord[i] = cluster_radius; // radius of nearest neighbours cluster
                get_coordinates<fractional, direct>(cart_coord, frac_coord);
                for (int i = 0; i < 3; i++)
                    max_frac_coord[i] = std::max(max_frac_coord[i], abs(int(frac_coord[i])) + 1);
            }
           
            nearest_neighbours_.clear();
            nearest_neighbours_.resize(num_atoms());
            for (int ia = 0; ia < num_atoms(); ia++)
            {
                double iapos[3];
                get_coordinates<cartesian, direct>(atom(ia)->position(), iapos);
                
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
                            get_coordinates<cartesian, direct>(nnd.translation, vt);
                            
                            for (int ja = 0; ja < num_atoms(); ja++)
                            {
                                nnd.atom_id = ja;

                                double japos[3];
                                get_coordinates<cartesian, direct>(atom(ja)->position(), japos);

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

        template <lattice_type Tl>
        void find_translation_limits(double radius, int* limits)
        {
            limits[0] = limits[1] = limits[2] = 0;

            int n = 0;
            while(true)
            {
                bool found = false;
                for (int i0 = -n; i0 <= n; i0++)
                    for (int i1 = -n; i1 <= n; i1++)
                        for (int i2 = -n; i2 <= n; i2++)
                            if (abs(i0) == n || abs(i1) == n || abs(i2) == n)
                            {
                                int vgf[] = {i0, i1, i2};
                                double vgc[3];
                                get_coordinates<cartesian, Tl>(vgf, vgc);
                                double len = vector_length(vgc);
                                if (len <= radius)
                                {
                                    found = true;
                                    for (int j = 0; j < 3; j++)
                                        limits[j] = std::max(2 * abs(vgf[j]) + 1, limits[j]);
                                }
                            }

                if (found) n++;
                else return;
            }
        }

        bool is_point_in_mt()
        {
            return true;
        }
};

};
