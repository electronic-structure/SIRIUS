template <lattice_t Tl>
void Unit_cell::find_translation_limits(double radius, int* limits)
{
    limits[0] = limits[1] = limits[2] = 0;

    int n = 0;
    while(true)
    {
        bool found = false;
        for (int i0 = -n; i0 <= n; i0++)
        {
            for (int i1 = -n; i1 <= n; i1++)
            {
                for (int i2 = -n; i2 <= n; i2++)
                {
                    if (abs(i0) == n || abs(i1) == n || abs(i2) == n)
                    {
                        int vgf[] = {i0, i1, i2};
                        double vgc[3];
                        get_coordinates<cartesian, Tl>(vgf, vgc);
                        double len = Utils::vector_length(vgc);
                        if (len <= radius)
                        {
                            found = true;
                            for (int j = 0; j < 3; j++) limits[j] = std::max(2 * abs(vgf[j]) + 1, limits[j]);
                        }
                    }
                }
            }
        }

        if (found) 
        {
            n++;
        }
        else 
        {
            return;
        }
    }
}

template <lattice_t Tl>
void Unit_cell::reduce_coordinates(double vc[3], int ntr[3], double vf[3])
{
    get_coordinates<fractional, Tl>(vc, vf);
    for (int i = 0; i < 3; i++)
    {
        ntr[i] = (int)floor(vf[i]);
        vf[i] -= ntr[i];
        if (vf[i] < 0.0 || vf[i] >= 1.0) error(__FILE__, __LINE__, "wrong fractional coordinates");
    }
}

template<coordinates_t cT, lattice_t lT, typename T>
void Unit_cell::get_coordinates(T* a, double* b)
{
    b[0] = b[1] = b[2] = 0.0;
    
    if (lT == direct)
    {
        if (cT == fractional)
        {    
            for (int l = 0; l < 3; l++)
            {
                for (int x = 0; x < 3; x++) b[l] += a[x] * inverse_lattice_vectors_[x][l];
            }
        }

        if (cT == cartesian)
        {
            for (int x = 0; x < 3; x++)
            {
                for (int l = 0; l < 3; l++) b[x] += a[l] * lattice_vectors_[l][x];
            }
        }
    }

    if (lT == reciprocal)
    {
        if (cT == fractional)
        {
            for (int l = 0; l < 3; l++)
            {
                for (int x = 0; x < 3; x++) b[l] += lattice_vectors_[l][x] * a[x] / twopi;
            }
        }

        if (cT == cartesian)
        {
            for (int x = 0; x < 3; x++)
            {
                for (int l = 0; l < 3; l++) b[x] += a[l] * reciprocal_lattice_vectors_[l][x];
            }
        }
    }
}

void Unit_cell::add_atom_type(int atom_type_id, const std::string label)
{
    if (atom_type_index_by_id_.count(atom_type_id) != 0) 
    {   
        std::stringstream s;
        s << "atom type with id " << atom_type_id << " is already in list";
        error(__FILE__, __LINE__, s);
    }
    atom_types_.push_back(new Atom_type(atom_type_id, label));
    atom_type_index_by_id_[atom_type_id] = (int)atom_types_.size() - 1;
}

void Unit_cell::add_atom(int atom_type_id, double* position, double* vector_field)
{
    double eps = 1e-10;
    double pos[3];
    
    if (atom_type_index_by_id_.count(atom_type_id) == 0)
    {
        std::stringstream s;
        s << "atom type with id " << atom_type_id << " is not found";
        error(__FILE__, __LINE__, s);
    }

    for (int i = 0; i < num_atoms(); i++)
    {
        atom(i)->get_position(pos);
        if (fabs(pos[0] - position[0]) < eps && fabs(pos[1] - position[1]) < eps && fabs(pos[2] - position[2]) < eps)
        {
            std::stringstream s;
            s << "atom with the same position is already in list" << std::endl
              << "  position : " << position[0] << " " << position[1] << " " << position[2];
            
            error(__FILE__, __LINE__, s);
        }
    }

    atoms_.push_back(new Atom(atom_type_by_id(atom_type_id), position, vector_field));
}

void Unit_cell::get_symmetry()
{
    Timer t("sirius::UnitCell::get_symmetry");
    
    if (num_atoms() == 0) error(__FILE__, __LINE__, "no atoms");
    
    if (spg_dataset_) spg_free_dataset(spg_dataset_);
        
    if (atom_symmetry_classes_.size() != 0)
    {
        for (int ic = 0; ic < (int)atom_symmetry_classes_.size(); ic++) delete atom_symmetry_classes_[ic];
        atom_symmetry_classes_.clear();

        for (int ia = 0; ia < num_atoms(); ia++) atom(ia)->set_symmetry_class(NULL);
    }

    double lattice[3][3];

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++) lattice[i][j] = lattice_vectors_[j][i];
    }

    mdarray<double, 2> positions(3, num_atoms());
    
    std::vector<int> types(num_atoms());

    for (int i = 0; i < num_atoms(); i++)
    {
        atoms_[i]->get_position(&positions(0, i));
        types[i] = atoms_[i]->type_id();
    }
    spg_dataset_ = spg_get_dataset(lattice, (double(*)[3])&positions(0, 0), &types[0], num_atoms(), 1e-5);

    if (spg_dataset_->spacegroup_number == 0)
        error(__FILE__, __LINE__, "spg_get_dataset() returned 0 for the space group");

    if (spg_dataset_->n_atoms != num_atoms()) error(__FILE__, __LINE__, "wrong number of atoms");

    Atom_symmetry_class* atom_symmetry_class;
    
    int atom_class_id = -1;

    for (int i = 0; i < num_atoms(); i++)
    {
        if (!atoms_[i]->symmetry_class()) // if symmetry class is not assigned to this atom
        {
            atom_class_id++; // take next id 
            atom_symmetry_class = new Atom_symmetry_class(atom_class_id, atoms_[i]->type());
            atom_symmetry_classes_.push_back(atom_symmetry_class);

            for (int j = 0; j < num_atoms(); j++) // scan all atoms
            {
                bool is_equal = (equivalent_atoms_.size()) ? (equivalent_atoms_[j] == equivalent_atoms_[i]) :  
                                (spg_dataset_->equivalent_atoms[j] == spg_dataset_->equivalent_atoms[i]);
                
                if (is_equal) // assign new class id for all equivalent atoms
                {
                    atom_symmetry_class->add_atom_id(j);
                    atoms_[j]->set_symmetry_class(atom_symmetry_class);
                }
            }
        }
    }
}

void Unit_cell::find_mt_radii()
{
    // TODO: update documentation

    if (nearest_neighbours_.size() == 0) error(__FILE__, __LINE__, "array of nearest neighbours is empty");

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
        double R = 0.95 * (dist / 2);

        // take minimal R for the given atom type
        rmt[atom_type_index_by_id(id1)] = std::min(R, rmt[atom_type_index_by_id(id1)]);
        rmt[atom_type_index_by_id(id2)] = std::min(R, rmt[atom_type_index_by_id(id2)]);
    }

    std::vector<bool> can_scale(num_atom_types(), true);
    for (int ia = 0; ia < num_atoms(); ia++)
    {
        int id1 = atom(ia)->type_id();
        int ja = nearest_neighbours_[ia][1].atom_id;
        int id2 = atom(ja)->type_id();
        double dist = nearest_neighbours_[ia][1].distance;
        
        if (rmt[atom_type_index_by_id(id1)] + rmt[atom_type_index_by_id(id2)] > dist * 0.94)
        {
            can_scale[atom_type_index_by_id(id1)] = false;
            can_scale[atom_type_index_by_id(id2)] = false;
        }
    }

    for (int ia = 0; ia < num_atoms(); ia++)
    {
        int id1 = atom(ia)->type_id();
        int ja = nearest_neighbours_[ia][1].atom_id;
        int id2 = atom(ja)->type_id();
        double dist = nearest_neighbours_[ia][1].distance;
        
        if (can_scale[atom_type_index_by_id(id1)])
            rmt[atom_type_index_by_id(id1)] = 0.95 * (dist - rmt[atom_type_index_by_id(id2)]);
    }
    
    for (int i = 0; i < num_atom_types(); i++)
    {
        int id = atom_type(i)->id();
        atom_type_by_id(id)->set_mt_radius(std::min(rmt[i], 3.0));
        atom_type_by_id(id)->init_radial_grid();
    }
}

bool Unit_cell::check_mt_overlap(int& ia__, int& ja__)
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
            ia__ = ia;
            ja__ = ja;
            return true;
        }
    }
    
    return false;
}

void Unit_cell::init(int lmax_apw, int lmax_pot, int num_mag_dims, int init_radial_grid__, int init_aw_descriptors__)
{
    // =====================
    // initialize atom types
    // =====================
    for (int i = 0; i < num_atom_types(); i++)
    {
         if (init_radial_grid__) atom_type(i)->init_radial_grid();
         if (init_aw_descriptors__) atom_type(i)->init_aw_descriptors(lmax_apw);
         atom_type(i)->init(lmax_apw);
    }
    
    // ================
    // find the charges
    // ================
    total_nuclear_charge_ = 0;
    num_core_electrons_ = 0;
    num_valence_electrons_ = 0;
    for (int i = 0; i < num_atoms(); i++)
    {
        total_nuclear_charge_ += atom(i)->type()->zn();
        num_core_electrons_ += atom(i)->type()->num_core_electrons();
        num_valence_electrons_ += atom(i)->type()->num_valence_electrons();
    }
    num_electrons_ = num_core_electrons_ + num_valence_electrons_;
   
    // ================
    // initialize atoms
    // ================
    mt_basis_size_ = 0;
    mt_aw_basis_size_ = 0;
    mt_lo_basis_size_ = 0;
    max_mt_basis_size_ = 0;
    max_mt_radial_basis_size_ = 0;
    max_mt_aw_basis_size_ = 0;
    for (int ia = 0; ia < num_atoms(); ia++)
    {
        atom(ia)->init(lmax_pot, num_mag_dims, mt_aw_basis_size_, mt_lo_basis_size_, mt_basis_size_);
        mt_aw_basis_size_ += atom(ia)->type()->mt_aw_basis_size();
        mt_lo_basis_size_ += atom(ia)->type()->mt_lo_basis_size();
        mt_basis_size_ += atom(ia)->type()->mt_basis_size();
        max_mt_basis_size_ = std::max(max_mt_basis_size_, atom(ia)->type()->mt_basis_size());
        max_mt_radial_basis_size_ = std::max(max_mt_radial_basis_size_, atom(ia)->type()->mt_radial_basis_size());
        max_mt_aw_basis_size_ = std::max(max_mt_aw_basis_size_, atom(ia)->type()->mt_aw_basis_size());
    }

    assert(mt_basis_size_ == mt_aw_basis_size_ + mt_lo_basis_size_);
    assert(num_atoms() != 0);
    assert(num_atom_types() != 0);

    update_symmetry();
    assert(num_atom_symmetry_classes() != 0);
}

void Unit_cell::update_symmetry()
{
    find_nearest_neighbours(25.0);
    
    if (auto_rmt() != 0) find_mt_radii();

    int ia, ja;
    if (check_mt_overlap(ia, ja))
    {
        std::stringstream s;
        s << "overlaping muffin-tin spheres for atoms " << ia << " and " << ja << std::endl
          << "  radius of atom " << ia << " : " << atom(ia)->type()->mt_radius() << std::endl
          << "  radius of atom " << ja << " : " << atom(ja)->type()->mt_radius();
        error(__FILE__, __LINE__, s, fatal_err);
    }
    
    get_symmetry();

    for (int ic = 0; ic < num_atom_symmetry_classes(); ic++) atom_symmetry_class(ic)->initialize();
    
    max_num_mt_points_ = 0;
    min_mt_radius_ = 1e100;
    max_mt_radius_ = 0;
    for (int i = 0; i < num_atom_types(); i++)
    {
         max_num_mt_points_ = std::max(max_num_mt_points_, atom_type(i)->num_mt_points());
         min_mt_radius_ = std::min(min_mt_radius_, atom_type(i)->mt_radius());
         max_mt_radius_ = std::max(max_mt_radius_, atom_type(i)->mt_radius());
    }
}

void Unit_cell::clear()
{
    if (spg_dataset_)
    {
        spg_free_dataset(spg_dataset_);
        spg_dataset_ = NULL;
    }
    
    // delete atom types
    for (int i = 0; i < (int)atom_types_.size(); i++) delete atom_types_[i];
    atom_types_.clear();
    atom_type_index_by_id_.clear();

    // delete atom classes
    for (int i = 0; i < (int)atom_symmetry_classes_.size(); i++) delete atom_symmetry_classes_[i];
    atom_symmetry_classes_.clear();

    // delete atoms
    for (int i = 0; i < num_atoms(); i++) delete atoms_[i];
    atoms_.clear();

    equivalent_atoms_.clear();
}

void Unit_cell::print_info()
{
    printf("\n");
    printf("Unit cell\n");
    for (int i = 0; i < 80; i++) printf("-");
    printf("\n");
    
    printf("lattice vectors\n");
    for (int i = 0; i < 3; i++)
    {
        printf("  a%1i : %18.10f %18.10f %18.10f \n", i + 1, lattice_vectors(i, 0), 
                                                             lattice_vectors(i, 1), 
                                                             lattice_vectors(i, 2)); 
    }
    printf("reciprocal lattice vectors\n");
    for (int i = 0; i < 3; i++)
    {
        printf("  b%1i : %18.10f %18.10f %18.10f \n", i + 1, reciprocal_lattice_vectors(i, 0), 
                                                             reciprocal_lattice_vectors(i, 1), 
                                                             reciprocal_lattice_vectors(i, 2));
    }
    printf("\n");
    printf("unit cell volume : %18.8f [a.u.^3]\n", omega());
    printf("1/sqrt(omega)    : %18.8f\n", 1.0 / sqrt(omega()));
    
    printf("\n"); 
    printf("number of atom types : %i\n", num_atom_types());
    for (int i = 0; i < num_atom_types(); i++)
    {
        int id = atom_type(i)->id();
        printf("type id : %i   symbol : %2s   label : %2s   mt_radius : %10.6f\n", id,
                                                                                   atom_type(i)->symbol().c_str(), 
                                                                                   atom_type(i)->label().c_str(),
                                                                                   atom_type(i)->mt_radius()); 
    }

    printf("number of atoms : %i\n", num_atoms());
    printf("number of symmetry classes : %i\n", num_atom_symmetry_classes());
    printf("\n"); 
    printf("atom id              position            type id    class id\n");
    printf("------------------------------------------------------------\n");
    for (int i = 0; i < num_atoms(); i++)
    {
        printf("%6i      %f %f %f   %6i      %6i\n", i, atom(i)->position(0), 
                                                        atom(i)->position(1), 
                                                        atom(i)->position(2), 
                                                        atom(i)->type_id(),
                                                        atom(i)->symmetry_class_id());
    }
   
    printf("\n");
    for (int ic = 0; ic < num_atom_symmetry_classes(); ic++)
    {
        printf("class id : %i   atom id : ", ic);
        for (int i = 0; i < atom_symmetry_class(ic)->num_atoms(); i++)
            printf("%i ", atom_symmetry_class(ic)->atom_id(i));  
        printf("\n");
    }

    printf("\n");
    printf("space group number   : %i\n", spg_dataset_->spacegroup_number);
    printf("international symbol : %s\n", spg_dataset_->international_symbol);
    printf("Hall symbol          : %s\n", spg_dataset_->hall_symbol);
    printf("number of operations : %i\n", spg_dataset_->n_operations);
    
    printf("\n");
    printf("total nuclear charge        : %i\n", total_nuclear_charge_);
    printf("number of core electrons    : %f\n", num_core_electrons_);
    printf("number of valence electrons : %f\n", num_valence_electrons_);
    printf("total number of electrons   : %f\n", num_electrons_);
}

void Unit_cell::write_cif()
{
    if (Platform::mpi_rank() == 0)
    {
        FILE* fout = fopen("unit_cell.cif", "w");

        double a = Utils::vector_length(lattice_vectors_[0]);
        double b = Utils::vector_length(lattice_vectors_[1]);
        double c = Utils::vector_length(lattice_vectors_[2]);
        fprintf(fout, "_cell_length_a %f\n", a);
        fprintf(fout, "_cell_length_b %f\n", b);
        fprintf(fout, "_cell_length_c %f\n", c);

        double alpha = acos(Utils::scalar_product(lattice_vectors_[1], lattice_vectors_[2]) / b / c) * 180 / pi;
        double beta = acos(Utils::scalar_product(lattice_vectors_[0], lattice_vectors_[2]) / a / c) * 180 / pi;
        double gamma = acos(Utils::scalar_product(lattice_vectors_[0], lattice_vectors_[1]) / a / b) * 180 / pi;
        fprintf(fout, "_cell_angle_alpha %f\n", alpha);
        fprintf(fout, "_cell_angle_beta %f\n", beta);
        fprintf(fout, "_cell_angle_gamma %f\n", gamma);

        //fprintf(fout, "loop_\n");
        //fprintf(fout, "_symmetry_equiv_pos_as_xyz\n");

        fprintf(fout, "loop_\n");
        fprintf(fout, "_atom_site_label\n");
        fprintf(fout, "_atom_type_symbol\n");
        fprintf(fout, "_atom_site_fract_x\n");
        fprintf(fout, "_atom_site_fract_y\n");
        fprintf(fout, "_atom_site_fract_z\n");
        for (int ia = 0; ia < num_atoms(); ia++)
        {
            std::stringstream s;
            s << ia + 1 << " " << atom(ia)->type()->symbol() << " " << atom(ia)->position(0) << " " << 
                 atom(ia)->position(1) << " " << atom(ia)->position(2);
            fprintf(fout,"%s\n",s.str().c_str());
        }
        fclose(fout);
    }
}

void Unit_cell::set_lattice_vectors(double* a1, double* a2, double* a3)
{
    for (int x = 0; x < 3; x++)
    {
        lattice_vectors_[0][x] = a1[x];
        lattice_vectors_[1][x] = a2[x];
        lattice_vectors_[2][x] = a3[x];
    }
    double a[3][3];
    memcpy(&a[0][0], &lattice_vectors_[0][0], 9 * sizeof(double));
    
    double t1 = a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]) + 
                a[0][1] * (a[1][2] * a[2][0] - a[1][0] * a[2][2]) + 
                a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]);
    
    omega_ = fabs(t1);
    
    if (omega_ < 1e-10) error(__FILE__, __LINE__, "lattice vectors are linearly dependent");
    
    t1 = 1.0 / t1;

    double b[3][3];
    b[0][0] = t1 * (a[1][1] * a[2][2] - a[1][2] * a[2][1]);
    b[0][1] = t1 * (a[0][2] * a[2][1] - a[0][1] * a[2][2]);
    b[0][2] = t1 * (a[0][1] * a[1][2] - a[0][2] * a[1][1]);
    b[1][0] = t1 * (a[1][2] * a[2][0] - a[1][0] * a[2][2]);
    b[1][1] = t1 * (a[0][0] * a[2][2] - a[0][2] * a[2][0]);
    b[1][2] = t1 * (a[0][2] * a[1][0] - a[0][0] * a[1][2]);
    b[2][0] = t1 * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    b[2][1] = t1 * (a[0][1] * a[2][0] - a[0][0] * a[2][1]);
    b[2][2] = t1 * (a[0][0] * a[1][1] - a[0][1] * a[1][0]);

    memcpy(&inverse_lattice_vectors_[0][0], &b[0][0], 9 * sizeof(double));

    for (int l = 0; l < 3; l++)
    {
        for (int x = 0; x < 3; x++) 
            reciprocal_lattice_vectors_[l][x] = twopi * inverse_lattice_vectors_[x][l];
    }
}

void Unit_cell::find_nearest_neighbours(double cluster_radius)
{
    Timer t("sirius::Unit_cell::find_nearest_neighbours");

    int max_frac_coord[3];
    find_translation_limits<direct>(cluster_radius, max_frac_coord);
   
    nearest_neighbours_.clear();
    nearest_neighbours_.resize(num_atoms());

    #pragma omp parallel for default(shared)
    for (int ia = 0; ia < num_atoms(); ia++)
    {
        double iapos[3];
        get_coordinates<cartesian, direct>(atom(ia)->position(), iapos);
        
        std::vector<nearest_neighbour_descriptor> nn;

        std::vector< std::pair<double, int> > nn_sort;

        for (int i0 = -max_frac_coord[0]; i0 <= max_frac_coord[0]; i0++)
        {
            for (int i1 = -max_frac_coord[1]; i1 <= max_frac_coord[1]; i1++)
            {
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
                        for (int x = 0; x < 3; x++) v[x] = japos[x] + vt[x] - iapos[x];

                        nnd.distance = Utils::vector_length(v);
                        
                        if (nnd.distance <= cluster_radius)
                        {
                            nn.push_back(nnd);

                            nn_sort.push_back(std::pair<double, int>(nnd.distance, (int)nn.size() - 1));
                        }
                    }

                }
            }
        }
        
        std::sort(nn_sort.begin(), nn_sort.end());
        nearest_neighbours_[ia].resize(nn.size());
        for (int i = 0; i < (int)nn.size(); i++) nearest_neighbours_[ia][i] = nn[nn_sort[i].second];
    }

    if (Platform::mpi_rank() == 0)
    {
        FILE* fout = fopen("nghbr.txt", "w");
        for (int ia = 0; ia < num_atoms(); ia++)
        {
            fprintf(fout, "Central atom: %s (%i)\n", atom(ia)->type()->label().c_str(), ia);
            for (int i = 0; i < 80; i++) fprintf(fout, "-");
            fprintf(fout, "\n");
            fprintf(fout, "atom (  id)       D [a.u.]    translation  R\n");
            for (int i = 0; i < 80; i++) fprintf(fout, "-");
            fprintf(fout, "\n");
            for (int i = 0; i < (int)nearest_neighbours_[ia].size(); i++)
            {
                int ja = nearest_neighbours_[ia][i].atom_id;
                fprintf(fout, "%4s (%4i)   %12.6f\n", atom(ja)->type()->label().c_str(), ja, 
                                         nearest_neighbours_[ia][i].distance);
            }
            fprintf(fout, "\n");
        }
        fclose(fout);
    }
}

bool Unit_cell::is_point_in_mt(double vc[3], int& ja, int& jr, double& dr, double tp[2])
{
    int ntr[3];
    double vf[3];
    
    // reduce coordinates to the primitive unit cell
    reduce_coordinates<direct>(vc, ntr, vf);

    for (int ia = 0; ia < num_atoms(); ia++)
    {
        for (int i0 = -1; i0 <= 1; i0++)
        {
            for (int i1 = -1; i1 <= 1; i1++)
            {
                for (int i2 = -1; i2 <= 1; i2++)
                {
                    // atom position
                    double posf[] = {double(i0), double(i1), double(i2)};
                    for (int i = 0; i < 3; i++) posf[i] += atom(ia)->position(i);
                    
                    // vector connecting center of atom and reduced point
                    double vf1[3];
                    for (int i = 0; i < 3; i++) vf1[i] = vf[i] - posf[i];
                    
                    // convert to Cartesian coordinates
                    double vc1[3];
                    get_coordinates<cartesian, direct>(vf1, vc1);

                    double r = Utils::vector_length(vc1);

                    if (r <= atom(ia)->type()->mt_radius())
                    {
                        ja = ia;
                        double vs1[3];                       
                        SHT::spherical_coordinates(vc1, vs1);
                        tp[0] = vs1[1]; // theta
                        tp[1] = vs1[2]; // phi

                        if (r < atom(ia)->type()->radial_grid(0))
                        {
                            jr = 0;
                            dr = 0.0;
                        }
                        else
                        {
                            for (int ir = 0; ir < atom(ia)->num_mt_points() - 1; ir++)
                            {
                                if ((r >= atom(ia)->type()->radial_grid(ir)) && 
                                    (r <= atom(ia)->type()->radial_grid(ir + 1)))
                                {
                                    jr = ir;
                                    dr = r - atom(ia)->type()->radial_grid(ir);
                                    break;
                                }
                            }
                        }

                        return true;
                    }

                }
            }
        }
    }
    
    return false;
}

