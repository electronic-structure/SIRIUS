// Copyright (c) 2013-2014 Anton Kozhevnikov, Thomas Schulthess
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that 
// the following conditions are met:
// 
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the 
//    following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions 
//    and the following disclaimer in the documentation and/or other materials provided with the distribution.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED 
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR 
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/** \file unit_cell.cpp
 *   
 *  \brief Contains remaining implementation of sirius::Unit_cell class.
 */

#include "unit_cell.h"

namespace sirius {

int Unit_cell::next_atom_type_id(const std::string label)
{
    /* check if the label was already added */
    if (atom_type_id_map_.count(label) != 0) 
    {   
        std::stringstream s;
        s << "atom type with label " << label << " is already in list";
        error_local(__FILE__, __LINE__, s);
    }
    /* take text id */
    atom_type_id_map_[label] = (int)atom_types_.size();
    return atom_type_id_map_[label];
}

void Unit_cell::add_atom_type(const std::string label, const std::string file_name, 
                              electronic_structure_method_t esm_type)
{
    int id = next_atom_type_id(label);
    atom_types_.push_back(new Atom_type(id, label, file_name, esm_type));
}

void Unit_cell::add_atom(const std::string label, double* position, double* vector_field)
{
    if (atom_type_id_map_.count(label) == 0)
    {
        std::stringstream s;
        s << "atom type with label " << label << " is not found";
        error_local(__FILE__, __LINE__, s);
    }
    if (atom_id_by_position(position) >= 0)
    {
        std::stringstream s;
        s << "atom with the same position is already in list" << std::endl
          << "  position : " << position[0] << " " << position[1] << " " << position[2];
        
        error_local(__FILE__, __LINE__, s);
    }

    atoms_.push_back(new Atom(atom_type(label), position, vector_field));
    atom_type(label)->add_atom_id((int)atoms_.size() - 1);
}

void Unit_cell::add_atom(const std::string label, double* position)
{
    double vector_field[] = {0, 0, 0};
    add_atom(label, position, vector_field);
}

void Unit_cell::get_symmetry()
{
    Timer t("sirius::Unit_cell::get_symmetry");
    
    if (num_atoms() == 0) return;
    
    if (atom_symmetry_classes_.size() != 0)
    {
        for (int ic = 0; ic < (int)atom_symmetry_classes_.size(); ic++) delete atom_symmetry_classes_[ic];
        atom_symmetry_classes_.clear();

        for (int ia = 0; ia < num_atoms(); ia++) atom(ia)->set_symmetry_class(NULL);
    }

    if (symmetry_ != nullptr)
    {
        TERMINATE("Symmetry() object is already allocated");
    }

    mdarray<double, 2> positions(3, num_atoms());
    std::vector<int> types(num_atoms());
    for (int ia = 0; ia < num_atoms(); ia++)
    {
        for (int x = 0; x < 3; x++) positions(x, ia) = atom(ia)->position(x);
        types[ia] = atom(ia)->type_id();
    }
    
    symmetry_ = new Symmetry(lattice_vectors_, num_atoms(), positions, types, 1e-4);





    //== for (int isym = 0; isym < s.num_sym_op(); isym++)
    //== {
    //==     std::cout << std::endl;
    //==     std::cout << "symmetry operation : " << isym << std::endl;

    //==     vector3d<double> ang = s.euler_angles(isym);

    //==     std::cout << "Euler angles : " << ang[0] / pi << " " << ang[1] / pi << " " << ang[2] / pi << std::endl;

    //==     //ang[0] = -ang[0];
    //==     //ang[1] = -ang[1];
    //==     //ang[2] = -ang[2];
    //==     int proper_rotation = s.proper_rotation(isym);

    //==     
    //==     vector3d<double> coord(double(rand()) / RAND_MAX, double(rand()) / RAND_MAX, double(rand()) / RAND_MAX);
    //==     vector3d<double> scoord;
    //==     SHT::spherical_coordinates(coord, &scoord[0]);

    //==     int lmax = 10;
    //==     mdarray<double_complex, 1> ylm(Utils::lmmax(lmax));
    //==     SHT::spherical_harmonics(lmax, scoord[1], scoord[2], &ylm(0));

    //==     matrix3d<double> rotm = inverse(s.rot_mtrx(isym));
    //==     printf("3x3 rotation matrix\n");
    //==     for (int i = 0; i < 3; i++)
    //==     {
    //==         for (int j = 0; j < 3; j++) printf("%8.4f ", rotm(i, j));
    //==         printf("\n");
    //==     }

    //==     vector3d<double> coord2;
    //==     for (int i = 0; i < 3; i++)
    //==     {
    //==         for (int j = 0; j < 3; j++) coord2[i] += rotm(i, j) * coord[j];
    //==     }

    //==     //std::cout << "initial coordinates : " << coord[0] << " " << coord[1] << " " << coord[2] << std::endl;
    //==     //std::cout << "rotated coordinates : " << coord2[0] << " " << coord2[1] << " " << coord2[2] << std::endl;

    //==     vector3d<double> scoord2;
    //==     SHT::spherical_coordinates(coord2, &scoord2[0]);

    //==     mdarray<double_complex, 1> ylm2(Utils::lmmax(lmax));
    //==     SHT::spherical_harmonics(lmax, scoord2[1], scoord2[2], &ylm2(0));

    //==     mdarray<double_complex, 2> ylm_rot_mtrx(Utils::lmmax(lmax), Utils::lmmax(lmax));
    //==     SHT::rotation_matrix(lmax, ang, proper_rotation, ylm_rot_mtrx); 

    //==     //== printf("Ylm rotation matrix\n");
    //==     //== for (int i = 0; i < Utils::lmmax(lmax); i++)
    //==     //== {
    //==     //==     for (int j = 0; j < Utils::lmmax(lmax); j++) 
    //==     //==         printf("(%12.6f, %12.6f)  ", real(ylm_rot_mtrx(i, j)), imag(ylm_rot_mtrx(i, j)));
    //==     //==     printf("\n");
    //==     //== }
    //==         





    //==     mdarray<double_complex, 1> ylm1(Utils::lmmax(lmax));
    //==     ylm1.zero();

    //==     for (int i = 0; i < Utils::lmmax(lmax); i++)
    //==     {
    //==         for (int j = 0; j < Utils::lmmax(lmax); j++) ylm1(i) += ylm_rot_mtrx(j, i) * ylm(j);
    //==     }

    //==     double d = 0;
    //==     for (int i = 0; i < Utils::lmmax(lmax); i++) d += abs(ylm1(i) - ylm2(i));

    //==     std::cout << "symmetry op : " << isym << ", defference of Ylm : " << d << ", proper_rotation : " << proper_rotation << std::endl;


    //==         
    //==             


















    //== }

    Atom_symmetry_class* atom_symmetry_class;
    
    int atom_class_id = -1;

    for (int i = 0; i < num_atoms(); i++)
    {
        /* if symmetry class is not assigned to this atom */
        if (!atoms_[i]->symmetry_class())
        {
            /* take next id */
            atom_class_id++;
            atom_symmetry_class = new Atom_symmetry_class(atom_class_id, atoms_[i]->type());
            atom_symmetry_classes_.push_back(atom_symmetry_class);

            for (int j = 0; j < num_atoms(); j++) // scan all atoms
            {
                bool is_equal = (equivalent_atoms_.size()) ? (equivalent_atoms_[j] == equivalent_atoms_[i]) :  
                                (symmetry_->atom_symmetry_class(j) == symmetry_->atom_symmetry_class(i));
                
                if (is_equal) // assign new class id for all equivalent atoms
                {
                    atom_symmetry_class->add_atom_id(j);
                    atoms_[j]->set_symmetry_class(atom_symmetry_class);
                }
            }
        }
    }
    
    assert(num_atom_symmetry_classes() != 0);
}

std::vector<double> Unit_cell::find_mt_radii()
{
    if (nearest_neighbours_.size() == 0) error_local(__FILE__, __LINE__, "array of nearest neighbours is empty");

    std::vector<double> Rmt(num_atom_types(), 1e10);
    
    for (int ia = 0; ia < num_atoms(); ia++)
    {
        int id1 = atom(ia)->type_id();
        if (nearest_neighbours_[ia].size() > 1)
        {
            /* don't allow spheres to touch: take a smaller value than half a distance */
            double R = 0.95 * nearest_neighbours_[ia][1].distance / 2;

            /* take minimal R for the given atom type */
            Rmt[id1] = std::min(R, Rmt[id1]);
        }
        else
        {
            Rmt[id1] = std::min(3.0, Rmt[id1]);
        }
    }
    
    /* Suppose we have 3 different atoms. First we determint Rmt between 1st and 2nd atom, 
     * then we determine Rmt between (let's say) 2nd and 3rd atom and at this point we reduce 
     * the Rmt of the 2nd atom. This means that the 1st atom gets a possibility to expand if 
     * he is far from the 3rd atom.
     */
    bool inflate = true;
    
    if (inflate)
    {
        std::vector<bool> scale_Rmt(num_atom_types(), true);
        for (int ia = 0; ia < num_atoms(); ia++)
        {
            int id1 = atom(ia)->type_id();

            if (nearest_neighbours_[ia].size() > 1)
            {
                int ja = nearest_neighbours_[ia][1].atom_id;
                int id2 = atom(ja)->type_id();
                double dist = nearest_neighbours_[ia][1].distance;
            
                if (Rmt[id1] + Rmt[id2] > dist * 0.94)
                {
                    scale_Rmt[id1] = false;
                    scale_Rmt[id2] = false;
                }
            }
        }

        for (int ia = 0; ia < num_atoms(); ia++)
        {
            int id1 = atom(ia)->type_id();
            if (nearest_neighbours_[ia].size() > 1)
            {
                int ja = nearest_neighbours_[ia][1].atom_id;
                int id2 = atom(ja)->type_id();
                double dist = nearest_neighbours_[ia][1].distance;
                
                if (scale_Rmt[id1]) Rmt[id1] = 0.95 * (dist - Rmt[id2]);
            }
        }
    }
    
    for (int i = 0; i < num_atom_types(); i++) 
    {
        Rmt[i] = std::min(Rmt[i], 3.0);
        if (Rmt[i] < 0.3) error_global(__FILE__, __LINE__, "Muffin-tin radius is too small");
    }

    return Rmt;
}

bool Unit_cell::check_mt_overlap(int& ia__, int& ja__)
{
    if (num_atoms() != 0 && nearest_neighbours_.size() == 0) 
        error_local(__FILE__, __LINE__, "array of nearest neighbours is empty");

    for (int ia = 0; ia < num_atoms(); ia++)
    {
        if (nearest_neighbours_[ia].size() <= 1) // first atom is always the central one itself
        {
            std::stringstream s;
            s << "array of nearest neighbours for atom " << ia << " is empty";
            error_local(__FILE__, __LINE__, s);
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

void Unit_cell::initialize(int lmax_apw__, int lmax_pot__, int num_mag_dims__)
{
    /* split number of atom between all MPI ranks */
    spl_num_atoms_ = splindex<block>(num_atoms(), comm_.size(), comm_.rank());

    if (num_atoms() != 0)
    {
        comm_bundle_atoms_ = Communicator_bundle(comm_, num_atoms());
        spl_atoms_ = splindex<block>(num_atoms(), comm_bundle_atoms_.size(), comm_bundle_atoms_.id());
        for (int ia = 0; ia < num_atoms(); ia++)
        {
            int rank = spl_num_atoms().local_rank(ia);
            if (comm_.rank() == rank)
            {
                if (comm_bundle_atoms_.comm().rank() != 0) error_local(__FILE__, __LINE__, "wrong root rank");
            }
        }
    }

    /* initialize atom types */
    max_num_mt_points_ = 0;
    max_mt_basis_size_ = 0;
    max_mt_radial_basis_size_ = 0;
    max_mt_aw_basis_size_ = 0;
    lmax_beta_ = -1;
    int offs_lo = 0;
    for (int iat = 0; iat < num_atom_types(); iat++)
    {
        atom_type(iat)->init(lmax_apw__, offs_lo);
        max_num_mt_points_ = std::max(max_num_mt_points_, atom_type(iat)->num_mt_points());
        max_mt_basis_size_ = std::max(max_mt_basis_size_, atom_type(iat)->mt_basis_size());
        max_mt_radial_basis_size_ = std::max(max_mt_radial_basis_size_, atom_type(iat)->mt_radial_basis_size());
        max_mt_aw_basis_size_ = std::max(max_mt_aw_basis_size_, atom_type(iat)->mt_aw_basis_size());
        lmax_beta_ = std::max(lmax_beta_, atom_type(iat)->indexr().lmax());
        offs_lo += atom_type(iat)->mt_lo_basis_size(); 
    }
    
    /* find the charges */
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
    
    /* initialize atoms */
    mt_basis_size_ = 0;
    mt_aw_basis_size_ = 0;
    mt_lo_basis_size_ = 0;
    for (int ia = 0; ia < num_atoms(); ia++)
    {
        atom(ia)->init(lmax_pot__, num_mag_dims__, mt_aw_basis_size_, mt_lo_basis_size_, mt_basis_size_);
        mt_aw_basis_size_ += atom(ia)->type()->mt_aw_basis_size();
        mt_lo_basis_size_ += atom(ia)->type()->mt_lo_basis_size();
        mt_basis_size_ += atom(ia)->type()->mt_basis_size();
    }

    assert(mt_basis_size_ == mt_aw_basis_size_ + mt_lo_basis_size_);

    update();

    if (esm_type_ == ultrasoft_pseudopotential || esm_type_ == norm_conserving_pseudopotential)
    {
        /* split beta-projectors into chunks */
        int num_atoms_in_chunk = (comm_.size() == 1) ? num_atoms() : std::min(num_atoms(), 256);
        int num_beta_chunks = num_atoms() / num_atoms_in_chunk + std::min(1, num_atoms() % num_atoms_in_chunk);
        splindex<block> spl_beta_chunks(num_atoms(), num_beta_chunks, 0);
        beta_chunks_.resize(num_beta_chunks);
        
        for (int ib = 0; ib < num_beta_chunks; ib++)
        {
            /* number of atoms in chunk */
            int na = (int)spl_beta_chunks.local_size(ib);
            beta_chunks_[ib].num_atoms_ = na;
            beta_chunks_[ib].desc_ = mdarray<int, 2>(4, na);
            beta_chunks_[ib].atom_pos_ = mdarray<double, 2>(3, na);

            int num_beta = 0;
    
            for (int i = 0; i < na; i++)
            {
                int ia = (int)spl_beta_chunks.global_index(i, ib);
                auto type = atom(ia)->type();
                /* atom fractional coordinates */
                for (int x = 0; x < 3; x++) beta_chunks_[ib].atom_pos_(x, i) = atom(ia)->position(x);
                /* number of beta functions for atom */
                beta_chunks_[ib].desc_(0, i) = type->mt_basis_size();
                /* offset in beta_gk*/
                beta_chunks_[ib].desc_(1, i) = num_beta;
                /* offset in beta_gk_t */
                beta_chunks_[ib].desc_(2, i) = type->offset_lo();
                beta_chunks_[ib].desc_(3, i) = ia;
    
                num_beta += type->mt_basis_size();
            }
            beta_chunks_[ib].num_beta_ = num_beta;

            if (pu_ == GPU)
            {
                #ifdef _GPU_
                beta_chunks_[ib].desc_.allocate_on_device();
                beta_chunks_[ib].desc_.copy_to_device();

                beta_chunks_[ib].atom_pos_.allocate_on_device();
                beta_chunks_[ib].atom_pos_.copy_to_device();
                #endif
            }
        }

        num_beta_t_ = 0;
        for (int iat = 0; iat < num_atom_types(); iat++) num_beta_t_ += atom_type(iat)->mt_lo_basis_size();
    }
            
    mt_aw_basis_descriptors_.resize(mt_aw_basis_size_);
    for (int ia = 0, n = 0; ia < num_atoms(); ia++)
    {
        for (int xi = 0; xi < atom(ia)->mt_aw_basis_size(); xi++, n++)
        {
            mt_aw_basis_descriptors_[n].ia = ia;
            mt_aw_basis_descriptors_[n].xi = xi;
        }
    }

    mt_lo_basis_descriptors_.resize(mt_lo_basis_size_);
    for (int ia = 0, n = 0; ia < num_atoms(); ia++)
    {
        for (int xi = 0; xi < atom(ia)->mt_lo_basis_size(); xi++, n++)
        {
            mt_lo_basis_descriptors_[n].ia = ia;
            mt_lo_basis_descriptors_[n].xi = xi;
        }
    }
}

void Unit_cell::update()
{
    vector3d<double> v0(lattice_vectors_(0, 0), lattice_vectors_(1, 0), lattice_vectors_(2, 0));
    vector3d<double> v1(lattice_vectors_(0, 1), lattice_vectors_(1, 1), lattice_vectors_(2, 1));
    vector3d<double> v2(lattice_vectors_(0, 2), lattice_vectors_(1, 2), lattice_vectors_(2, 2));

    //== find_nearest_neighbours(v0.length() + v1.length() + v2.length());
    double r = std::max(v0.length(), std::max(v1.length(), v2.length()));
    find_nearest_neighbours(r);

    if (full_potential())
    {
        /* find new MT radii and initialize radial grid */
        if (auto_rmt())
        {
            std::vector<double> Rmt = find_mt_radii();
            for (int iat = 0; iat < num_atom_types(); iat++) 
            {
                atom_type(iat)->set_mt_radius(Rmt[iat]);
                atom_type(iat)->set_radial_grid();
            }
        }
        
        int ia, ja;
        if (check_mt_overlap(ia, ja))
        {
            std::stringstream s;
            s << "overlaping muffin-tin spheres for atoms " << ia << "(" << atom(ia)->type()->symbol() << ")" << " and " 
              << ja << "(" << atom(ja)->type()->symbol() << ")" << std::endl 
              << "  radius of atom " << ia << " : " << atom(ia)->type()->mt_radius() << std::endl
              << "  radius of atom " << ja << " : " << atom(ja)->type()->mt_radius() << std::endl
              << "  distance : " << nearest_neighbours_[ia][1].distance << " " << nearest_neighbours_[ja][1].distance;
            error_local(__FILE__, __LINE__, s);
        }
        
        min_mt_radius_ = 1e100;
        max_mt_radius_ = 0;
        for (int i = 0; i < num_atom_types(); i++)
        {
             min_mt_radius_ = std::min(min_mt_radius_, atom_type(i)->mt_radius());
             max_mt_radius_ = std::max(max_mt_radius_, atom_type(i)->mt_radius());
        }
    }
    
    get_symmetry();
    
    spl_num_atom_symmetry_classes_ = splindex<block>(num_atom_symmetry_classes(), comm_.size(), comm_.rank());
    
    volume_mt_ = 0.0;
    if (full_potential())
    {
        for (int ia = 0; ia < num_atoms(); ia++)
        {
            volume_mt_ += fourpi * pow(atom(ia)->mt_radius(), 3) / 3.0; 
        }
    }
    
    volume_it_ = omega() - volume_mt_;
}

void Unit_cell::clear()
{
    delete symmetry_;

    /* delete atom types */
    for (int i = 0; i < (int)atom_types_.size(); i++) delete atom_types_[i];
    atom_types_.clear();
    atom_type_id_map_.clear();

    /* delete atom classes */
    for (int i = 0; i < (int)atom_symmetry_classes_.size(); i++) delete atom_symmetry_classes_[i];
    atom_symmetry_classes_.clear();

    /* delete atoms */
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
        printf("  a%1i : %18.10f %18.10f %18.10f \n", i + 1, lattice_vectors_(0, i), 
                                                             lattice_vectors_(1, i), 
                                                             lattice_vectors_(2, i)); 
    }
    printf("reciprocal lattice vectors\n");
    for (int i = 0; i < 3; i++)
    {
        printf("  b%1i : %18.10f %18.10f %18.10f \n", i + 1, reciprocal_lattice_vectors_(0, i), 
                                                             reciprocal_lattice_vectors_(1, i), 
                                                             reciprocal_lattice_vectors_(2, i));
    }
    printf("\n");
    printf("unit cell volume : %18.8f [a.u.^3]\n", omega());
    printf("1/sqrt(omega)    : %18.8f\n", 1.0 / sqrt(omega()));
    printf("MT volume        : %f (%5.2f%%)\n", volume_mt(), volume_mt() * 100 / omega());
    printf("IT volume        : %f (%5.2f%%)\n", volume_it(), volume_it() * 100 / omega());
    
    printf("\n"); 
    printf("number of atom types : %i\n", num_atom_types());
    for (int i = 0; i < num_atom_types(); i++)
    {
        int id = atom_type(i)->id();
        printf("type id : %i   symbol : %2s   mt_radius : %10.6f\n", id, atom_type(i)->symbol().c_str(), 
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

    if (symmetry_ != nullptr)
    {
        printf("\n");
        printf("space group number   : %i\n", symmetry_->spacegroup_number());
        printf("international symbol : %s\n", symmetry_->international_symbol().c_str());
        printf("Hall symbol          : %s\n", symmetry_->hall_symbol().c_str());
        printf("number of operations : %i\n", symmetry_->num_sym_op());
        printf("transformation matrix : \n");
        auto tm = symmetry_->transformation_matrix();
        for (int i = 0; i < 3; i++)
        {

            for (int j = 0; j < 3; j++) printf("%12.6f ", tm(i, j));
            printf("\n");
        }
        printf("origin shift : \n");
        auto t = symmetry_->origin_shift();
        printf("%12.6f %12.6f %12.6f\n", t[0], t[1], t[2]);

        printf("symmetry operations  : \n");
        for (int isym = 0; isym < symmetry_->num_sym_op(); isym++)
        {
            auto R = symmetry_->rot_mtrx(isym);
            auto t = symmetry_->fractional_translation(isym);
            printf("isym : %i\n", isym);
            printf("R : ");
            for (int i = 0; i < 3; i++)
            {
                if (i) printf("    ");
                for (int j = 0; j < 3; j++) printf("%3i ", R(i, j));
                printf("\n");
            }
            printf("T : ");
            for (int j = 0; j < 3; j++) printf("%f8.4 ", t[j]);
            printf("\n");
        }
    }
    
    printf("\n");
    printf("total nuclear charge        : %i\n", total_nuclear_charge_);
    printf("number of core electrons    : %f\n", num_core_electrons_);
    printf("number of valence electrons : %f\n", num_valence_electrons_);
    printf("total number of electrons   : %f\n", num_electrons_);
}

unit_cell_parameters_descriptor Unit_cell::unit_cell_parameters()
{
    unit_cell_parameters_descriptor d;

    vector3d<double> v0(lattice_vectors_(0, 0), lattice_vectors_(1, 0), lattice_vectors_(2, 0));
    vector3d<double> v1(lattice_vectors_(0, 1), lattice_vectors_(1, 1), lattice_vectors_(2, 1));
    vector3d<double> v2(lattice_vectors_(0, 2), lattice_vectors_(1, 2), lattice_vectors_(2, 2));

    d.a = v0.length();
    d.b = v1.length();
    d.c = v2.length();

    d.alpha = acos((v1 * v2) / d.b / d.c) * 180 / pi;
    d.beta  = acos((v0 * v2) / d.a / d.c) * 180 / pi;
    d.gamma = acos((v0 * v1) / d.a / d.b) * 180 / pi;

    return d;
}

void Unit_cell::write_cif()
{
    if (comm_.rank() == 0)
    {
        FILE* fout = fopen("unit_cell.cif", "w");

        auto d = unit_cell_parameters();
        
        fprintf(fout, "_cell_length_a %f\n", d.a);
        fprintf(fout, "_cell_length_b %f\n", d.b);
        fprintf(fout, "_cell_length_c %f\n", d.c);
        fprintf(fout, "_cell_angle_alpha %f\n", d.alpha);
        fprintf(fout, "_cell_angle_beta %f\n", d.beta);
        fprintf(fout, "_cell_angle_gamma %f\n", d.gamma);

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
            s << ia + 1 << " " << atom(ia)->type()->label() << " " << atom(ia)->position(0) << " " << 
                 atom(ia)->position(1) << " " << atom(ia)->position(2);
            fprintf(fout,"%s\n",s.str().c_str());
        }
        fclose(fout);
    }
}

void Unit_cell::write_json()
{
    if (comm_.rank() == 0)
    {
        JSON_write out("unit_cell.json");

        out.begin_set("unit_cell");
        out.begin_array("lattice_vectors");
        for (int i = 0; i < 3; i++)
        {
            std::vector<double> v(3);
            for (int x = 0; x < 3; x++) v[x] = lattice_vectors_(x, i);
            out.write(v);
        }
        out.end_array();
        out.begin_array("atom_types");
        for (int iat = 0; iat < num_atom_types(); iat++) out.write(atom_type(iat)->label());
        out.end_array();
        out.begin_set("atom_files");
        for (int iat = 0; iat < num_atom_types(); iat++) out.single(atom_type(iat)->label().c_str(), atom_type(iat)->file_name().c_str());
        out.end_set();

        out.begin_set("atoms");
        for (int iat = 0; iat < num_atom_types(); iat++)
        {
            out.begin_array(atom_type(iat)->label().c_str());
            for (int i = 0; i < atom_type(iat)->num_atoms(); i++)
            {
                int ia = atom_type(iat)->atom_id(i);
                std::vector<double> v(3);
                for (int x = 0; x < 3; x++) v[x] = atom(ia)->position(x);
                out.write(v);
            }
            out.end_array();
        }
        out.end_set();
        out.end_set();
    }
}

void Unit_cell::set_lattice_vectors(double* a0__, double* a1__, double* a2__)
{
    for (int x = 0; x < 3; x++)
    {
        lattice_vectors_(x, 0) = a0__[x];
        lattice_vectors_(x, 1) = a1__[x];
        lattice_vectors_(x, 2) = a2__[x];
    }
    inverse_lattice_vectors_ = inverse(lattice_vectors_);
    omega_ = std::abs(lattice_vectors_.det());
    reciprocal_lattice_vectors_ = transpose(inverse(lattice_vectors_)) * twopi;
}

void Unit_cell::find_nearest_neighbours(double cluster_radius)
{
    Timer t("sirius::Unit_cell::find_nearest_neighbours");

    vector3d<int> max_frac_coord = Utils::find_translation_limits(cluster_radius, lattice_vectors_);
   
    nearest_neighbours_.clear();
    nearest_neighbours_.resize(num_atoms());

    #pragma omp parallel for default(shared)
    for (int ia = 0; ia < num_atoms(); ia++)
    {
        vector3d<double> iapos = get_cartesian_coordinates(atom(ia)->position());
        
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
                    
                    vector3d<double> vt = get_cartesian_coordinates(nnd.translation);
                    
                    for (int ja = 0; ja < num_atoms(); ja++)
                    {
                        nnd.atom_id = ja;

                        vector3d<double> japos = get_cartesian_coordinates(atom(ja)->position());

                        vector3d<double> v = japos + vt - iapos;

                        nnd.distance = v.length();
                        
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

    //== if (Platform::mpi_rank() == 0) // TODO: move to a separate task
    //== {
    //==     FILE* fout = fopen("nghbr.txt", "w");
    //==     for (int ia = 0; ia < num_atoms(); ia++)
    //==     {
    //==         fprintf(fout, "Central atom: %s (%i)\n", atom(ia)->type()->symbol().c_str(), ia);
    //==         for (int i = 0; i < 80; i++) fprintf(fout, "-");
    //==         fprintf(fout, "\n");
    //==         fprintf(fout, "atom (  id)       D [a.u.]    translation  R\n");
    //==         for (int i = 0; i < 80; i++) fprintf(fout, "-");
    //==         fprintf(fout, "\n");
    //==         for (int i = 0; i < (int)nearest_neighbours_[ia].size(); i++)
    //==         {
    //==             int ja = nearest_neighbours_[ia][i].atom_id;
    //==             fprintf(fout, "%4s (%4i)   %12.6f\n", atom(ja)->type()->symbol().c_str(), ja, 
    //==                                                   nearest_neighbours_[ia][i].distance);
    //==         }
    //==         fprintf(fout, "\n");
    //==     }
    //==     fclose(fout);
    //== }
}

bool Unit_cell::is_point_in_mt(vector3d<double> vc, int& ja, int& jr, double& dr, double tp[2])
{
    vector3d<int> ntr;
    
    /* reduce coordinates to the primitive unit cell */
    auto vr = Utils::reduce_coordinates(get_fractional_coordinates(vc));

    for (int ia = 0; ia < num_atoms(); ia++)
    {
        for (int i0 = -1; i0 <= 1; i0++)
        {
            for (int i1 = -1; i1 <= 1; i1++)
            {
                for (int i2 = -1; i2 <= 1; i2++)
                {
                    /* atom position */
                    vector3d<double> posf = vector3d<double>(i0, i1, i2) + atom(ia)->position();

                    /* vector connecting center of atom and reduced point */
                    vector3d<double> vf = vr.first - posf;
                    
                    /* convert to spherical coordinates */
                    auto vs = SHT::spherical_coordinates(get_cartesian_coordinates(vf));

                    if (vs[0] < atom(ia)->mt_radius())
                    {
                        ja = ia;
                        tp[0] = vs[1]; // theta
                        tp[1] = vs[2]; // phi

                        if (vs[0] < atom(ia)->type()->radial_grid(0))
                        {
                            jr = 0;
                            dr = 0.0;
                        }
                        else
                        {
                            for (int ir = 0; ir < atom(ia)->num_mt_points() - 1; ir++)
                            {
                                if (vs[0] >= atom(ia)->type()->radial_grid(ir) && vs[0] < atom(ia)->type()->radial_grid(ir + 1))
                                {
                                    jr = ir;
                                    dr = vs[0] - atom(ia)->type()->radial_grid(ir);
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
    ja = -1;
    jr = -1;
    return false;
}

void Unit_cell::generate_radial_functions() 
{
    Timer t("sirius::Unit_cell::generate_radial_functions");
   
    for (int icloc = 0; icloc < (int)spl_num_atom_symmetry_classes().local_size(); icloc++)
        atom_symmetry_class(spl_num_atom_symmetry_classes(icloc))->generate_radial_functions();

    for (int ic = 0; ic < num_atom_symmetry_classes(); ic++)
    {
        int rank = spl_num_atom_symmetry_classes().local_rank(ic);
        atom_symmetry_class(ic)->sync_radial_functions(comm_, rank);
    }
    
    if (verbosity_level >= 4)
    {
        pstdout pout(comm_);
        
        for (int icloc = 0; icloc < (int)spl_num_atom_symmetry_classes().local_size(); icloc++)
        {
            int ic = spl_num_atom_symmetry_classes(icloc);
            atom_symmetry_class(ic)->write_enu(pout);
        }

        if (comm_.rank() == 0)
        {
            printf("\n");
            printf("Linearization energies\n");
        }
    }
}

void Unit_cell::generate_radial_integrals()
{
    Timer t("sirius::Unit_cell::generate_radial_integrals");
    
    for (int icloc = 0; icloc < (int)spl_num_atom_symmetry_classes().local_size(); icloc++)
        atom_symmetry_class(spl_num_atom_symmetry_classes(icloc))->generate_radial_integrals();

    for (int ic = 0; ic < num_atom_symmetry_classes(); ic++)
    {
        int rank = spl_num_atom_symmetry_classes().local_rank(ic);
        atom_symmetry_class(ic)->sync_radial_integrals(comm_, rank);
    }

    for (int ialoc = 0; ialoc < (int)spl_atoms_.local_size(); ialoc++)
    {
        int ia = (int)spl_atoms_[ialoc];
        atom(ia)->generate_radial_integrals(comm_bundle_atoms_.comm());
    }
    
    for (int ia = 0; ia < num_atoms(); ia++)
    {
        int rank = spl_num_atoms().local_rank(ia);
        atom(ia)->sync_radial_integrals(comm_, rank);
    }
}

std::string Unit_cell::chemical_formula()
{
    std::string name;
    for (int iat = 0; iat < num_atom_types(); iat++)
    {
        name += atom_type(iat)->symbol();
        int n = 0;
        for (int ia = 0; ia < num_atoms(); ia++)
        {
            if (atom(ia)->type_id() == atom_type(iat)->id()) n++;
        }
        if (n != 1) 
        {
            std::stringstream s;
            s << n;
            name = (name + s.str());
        }
    }

    return name;
}

}
