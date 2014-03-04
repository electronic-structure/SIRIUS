#ifndef __ATOM_TYPE_H__
#define __ATOM_TYPE_H__

#include <string.h>
#include <vector>
#include "mdarray.h"
#include "descriptors.h"
#include "vector3d.h"
#include "utils.h"
#include "radial_grid.h"
#include "radial_solver.h"
#include "json_tree.h"
#include "libxc_interface.h"

namespace sirius {

class radial_functions_index
{
    private: 

        std::vector<radial_function_index_descriptor> radial_function_index_descriptors_;

        mdarray<int, 2> index_by_l_order_;

        mdarray<int, 1> index_by_idxlo_;

        /// number of radial functions for each angular momentum quantum number
        std::vector<int> num_rf_;

        /// number of local orbitals for each angular momentum quantum number
        std::vector<int> num_lo_;

        // maximum number of radial functions across all angular momentums
        int max_num_rf_;

        int lmax_aw_;

        int lmax_lo_;

        int lmax_;
    
    public:

        void init(const std::vector<radial_solution_descriptor_set>& aw_descriptors, 
                  const std::vector<local_orbital_descriptor>& lo_descriptors)
        {
            lmax_aw_ = (int)aw_descriptors.size() - 1;
            lmax_lo_ = -1;
            for (int idxlo = 0; idxlo < (int)lo_descriptors.size(); idxlo++)
            {
                int l = lo_descriptors[idxlo].l;
                lmax_lo_ = std::max(lmax_lo_, l);
            }

            lmax_ = std::max(lmax_aw_, lmax_lo_);

            num_rf_ = std::vector<int>(lmax_ + 1, 0);
            num_lo_ = std::vector<int>(lmax_ + 1, 0);
            
            max_num_rf_ = 0;

            radial_function_index_descriptors_.clear();

            for (int l = 0; l <= lmax_aw_; l++)
            {
                assert(aw_descriptors[l].size() <= 3);

                for (int order = 0; order < (int)aw_descriptors[l].size(); order++)
                {
                    radial_function_index_descriptors_.push_back(radial_function_index_descriptor(l, num_rf_[l]));
                    num_rf_[l]++;
                }
            }

            for (int idxlo = 0; idxlo < (int)lo_descriptors.size(); idxlo++)
            {
                int l = lo_descriptors[idxlo].l;
                radial_function_index_descriptors_.push_back(radial_function_index_descriptor(l, num_rf_[l], idxlo));
                num_rf_[l]++;
                num_lo_[l]++;
            }

            for (int l = 0; l <= lmax_; l++) max_num_rf_ = std::max(max_num_rf_, num_rf_[l]);

            index_by_l_order_.set_dimensions(lmax_ + 1, max_num_rf_);
            index_by_l_order_.allocate();

            if (lo_descriptors.size())
            {
                index_by_idxlo_.set_dimensions((int)lo_descriptors.size());
                index_by_idxlo_.allocate();
            }

            for (int i = 0; i < (int)radial_function_index_descriptors_.size(); i++)
            {
                int l = radial_function_index_descriptors_[i].l;
                int order = radial_function_index_descriptors_[i].order;
                int idxlo = radial_function_index_descriptors_[i].idxlo;
                index_by_l_order_(l, order) = i;
                if (idxlo >= 0) index_by_idxlo_(idxlo) = i; 
            }
        }

        inline int size()
        {
            return (int)radial_function_index_descriptors_.size();
        }

        inline radial_function_index_descriptor& operator[](int i)
        {
            assert(i >= 0 && i < (int)radial_function_index_descriptors_.size());
            return radial_function_index_descriptors_[i];
        }

        inline int index_by_l_order(int l, int order)
        {
            return index_by_l_order_(l, order);
        }

        inline int index_by_idxlo(int idxlo)
        {
            return index_by_idxlo_(idxlo);
        }

        /// Number of radial functions for a given orbital quantum number.
        inline int num_rf(int l)
        {
            assert(l >= 0 && l < (int)num_rf_.size());
            return num_rf_[l];
        }

        /// Number of local orbitals for a given orbital quantum number.
        inline int num_lo(int l)
        {
            assert(l >= 0 && l < (int)num_lo_.size());
            return num_lo_[l];
        }
        
        /// Maximum possible number of radial functions for an orbital quantum number.
        inline int max_num_rf()
        {
            return max_num_rf_;
        }

        inline int lmax()
        {
            return lmax_;
        }
        
        inline int lmax_lo()
        {
            return lmax_lo_;
        }
};

class basis_functions_index
{
    private:

        std::vector<basis_function_index_descriptor> basis_function_index_descriptors_; 
       
        mdarray<int, 2> index_by_lm_order_;

        mdarray<int, 1> index_by_idxrf_;
       
        /// number of augmented wave basis functions
        int size_aw_;
       
        /// number of local orbital basis functions
        int size_lo_;

    public:

        basis_functions_index() : size_aw_(0), size_lo_(0)
        {
        }
        
        void init(radial_functions_index& indexr)
        {
            basis_function_index_descriptors_.clear();

            index_by_idxrf_.set_dimensions(indexr.size());
            index_by_idxrf_.allocate();

            for (int idxrf = 0; idxrf < indexr.size(); idxrf++)
            {
                int l = indexr[idxrf].l;
                int order = indexr[idxrf].order;
                int idxlo = indexr[idxrf].idxlo;

                index_by_idxrf_(idxrf) = (int)basis_function_index_descriptors_.size();

                for (int m = -l; m <= l; m++)
                    basis_function_index_descriptors_.push_back(basis_function_index_descriptor(l, m, order, idxlo, idxrf));
            }

            index_by_lm_order_.set_dimensions(Utils::lmmax(indexr.lmax()), indexr.max_num_rf());
            index_by_lm_order_.allocate();

            for (int i = 0; i < (int)basis_function_index_descriptors_.size(); i++)
            {
                int lm = basis_function_index_descriptors_[i].lm;
                int order = basis_function_index_descriptors_[i].order;
                index_by_lm_order_(lm, order) = i;
                
                // get number of aw basis functions
                if (basis_function_index_descriptors_[i].idxlo < 0) size_aw_ = i + 1;
            }

            size_lo_ = (int)basis_function_index_descriptors_.size() - size_aw_;

            assert(size_aw_ >= 0);
            assert(size_lo_ >= 0);
        } 

        /// Return total number of MT basis functions.
        inline int size()
        {
            return (int)basis_function_index_descriptors_.size();
        }

        inline int size_aw()
        {
            return size_aw_;
        }

        inline int size_lo()
        {
            return size_lo_;
        }
        
        inline int index_by_l_m_order(int l, int m, int order)
        {
            return index_by_lm_order_(Utils::lm_by_l_m(l, m), order);
        }
        
        inline int index_by_lm_order(int lm, int order)
        {
            return index_by_lm_order_(lm, order);
        }

        inline int index_by_idxrf(int idxrf)
        {
            return index_by_idxrf_(idxrf);
        }
        
        inline basis_function_index_descriptor& operator[](int i)
        {
            assert(i >= 0 && i < (int)basis_function_index_descriptors_.size());
            return basis_function_index_descriptors_[i];
        }
};

/**
    \todo Arbitrary AW order
*/
class Atom_type
{
    private:

        /// unique id of atom type
        int id_;
    
        /// chemical element symbol
        std::string symbol_;

        /// chemical element name
        std::string name_;
        
        /// nucleus charge, treated as positive(!) integer
        int zn_;

        /// atom mass
        double mass_;

        /// muffin-tin radius
        double mt_radius_;

        /// number of muffin-tin points
        int num_mt_points_;
        
        /// beginning of the radial grid
        double radial_grid_origin_;
        
        /// effective infinity distance
        double radial_grid_infinity_;
        
        /// radial grid
        Radial_grid* radial_grid_;

        /// list of atomic levels 
        std::vector<atomic_level_descriptor> atomic_levels_;

        /// number of core electrons
        double num_core_electrons_;

        /// number of valence electrons
        double num_valence_electrons_;
        
        /// default augmented wave configuration
        radial_solution_descriptor_set aw_default_l_;
        
        /// augmented wave configuration for specific l
        std::vector<radial_solution_descriptor_set> aw_specific_l_;

        /// list of radial descriptor sets used to construct augmented waves 
        std::vector<radial_solution_descriptor_set> aw_descriptors_;
        
        /// list of radial descriptor sets used to construct local orbitals
        std::vector<local_orbital_descriptor> lo_descriptors_;
        
        /// density of a free atom
        std::vector<double> free_atom_density_;
        
        /// potential of a free atom
        std::vector<double> free_atom_potential_;

        mdarray<double, 2> free_atom_radial_functions_;

        /// maximum number of aw radial functions across angular momentums
        int max_aw_order_;

        radial_functions_index indexr_;
        
        basis_functions_index indexb_;

        uspp_descriptor uspp_;

        std::vector<int> atom_id_;

        /// type of electronic structure method used
        electronic_structure_method_t esm_type_;
        
        /// atom type label
        std::string label_;

        std::string file_name_;

        bool initialized_;
       
        // forbid copy constructor
        Atom_type(const Atom_type& src);
        
        // forbid assignment operator
        Atom_type& operator=(const Atom_type& src);
        
        void read_input_core(JSON_tree& parser);

        void read_input_aw(JSON_tree& parser);

        void read_input_lo(JSON_tree& parser);

        void read_input(const std::string& fname);
    
        void init_aw_descriptors(int lmax);

    public:
        
        Atom_type(const char* symbol__, const char* name__, int zn__, double mass__, 
                  std::vector<atomic_level_descriptor>& levels__);
 
        Atom_type(int id__, const std::string label, const std::string file_name__, electronic_structure_method_t esm_type__);

        Atom_type(int id__);

        ~Atom_type();
        
        void init(int lmax_apw);

        void set_radial_grid(int num_points = -1, double* points = NULL);

        /// Add augmented-wave descriptor.
        void add_aw_descriptor(int n, int l, double enu, int dme, int auto_enu);
        
        /// Add local orbital descriptor
        void add_lo_descriptor(int ilo, int n, int l, double enu, int dme, int auto_enu);

        /// Solve free atom and find SCF density and potential.
        /** Free atom potential is used to augment the MT potential and find the energy of the bound states which is used
            as a linearization energy (auto_enu = 1). */ 
        double solve_free_atom(double solver_tol, double energy_tol, double charge_tol, std::vector<double>& enu);

        void print_info();
        
        void sync_free_atom(int rank);

        void fix_q_radial_function(int l, int i, int j, double* qrf);
        
        inline int id()
        {
            return id_;
        }
        
        inline int zn()
        {
            assert(zn_ > 0);
            return zn_;
        }
        
        const std::string& symbol()
        { 
            return symbol_;
        }

        const std::string& name()
        { 
            return name_;
        }
        
        inline double mass()
        {
            return mass_;
        }
        
        inline double mt_radius()
        {
            return mt_radius_;
        }
        
        inline int num_mt_points()
        {
            assert(num_mt_points_ > 0);
            return num_mt_points_;
        }
        
        inline Radial_grid& radial_grid()
        {
            assert(num_mt_points_ > 0);
            assert(radial_grid_->size() > 0);
            return (*radial_grid_);
        }
        
        inline double radial_grid(int ir)
        {
            return (*radial_grid_)[ir];
        }
        
        inline int num_atomic_levels()
        {
            return (int)atomic_levels_.size();
        }    
        
        inline atomic_level_descriptor& atomic_level(int idx)
        {
            return atomic_levels_[idx];
        }
        
        inline double num_core_electrons()
        {
            return num_core_electrons_;
        }
        
        inline double num_valence_electrons()
        {
            return num_valence_electrons_;
        }
        
        inline double free_atom_density(const int idx)
        {
            assert(idx >= 0 && idx < (int)free_atom_density_.size());

            return free_atom_density_[idx];
        }
        
        inline double free_atom_potential(const int idx)
        {
            assert(idx >= 0 && idx < (int)free_atom_potential_.size());
            return free_atom_potential_[idx];
        }

        inline int num_aw_descriptors()
        {
            return (int)aw_descriptors_.size();
        }

        inline radial_solution_descriptor_set& aw_descriptor(int l)
        {
            assert(l < (int)aw_descriptors_.size());
            return aw_descriptors_[l];
        }
        
        inline int num_lo_descriptors()
        {
            return (int)lo_descriptors_.size();
        }

        inline local_orbital_descriptor& lo_descriptor(int idx)
        {
            return lo_descriptors_[idx];
        }

        inline int max_aw_order()
        {
            return max_aw_order_;
        }

        inline radial_functions_index& indexr()
        {
            return indexr_;
        }
        
        inline radial_function_index_descriptor& indexr(int i)
        {
            assert(i >= 0 && i < (int)indexr_.size());
            return indexr_[i];
        }

        inline int indexr_by_l_order(int l, int order)
        {
            return indexr_.index_by_l_order(l, order);
        }
        
        inline int indexr_by_idxlo(int idxlo)
        {
            return indexr_.index_by_idxlo(idxlo);
        }

        inline basis_functions_index& indexb()
        {
            return indexb_;
        }

        inline basis_function_index_descriptor& indexb(int i)
        {
            assert(i >= 0 && i < (int)indexb_.size());
            return indexb_[i];
        }

        inline int indexb_by_l_m_order(int l, int m, int order)
        {
            return indexb_.index_by_l_m_order(l, m, order);
        }
        
        inline int indexb_by_lm_order(int lm, int order)
        {
            return indexb_.index_by_lm_order(lm, order);
        }

        inline int mt_aw_basis_size()
        {
            return indexb_.size_aw();
        }
        
        inline int mt_lo_basis_size()
        {
            return indexb_.size_lo();
        }

        inline int mt_basis_size()
        {
            return indexb_.size();
        }

        inline int mt_radial_basis_size()
        {
            return indexr_.size();
        }

        inline double free_atom_radial_function(int ir, int ist)
        {
            return free_atom_radial_functions_(ir, ist);
        }

        inline std::vector<double>& free_atom_potential()
        {
            return free_atom_potential_;
        }

        inline uspp_descriptor& uspp()
        {
            return uspp_;
        }

        inline void set_symbol(const std::string symbol__)
        {
            symbol_ = symbol__;
        }

        inline void set_zn(int zn__)
        {
            zn_ = zn__;
        }

        inline void set_mass(double mass__)
        {
            mass_ = mass__;
        }
        
        inline void set_mt_radius(double mt_radius__)
        {
            mt_radius_ = mt_radius__;
        }

        inline void set_num_mt_points(int num_mt_points__)
        {
            num_mt_points_ = num_mt_points__;
        }

        inline void set_radial_grid_origin(double radial_grid_origin__)
        {
            radial_grid_origin_ = radial_grid_origin__;
        }

        inline void set_radial_grid_infinity(double radial_grid_infinity__)
        {
            radial_grid_infinity_ = radial_grid_infinity__;
        }

        inline void set_configuration(int n, int l, int k, double occupancy, bool core)
        {
            atomic_level_descriptor level;
            level.n = n;
            level.l = l;
            level.k = k;
            level.occupancy = occupancy;
            level.core = core;
            atomic_levels_.push_back(level);
        }

        inline int num_atoms()
        {
            return (int)atom_id_.size();
        }

        inline int atom_id(int idx)
        {
            return atom_id_[idx];
        }

        inline void add_atom_id(int atom_id__)
        {
            atom_id_.push_back(atom_id__);
        }

        inline bool initialized()
        {
            return initialized_;
        }

        inline std::string label()
        {
            return label_;
        }
};

};

#endif // __ATOM_TYPE_H__

