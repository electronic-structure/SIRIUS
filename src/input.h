#ifndef __INPUT_H__
#define __INPUT_H__

#include <omp.h>
#include "json_tree.h"

namespace sirius {

/// Parse unit cell input section.
/** The following part of the input file is parsed:
 *  \code{.json}
 *      "unit_cell" : {
 *          "lattice_vectors" : [
 *              [a1_x, a1_y, a1_z],
 *              [a2_x, a2_y, a2_z],
 *              [a3_x, a3_y, a3_z]
 *          ],
 *
 *          "lattice_vectors_scale" : scale,
 *
 *          "atom_types" : [label_A, label_B, ...],
 *
 *          "atom_files" : {
 *              label_A : file_A, 
 *              label_B : file_B,
 *              ...
 *          },
 *
 *          "atoms" : {
 *              label_A: [
 *                  coordinates_A_1, 
 *                  coordinates_A_2,
 *                  ...
 *              ],
 *              label_B : [
 *                  coordinates_B_1,
 *                  coordinates_B_2,
 *                  ...
 *              ]
 *          }
 *      }
 *  \endcode
 */
struct Unit_cell_input_section
{
    double lattice_vectors_[3][3];

    std::vector<std::string> labels_;
    std::map<std::string, std::string> atom_files_;
    std::vector< std::vector< std::vector<double> > > coordinates_;

    bool exist_;

    Unit_cell_input_section() : exist_(false)
    {
    }

    void read(JSON_tree const& parser)
    {
        if (parser.exist("unit_cell"))
        {
            exist_ = true;

            auto section = parser["unit_cell"];
            std::vector<double> a0, a1, a2;
            section["lattice_vectors"][0] >> a0;
            section["lattice_vectors"][1] >> a1;
            section["lattice_vectors"][2] >> a2;

            if (a0.size() != 3 || a1.size() != 3 || a2.size() != 3)
                TERMINATE("wrong lattice vectors");

            double scale = section["lattice_vectors_scale"].get(1.0);

            for (int x = 0; x < 3; x++)
            {
                lattice_vectors_[0][x] = a0[x] * scale;
                lattice_vectors_[1][x] = a1[x] * scale;
                lattice_vectors_[2][x] = a2[x] * scale;
            }

            labels_.clear();
            coordinates_.clear();
            
            for (int iat = 0; iat < (int)section["atom_types"].size(); iat++)
            {
                std::string label;
                section["atom_types"][iat] >> label;
                for (int i = 0; i < (int)labels_.size(); i++)
                {
                    if (labels_[i] == label) 
                        TERMINATE("atom type with such label is already in list");
                }
                labels_.push_back(label);
            }
            
            if (section.exist("atom_files"))
            {
                for (int iat = 0; iat < (int)labels_.size(); iat++)
                    atom_files_[labels_[iat]] = section["atom_files"][labels_[iat]].get(std::string(""));
            }
            
            for (int iat = 0; iat < (int)labels_.size(); iat++)
            {
                coordinates_.push_back(std::vector< std::vector<double> >());
                for (int ia = 0; ia < section["atoms"][labels_[iat]].size(); ia++)
                {
                    std::vector<double> v;
                    section["atoms"][labels_[iat]][ia] >> v;

                    if (!(v.size() == 3 || v.size() == 6)) TERMINATE("wrong coordinates size");
                    if (v.size() == 3) v.resize(6, 0.0);

                    coordinates_[iat].push_back(v);
                }
            }
        }
    }
};

struct Mixer_input_section
{
    double beta_;
    double gamma_;
    std::string type_;
    int max_history_;

    bool exist_;

    Mixer_input_section() 
        : beta_(0.9),
          gamma_(1.0),
          type_("broyden2"),
          max_history_(8),
          exist_(false)
    {
    }

    void read(JSON_tree const& parser)
    {
        if (parser.exist("mixer"))
        {
            exist_ = true;
            auto section = parser["mixer"];
            beta_        = section["beta"].get(beta_);
            gamma_       = section["gamma"].get(gamma_);
            max_history_ = section["max_history"].get(max_history_);
            type_        = section["type"].get(type_);
        }
    }
};

//== /// Parse XC functionals input section.
//== /** The following part of the input file is parsed:
//==  *  \code{.json}
//==  *      "xc_functionals" : ["name1", "name2", ...]
//==  *  \endcode
//==  */
//== struct XC_functionals_input_section
//== {
//==     /// List of XC functionals.
//==     std::vector<std::string> xc_functional_names_;
//== 
//==     /// Set default variables.
//==     XC_functionals_input_section()
//==     {
//==         //== xc_functional_names_.push_back("XC_LDA_X");
//==         //== xc_functional_names_.push_back("XC_LDA_C_VWN");
//==     }
//== 
//==     void read(JSON_tree const& parser)
//==     {
//==         if (parser.exist("xc_functionals"))
//==         {
//==             xc_functional_names_.clear();
//==             for (int i = 0; i < parser["xc_functionals"].size(); i++)
//==             {
//==                 std::string s;
//==                 parser["xc_functionals"][i] >> s;
//==                 xc_functional_names_.push_back(s);
//==             }
//==         }
//==     }
//== };

/** \todo real-space projectors are not part of iterative solver */
struct Iterative_solver_input_section
{
    std::string type_;
    int num_steps_;
    int subspace_size_;
    double energy_tolerance_;
    double residual_tolerance_;
    int converge_by_energy_; // TODO: rename, this is meaningless
    int converge_occupied_;
    int min_num_res_;
    int real_space_prj_;
    double R_mask_scale_;
    double mask_alpha_;

    Iterative_solver_input_section() 
        : type_("davidson"),
          num_steps_(20),
          subspace_size_(4),
          energy_tolerance_(1e-6),
          residual_tolerance_(1e-6),
          converge_by_energy_(1),
          converge_occupied_(1),
          min_num_res_(0),
          real_space_prj_(0),
          R_mask_scale_(1.5),
          mask_alpha_(3)
    {
    }

    void read(JSON_tree const& parser)
    {
        type_               = parser["iterative_solver"]["type"].get(type_);
        num_steps_          = parser["iterative_solver"]["num_steps"].get(num_steps_);
        subspace_size_      = parser["iterative_solver"]["subspace_size"].get(subspace_size_);
        energy_tolerance_   = parser["iterative_solver"]["energy_tolerance"].get(energy_tolerance_);
        residual_tolerance_ = parser["iterative_solver"]["residual_tolerance"].get(residual_tolerance_);
        converge_by_energy_ = parser["iterative_solver"]["converge_by_energy"].get(converge_by_energy_);
        converge_occupied_  = parser["iterative_solver"]["converge_occupied"].get(converge_occupied_);
        min_num_res_        = parser["iterative_solver"]["min_num_res"].get(min_num_res_);
        real_space_prj_     = parser["iterative_solver"]["real_space_prj"].get(real_space_prj_);
        R_mask_scale_       = parser["iterative_solver"]["R_mask_scale"].get(R_mask_scale_);
        mask_alpha_         = parser["iterative_solver"]["mask_alpha"].get(mask_alpha_);
    }
};

struct Control_input_section
{
    std::vector<int> mpi_grid_dims_;
    int cyclic_block_size_;
    std::string std_evp_solver_name_;
    std::string gen_evp_solver_name_;
    std::string esm_;
    std::string fft_mode_;
    bool reduce_gvec_;

    /// Type of the processing unit.
    processing_unit_t processing_unit_;

    Control_input_section()
        : cyclic_block_size_(32),
          std_evp_solver_name_("lapack"),
          gen_evp_solver_name_("lapack"),
          esm_("none"),
          fft_mode_("serial"),
          reduce_gvec_(true)
    {
    }

    void read(JSON_tree const& parser)
    {
        mpi_grid_dims_       = parser["control"]["mpi_grid_dims"].get(mpi_grid_dims_); 
        cyclic_block_size_   = parser["control"]["cyclic_block_size"].get(cyclic_block_size_);
        std_evp_solver_name_ = parser["control"]["std_evp_solver_type"].get(std_evp_solver_name_);
        gen_evp_solver_name_ = parser["control"]["gen_evp_solver_type"].get(gen_evp_solver_name_);

        std::string pu = "cpu";
        pu = parser["control"]["processing_unit"].get(pu);
        std::transform(pu.begin(), pu.end(), pu.begin(), ::tolower);
        if (pu == "cpu")
        {
            processing_unit_ = CPU;
        }
        else if (pu == "gpu")
        {
            processing_unit_ = GPU;
        }
        else
        {
            TERMINATE("wrong processing unit");
        }

        esm_ = parser["control"]["electronic_structure_method"].get(esm_);
        std::transform(esm_.begin(), esm_.end(), esm_.begin(), ::tolower);

        fft_mode_ = parser["control"]["fft_mode"].get(fft_mode_);
        reduce_gvec_ = parser["control"]["reduce_gvec"].get<int>(reduce_gvec_);
    }
};

};

#endif // __INPUT_H__

