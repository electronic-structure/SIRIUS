#include "atom_type.h"


namespace sirius {

void Atom_type::init(int offset_lo__)
{
    /* check if the class instance was already initialized */
    if (initialized_) TERMINATE("can't initialize twice");

    offset_lo_ = offset_lo__;

    /* read data from file if it exists */
    if (file_name_.length() > 0)
    {
        if (!Utils::file_exists(file_name_))
        {
            std::stringstream s;
            s << "file " + file_name_ + " doesn't exist";
            TERMINATE(s);
        }
        else
        {
            read_input(file_name_);
        }
    }

    /* add valence levels to the list of core levels */
    if (parameters_.full_potential())
    {
        atomic_level_descriptor level;
        for (int ist = 0; ist < 28; ist++)
        {
            bool found = false;
            level.n = atomic_conf[zn_ - 1][ist][0];
            level.l = atomic_conf[zn_ - 1][ist][1];
            level.k = atomic_conf[zn_ - 1][ist][2];
            level.occupancy = double(atomic_conf[zn_ - 1][ist][3]);
            level.core = false;

            if (level.n != -1)
            {
                for (int jst = 0; jst < (int)atomic_levels_.size(); jst++)
                {
                    if ((atomic_levels_[jst].n == level.n) &&
                        (atomic_levels_[jst].l == level.l) &&
                        (atomic_levels_[jst].k == level.k)) found = true;
                }
                if (!found) atomic_levels_.push_back(level);
            }
        }
    }

    /* check the nuclear charge */
    if (zn_ == 0) TERMINATE("zero atom charge");

    /* set default radial grid if it was not done by user */
    if (radial_grid_.num_points() == 0) set_radial_grid();

    if (parameters_.esm_type() == full_potential_lapwlo)
    {
        /* initialize free atom density and potential */
        init_free_atom(false);

        /* initialize aw descriptors if they were not set manually */
        if (aw_descriptors_.size() == 0) init_aw_descriptors(parameters_.lmax_apw());

        if (static_cast<int>(aw_descriptors_.size()) != (parameters_.lmax_apw() + 1))
            TERMINATE("wrong size of augmented wave descriptors");

        max_aw_order_ = 0;
        for (int l = 0; l <= parameters_.lmax_apw(); l++) max_aw_order_ = std::max(max_aw_order_, (int)aw_descriptors_[l].size());

        if (max_aw_order_ > 3) TERMINATE("maximum aw order > 3");
    }

    if (!parameters_.full_potential())
    {
        local_orbital_descriptor lod;
        for (int i = 0; i < uspp_.num_beta_radial_functions; i++)
        {
            /* think of |beta> functions as of local orbitals */
            lod.l = uspp_.beta_l[i];
            lo_descriptors_.push_back(lod);
        }
    }

    /* initialize index of radial functions */
    indexr_.init(aw_descriptors_, lo_descriptors_);

    /* initialize index of muffin-tin basis functions */
    indexb_.init(indexr_);

    /* get the number of core electrons */
    num_core_electrons_ = 0;
    if (parameters_.full_potential())
    {
        for (int i = 0; i < (int)atomic_levels_.size(); i++)
        {
            if (atomic_levels_[i].core) num_core_electrons_ += atomic_levels_[i].occupancy;
        }
    }

    /* get number of valence electrons */
    num_valence_electrons_ = zn_ - num_core_electrons_;

    int lmmax_pot = Utils::lmmax(parameters_.lmax_pot());
    auto l_by_lm = Utils::l_by_lm(parameters_.lmax_pot());

    /* index the non-zero radial integrals */
    std::vector< std::pair<int, int> > non_zero_elements;

    for (int lm = 0; lm < lmmax_pot; lm++)
    {
        int l = l_by_lm[lm];

        for (int i2 = 0; i2 < indexr().size(); i2++)
        {
            int l2 = indexr(i2).l;

            for (int i1 = 0; i1 <= i2; i1++)
            {
                int l1 = indexr(i1).l;
                if ((l + l1 + l2) % 2 == 0)
                {
                    if (lm) non_zero_elements.push_back(std::pair<int, int>(i2, lm + lmmax_pot * i1));
                    for (int j = 0; j < parameters_.num_mag_dims(); j++)
                    {
                        int offs = (j + 1) * lmmax_pot * indexr().size();
                        non_zero_elements.push_back(std::pair<int, int>(i2, lm + lmmax_pot * i1 + offs));
                    }
                }
            }
        }
    }

    idx_radial_integrals_ = mdarray<int, 2>(2, non_zero_elements.size());

    for (size_t j = 0; j < non_zero_elements.size(); j++)
    {
        idx_radial_integrals_(0, j) = non_zero_elements[j].first;
        idx_radial_integrals_(1, j) = non_zero_elements[j].second;
    }

    if (parameters_.processing_unit() == GPU && parameters_.full_potential())
    {
        #ifdef __GPU
        idx_radial_integrals_.allocate_on_device();
        idx_radial_integrals_.copy_to_device();
        rf_coef_ = mdarray<double, 3>(nullptr, num_mt_points_, 4, indexr().size());
        rf_coef_.allocate(1);
        rf_coef_.allocate_on_device();
        vrf_coef_ = mdarray<double, 3>(nullptr, num_mt_points_, 4, lmmax_pot * indexr().size() * (parameters_.num_mag_dims() + 1));
        vrf_coef_.allocate(1);
        vrf_coef_.allocate_on_device();
        #else
        TERMINATE_NO_GPU
        #endif
    }

    initialized_ = true;
}





}
