#include "reciprocal_lattice.h"

namespace sirius {
        
Reciprocal_lattice::Reciprocal_lattice(Unit_cell* unit_cell__, electronic_structure_method_t esm_type__, double pw_cutoff__, 
                                       double gk_cutoff__, int lmax__) 
    : unit_cell_(unit_cell__), 
      esm_type_(esm_type__),
      pw_cutoff_(pw_cutoff__), 
      gk_cutoff_(gk_cutoff__),
      num_gvec_(0),
      num_gvec_coarse_(0)
{
    for (int l = 0; l < 3; l++)
    {
        for (int x = 0; x < 3; x++)
        {
            lattice_vectors_[l][x] = unit_cell_->lattice_vectors(l, x);
            reciprocal_lattice_vectors_[l][x] = unit_cell_->reciprocal_lattice_vectors(l, x);
        }
    }

    vector3d<int> max_frac_coord = Utils::find_translation_limits(pw_cutoff_, reciprocal_lattice_vectors_);
    fft_ = new FFT3D<cpu>(max_frac_coord);
    
    if (esm_type_ == ultrasoft_pseudopotential)
    {
        vector3d<int> max_frac_coord_coarse = Utils::find_translation_limits(gk_cutoff__ * 2, 
                                                                             reciprocal_lattice_vectors_);
        fft_coarse_ = new FFT3D<cpu>(max_frac_coord_coarse);
    }

    init(lmax__);
}

Reciprocal_lattice::~Reciprocal_lattice()
{
    delete fft_;
    if (esm_type_ == ultrasoft_pseudopotential) delete fft_coarse_;
}

void Reciprocal_lattice::init(int lmax)
{
    Timer t("sirius::Reciprocal_lattice::init");
    
    mdarray<int, 2> gvec_tmp(3, fft_->size());
    std::vector< std::pair<double, int> > gvec_tmp_length;
    
    int ig = 0;
    for (int i0 = fft_->grid_limits(0).first; i0 <= fft_->grid_limits(0).second; i0++)
    {
        for (int i1 = fft_->grid_limits(1).first; i1 <= fft_->grid_limits(1).second; i1++)
        {
            for (int i2 = fft_->grid_limits(2).first; i2 <= fft_->grid_limits(2).second; i2++)
            {
                gvec_tmp(0, ig) = i0;
                gvec_tmp(1, ig) = i1;
                gvec_tmp(2, ig) = i2;
                
                vector3d<double> vc = get_cartesian_coordinates(vector3d<int>(i0, i1, i2));

                gvec_tmp_length.push_back(std::pair<double, int>(vc.length(), ig++));
            }
        }
    }

    Timer t1("sirius::Reciprocal_lattice::init|sort_G");
    // sort G-vectors by length
    std::sort(gvec_tmp_length.begin(), gvec_tmp_length.end());
    t1.stop();

    // create sorted list of G-vectors
    gvec_.set_dimensions(3, fft_->size());
    gvec_.allocate();

    // find number of G-vectors within the cutoff
    num_gvec_ = 0;
    for (int i = 0; i < fft_->size(); i++)
    {
        for (int x = 0; x < 3; x++) gvec_(x, i) = gvec_tmp(x, gvec_tmp_length[i].second);
        
        if (gvec_tmp_length[i].first <= pw_cutoff_) num_gvec_++;
    }
    
    index_by_gvec_.set_dimensions(dimension(fft_->grid_limits(0).first, fft_->grid_limits(0).second),
                                  dimension(fft_->grid_limits(1).first, fft_->grid_limits(1).second),
                                  dimension(fft_->grid_limits(2).first, fft_->grid_limits(2).second));
    index_by_gvec_.allocate();
    
    fft_index_.resize(fft_->size());
    
    gvec_shell_.resize(fft_->size());
    gvec_shell_len_.clear();
    
    for (int ig = 0; ig < fft_->size(); ig++)
    {
        int i0 = gvec_(0, ig);
        int i1 = gvec_(1, ig);
        int i2 = gvec_(2, ig);

        // mapping from G-vector to it's index
        index_by_gvec_(i0, i1, i2) = ig;

        // mapping of FFT buffer linear index
        fft_index_[ig] = fft_->index(i0, i1, i2);

        // find G-shells
        double t = gvec_tmp_length[ig].first;
        if (gvec_shell_len_.empty() || fabs(t - gvec_shell_len_.back()) > 1e-10) gvec_shell_len_.push_back(t);
        gvec_shell_[ig] = (int)gvec_shell_len_.size() - 1;
    }

    // create split index
    spl_num_gvec_ = splindex<block>(num_gvec(), Platform::num_mpi_ranks(), Platform::mpi_rank());
    
    if (lmax >= 0)
    {
        // precompute spherical harmonics of G-vectors 
        gvec_ylm_.set_dimensions(Utils::lmmax(lmax), spl_num_gvec_.local_size());
        gvec_ylm_.allocate();
        
        Timer t2("sirius::Reciprocal_lattice::init|ylm_G");
        for (int igloc = 0; igloc < spl_num_gvec_.local_size(); igloc++)
        {
            int ig = spl_num_gvec_[igloc];
            double rtp[3];
            SHT::spherical_coordinates(gvec_cart(ig), rtp);
            SHT::spherical_harmonics(lmax, rtp[1], rtp[2], &gvec_ylm_(0, igloc));
        }
        t2.stop();
    }
    
    if (esm_type_ == ultrasoft_pseudopotential)
    {
        int nbeta = unit_cell_->max_mt_radial_basis_size();

        mdarray<double, 4> q_radial_functions(unit_cell_->max_num_mt_points(), lmax + 1, nbeta * (nbeta + 1) / 2, 
                                              unit_cell_->num_atom_types());

        fix_q_radial_functions(q_radial_functions);

        // TODO: in principle, this can be sistributed over G-shells (each mpi rank holds radial integrals only for
        //       G-shells of local fraction of G-vectors
        mdarray<double, 4> q_radial_integrals(nbeta * (nbeta + 1) / 2, lmax + 1, unit_cell_->num_atom_types(), 
                                              num_gvec_shells_inner());

        generate_q_radial_integrals(lmax, q_radial_functions, q_radial_integrals);

        generate_q_pw(lmax, q_radial_integrals);
        
        // get the number of G-vectors within the cutoff in the coarse grid
        num_gvec_coarse_ = 0;
        fft_index_coarse_.clear();
        gvec_index_.clear();
        for (int i0 = fft_coarse_->grid_limits(0).first; i0 <= fft_coarse_->grid_limits(0).second; i0++)
        {
            for (int i1 = fft_coarse_->grid_limits(1).first; i1 <= fft_coarse_->grid_limits(1).second; i1++)
            {
                for (int i2 = fft_coarse_->grid_limits(2).first; i2 <= fft_coarse_->grid_limits(2).second; i2++)
                {
                    vector3d<double> vc = get_cartesian_coordinates(vector3d<int>(i0, i1, i2));

                    if (vc.length() <= 2 * gk_cutoff_) 
                    {
                        // linear index inside coarse FFT buffer
                        fft_index_coarse_.push_back(fft_coarse_->index(i0, i1, i2));
                        
                        // corresponding G-vector index in the fine mesh
                        gvec_index_.push_back(index_by_gvec(i0, i1, i2));

                        num_gvec_coarse_++;
                    }
                }
            }
        }
    }

    update();
}

void Reciprocal_lattice::update()
{
    Timer t2("sirius::Reciprocal_lattice::update");
    // precompute G-vector phase factors
    gvec_phase_factors_.set_dimensions(spl_num_gvec_.local_size(), unit_cell_->num_atoms());
    gvec_phase_factors_.allocate();
    #pragma omp parallel for
    for (int igloc = 0; igloc < spl_num_gvec_.local_size(); igloc++)
    {
        int ig = spl_num_gvec_[igloc];
        for (int ia = 0; ia < unit_cell_->num_atoms(); ia++) gvec_phase_factors_(igloc, ia) = gvec_phase_factor<global>(ig, ia);
    }
}

void Reciprocal_lattice::print_info()
{
    printf("\n");
    printf("plane wave cutoff : %f\n", pw_cutoff_);
    printf("number of G-vectors within the cutoff : %i\n", num_gvec());
    printf("number of G-shells : %i\n", num_gvec_shells_inner());
    printf("FFT grid size : %i %i %i   total : %i\n", fft_->size(0), fft_->size(1), fft_->size(2), fft_->size());
    printf("FFT grid limits : %i %i   %i %i   %i %i\n", fft_->grid_limits(0).first, fft_->grid_limits(0).second,
                                                        fft_->grid_limits(1).first, fft_->grid_limits(1).second,
                                                        fft_->grid_limits(2).first, fft_->grid_limits(2).second);
    
    if (esm_type_ == ultrasoft_pseudopotential)
    {
        printf("number of G-vectors on the coarse grid within the cutoff : %i\n", num_gvec_coarse());
        printf("FFT coarse grid size : %i %i %i   total : %i\n", fft_coarse_->size(0), fft_coarse_->size(1), fft_coarse_->size(2), fft_coarse_->size());
        printf("FFT coarse grid limits : %i %i   %i %i   %i %i\n", fft_coarse_->grid_limits(0).first, fft_coarse_->grid_limits(0).second,
                                                                   fft_coarse_->grid_limits(1).first, fft_coarse_->grid_limits(1).second,
                                                                   fft_coarse_->grid_limits(2).first, fft_coarse_->grid_limits(2).second);
    }
}

std::vector<double_complex> Reciprocal_lattice::make_periodic_function(mdarray<double, 2>& form_factors, int ngv)
{
    assert((int)form_factors.size(0) == unit_cell_->num_atom_types());
    
    std::vector<double_complex> f_pw(ngv, double_complex(0, 0));

    double fourpi_omega = fourpi / unit_cell_->omega();

    splindex<block> spl_ngv(ngv, Platform::num_mpi_ranks(), Platform::mpi_rank());

    #pragma omp parallel
    for (auto it = splindex_iterator<block>(spl_ngv); it.valid(); it++)
    {
        int ig = it.idx();
        int igs = gvec_shell(ig);

        for (int ia = 0; ia < unit_cell_->num_atoms(); ia++)
        {            
            int iat = unit_cell_->atom(ia)->type_id();
            f_pw[ig] += fourpi_omega * conj(gvec_phase_factor<global>(ig, ia)) * form_factors(iat, igs);
        }
    }

    Platform::allgather(&f_pw[0], spl_ngv.global_offset(), spl_ngv.local_size());

    return f_pw;
}


void Reciprocal_lattice::fix_q_radial_functions(mdarray<double, 4>& qrf)
{
    Timer t("sirius::Reciprocal_lattice::fix_q_radial_functions");

    for (int iat = 0; iat < unit_cell_->num_atom_types(); iat++)
    {
        auto atom_type = unit_cell_->atom_type(iat);
        for (int l3 = 0; l3 <= 2 * atom_type->indexr().lmax(); l3++)
        {
            for (int idxrf2 = 0; idxrf2 < atom_type->mt_radial_basis_size(); idxrf2++)
            {
                for (int idxrf1 = 0; idxrf1 <= idxrf2; idxrf1++)
                {
                    int idx = idxrf2 * (idxrf2 + 1) / 2 + idxrf1;
                    memcpy(&qrf(0, l3, idx, iat), &atom_type->uspp().q_radial_functions(0, idx), 
                           atom_type->num_mt_points() * sizeof(double));
                    atom_type->fix_q_radial_function(l3, idxrf1, idxrf2, &qrf(0, l3, idx, iat));
                }
            }
        }
    }
}

void Reciprocal_lattice::generate_q_radial_integrals(int lmax, mdarray<double, 4>& qrf, mdarray<double, 4>& qri)
{
    Timer t("sirius::Reciprocal_lattice::generate_q_radial_integrals");

    qri.zero();
    
    splindex<block> spl_num_gvec_shells(num_gvec_shells_inner(), Platform::num_mpi_ranks(), Platform::mpi_rank());
    
    #pragma omp parallel
    {
        sbessel_pw<double> jl(unit_cell_, lmax);
        for (auto it = splindex_iterator<block>(spl_num_gvec_shells); it.valid(); it++)
        {
            int igs = it.idx();
            jl.load(gvec_shell_len(igs));

            for (int iat = 0; iat < unit_cell_->num_atom_types(); iat++)
            {
                auto atom_type = unit_cell_->atom_type(iat);
                Spline<double> s(atom_type->num_mt_points(), atom_type->radial_grid());

                for (int l3 = 0; l3 <= 2 * atom_type->indexr().lmax(); l3++)
                {
                    for (int idxrf2 = 0; idxrf2 < atom_type->mt_radial_basis_size(); idxrf2++)
                    {
                        int l2 = atom_type->indexr(idxrf2).l;
                        for (int idxrf1 = 0; idxrf1 <= idxrf2; idxrf1++)
                        {
                            int l1 = atom_type->indexr(idxrf1).l;

                            int idx = idxrf2 * (idxrf2 + 1) / 2 + idxrf1;
                            
                            if (l3 >= abs(l1 - l2) && l3 <= (l1 + l2) && (l1 + l2 + l3) % 2 == 0)
                            {
                                for (int ir = 0; ir < atom_type->num_mt_points(); ir++)
                                    s[ir] = jl(ir, l3, iat) * qrf(ir, l3, idx, iat);

                                qri(idx, l3, iat, igs) = s.interpolate().integrate(0);
                            }
                        }
                    }
                }
            }
        }
    }
    int ld = (int)(qri.size(0) * qri.size(1) * qri.size(2));
    Platform::allgather(&qri(0, 0, 0, 0), ld * spl_num_gvec_shells.global_offset(), ld * spl_num_gvec_shells.local_size());
}

void Reciprocal_lattice::generate_q_pw(int lmax, mdarray<double, 4>& qri)
{
    Timer t("sirius::Reciprocal_lattice::generate_q_pw");

    double fourpi_omega = fourpi / unit_cell_->omega();
    
    std::vector<int> l_by_lm = Utils::l_by_lm(lmax);

    std::vector<double_complex> zilm(Utils::lmmax(lmax));
    for (int l = 0, lm = 0; l <= lmax; l++)
    {
        for (int m = -l; m <= l; m++, lm++) zilm[lm] = pow(double_complex(0, 1), l);
    }

    for (int iat = 0; iat < unit_cell_->num_atom_types(); iat++)
    {
        auto atom_type = unit_cell_->atom_type(iat);
        int nbf = atom_type->mt_basis_size();
        int lmax_beta = atom_type->indexr().lmax();
        int lmmax = Utils::lmmax(lmax_beta * 2);
        Gaunt_coefficients<double> gaunt_coefs(lmax_beta, 2 * lmax_beta, lmax_beta);

        atom_type->uspp().q_mtrx.zero();
        
        atom_type->uspp().q_pw.set_dimensions(spl_num_gvec_.local_size(), nbf * (nbf + 1) / 2);
        atom_type->uspp().q_pw.allocate();

        for (int xi2 = 0; xi2 < nbf; xi2++)
        {
            int lm2 = atom_type->indexb(xi2).lm;
            int idxrf2 = atom_type->indexb(xi2).idxrf;

            for (int xi1 = 0; xi1 <= xi2; xi1++)
            {
                int lm1 = atom_type->indexb(xi1).lm;
                int idxrf1 = atom_type->indexb(xi1).idxrf;

                int idx12 = xi2 * (xi2 + 1) / 2 + xi1;
                int idxrf12 = idxrf2 * (idxrf2 + 1) / 2 + idxrf1;
                
                #pragma omp parallel
                {
                    std::vector<double_complex> v(lmmax);
                    for (auto it = splindex_iterator<block>(spl_num_gvec_); it.valid(); it++)
                    {
                        int igs = gvec_shell(it.idx());
                        int igloc = it.idx_local();
                        for (int lm3 = 0; lm3 < lmmax; lm3++)
                        {
                            v[lm3] = conj(zilm[lm3]) * gvec_ylm(lm3, igloc) * qri(idxrf12, l_by_lm[lm3], iat, igs);
                        }

                        atom_type->uspp().q_pw(igloc, idx12) = fourpi_omega * gaunt_coefs.sum_L3_gaunt(lm2, lm1, &v[0]);

                        if (igs == 0)
                        {
                            atom_type->uspp().q_mtrx(xi1, xi2) = unit_cell_->omega() * atom_type->uspp().q_pw(0, idx12);
                            atom_type->uspp().q_mtrx(xi2, xi1) = conj(atom_type->uspp().q_mtrx(xi1, xi2));
                        }
                    }
                }
            }
        }
        Platform::bcast(&atom_type->uspp().q_mtrx(0, 0), (int)atom_type->uspp().q_mtrx.size(), 0);
    }
}

}
