#include <sirius.hpp>

using namespace sirius;

void test_sym(cmd_args const& args__)
{
    auto pw_cutoff = args__.value<double>("pw_cutoff", 10);
    auto gk_cutoff = args__.value<double>("gk_cutoff", 3);

    /* create simulation context */
    Simulation_context ctx(
        "{"
        "   \"parameters\" : {"
        "        \"electronic_structure_method\" : \"pseudopotential\""
        "    },"
        "   \"control\" : {"
        "       \"verification\" : 0"
        "    }"
        "}");

    /* add a new atom type to the unit cell */
    auto& atype = ctx.unit_cell().add_atom_type("Cu");
    /* set pseudo charge */
    atype.zn(11);
    /* set radial grid */
    atype.set_radial_grid(radial_grid_t::lin_exp, 1000, 0.0, 100.0, 6);

    std::vector<double> ps_wf(atype.radial_grid().num_points());
    for (int l = 2; l <= 2; l++) {
        for (int i = 0; i < atype.radial_grid().num_points(); i++) {
            double x = atype.radial_grid(i);
            ps_wf[i] = std::exp(-x) * std::pow(x, l);
        }
        /* add radial function for l */
        atype.add_ps_atomic_wf(3, sirius::experimental::angular_momentum(l), ps_wf);
    }

    /* lattice constant */
    double a{6};
    /* set lattice vectors */
    ctx.unit_cell().set_lattice_vectors({{a, 0, 0},
                                         {0, a, 0},
                                         {0, 0, a}});
    //ctx.unit_cell().add_atom("Cu", {0.113500, 0.613500, 0.886500});
    //ctx.unit_cell().add_atom("Cu", {0.613500, 0.886500, 0.113500});
    //ctx.unit_cell().add_atom("Cu", {0.886500, 0.113500, 0.613500});
    //ctx.unit_cell().add_atom("Cu", {0.386500, 0.386500, 0.386500});
    //
    ctx.unit_cell().add_atom("Cu", {1.0/2, 1.0/2, 0});
    ctx.unit_cell().add_atom("Cu", {1.0/3, 1.0/3, 0});

    /* initialize the context */
    ctx.verbosity(2);
    ctx.pw_cutoff(pw_cutoff);
    ctx.gk_cutoff(gk_cutoff);
    ctx.num_bands(1);
    //ctx.so_correction(true);

    /* initialize simulation context */
    ctx.initialize();

    vector3d<int> k_grid(4, 4, 1);
    vector3d<int> k_shift(0, 0, 0);

    K_point_set kset_sym(ctx, k_grid, k_shift, true);
    K_point_set kset_nosym(ctx, k_grid, k_shift, false);

    int n{0};
    std::vector<int> atoms;
    std::vector<int> offset;
    for (int i = 0; i < ctx.unit_cell().num_atoms(); i++) {
        atoms.push_back(i);
        offset.push_back(n);
        n += 5;
    }

    auto idxb = [&](int iat)
    {
        return &ctx.unit_cell().atom_type(iat).indexb_wfs();
    };

    int na = ctx.unit_cell().num_atoms();
    int nwf = 5 * na;

    std::vector<Wave_functions> phi_sym;

    for (int ik = 0; ik < kset_sym.num_kpoints(); ik++) {
        phi_sym.emplace_back(kset_sym[ik]->gkvec_partition(), nwf, memory_t::host);
        kset_sym[ik]->generate_atomic_wave_functions(atoms, idxb, ctx.atomic_wf_ri(), phi_sym.back());
    }

    std::vector<Wave_functions> phi_nosym;

    for (int ik = 0; ik < kset_nosym.num_kpoints(); ik++) {
        phi_nosym.emplace_back(kset_nosym[ik]->gkvec_partition(), nwf, memory_t::host);
        kset_nosym[ik]->generate_atomic_wave_functions(atoms, idxb, ctx.atomic_wf_ri(), phi_nosym.back());
    }

    auto& sym = ctx.unit_cell().symmetry();

    for (int ik = 0; ik < kset_sym.num_kpoints(); ik++) {
        for (int isym = 0; isym < sym.size(); isym++) {
            auto R    = sym[isym].spg_op.R;
            int  pr   = sym[isym].spg_op.proper;
            auto eang = sym[isym].spg_op.euler_angles;

            auto rotm = sht::rotation_matrix<double>(2, eang, pr);

            auto vk1 = geometry3d::reduce_coordinates(dot(R, kset_sym[ik]->vk())).first;

            std::cout << "isym: " << isym << " k: " << kset_sym[ik]->vk() << " k1: " << vk1 << std::endl;

            /* compute <phi|G+k>w<G+k|phi> using k1 from the irreducible set */
            sddk::mdarray<double_complex, 3> dm(5, 5, na);
            dm.zero();

            int ik1 = kset_nosym.find_kpoint(vk1);
            for (int ia = 0; ia < na; ia++) {
                for (int m1 = 0; m1 < 5; m1++) {
                    for (int m2 = 0; m2 < 5; m2++) {
                        for (int ig = 0; ig < kset_nosym[ik1]->num_gkvec(); ig++) {
                            double w = 1.0 / (1.0 + kset_nosym[ik1]->gkvec().gkvec_cart<index_domain_t::global>(ig).length());
                            dm(m1, m2, ia) += std::conj(phi_nosym[ik1].pw_coeffs(0).prime(ig, m1 + offset[ia])) *
                                phi_nosym[ik1].pw_coeffs(0).prime(ig, m2 + offset[ia]) * w;
                        }
                    }
                }
            }

            /* now rotate the coefficients from the initial k-point */
            /* we know <G+k|phi>, we need to find <G+k|P^{-1} phi> */
            Wave_functions phi1(kset_sym[ik]->gkvec_partition(), nwf, memory_t::host);
            for (int ia = 0; ia < na; ia++) {
                int ja = sym[isym].spg_op.sym_atom[ia];
                int i_a = ia;
                int j_a = ja;

                double phase = twopi * dot(kset_sym[ik]->vk(), ctx.unit_cell().atom(i_a).position());
                auto dephase_k = std::exp(double_complex(0.0, phase));

                phase = twopi * dot(kset_sym[ik]->vk(), ctx.unit_cell().atom(j_a).position());
                auto phase_k = std::exp(double_complex(0.0, phase));

                std::cout << "ia : " << i_a << " -> " << j_a << std::endl;

                for (int ig = 0; ig < kset_sym[ik]->num_gkvec(); ig++) {
                    sddk::mdarray<double_complex, 1> v1(5);
                    v1.zero();
                    for (int m = 0; m < 5; m++) {
                        for (int mp = 0; mp < 5; mp++) {
                            v1[m] += rotm[2](m, mp) * phi_sym[ik].pw_coeffs(0).prime(ig, offset[i_a] + mp);
                        }
                    }
                    for (int mp = 0; mp < 5; mp++) {
                        phi1.pw_coeffs(0).prime(ig, offset[j_a] + mp) = v1[mp] * dephase_k * std::conj(phase_k);
                    }
                }
            }

            sddk::mdarray<double_complex, 3> dm1(5, 5, na);
            dm1.zero();

            for (int ia = 0; ia < na; ia++) {
                for (int m1 = 0; m1 < 5; m1++) {
                    for (int m2 = 0; m2 < 5; m2++) {
                        for (int ig = 0; ig < kset_sym[ik]->num_gkvec(); ig++) {
                            double w = 1.0 / (1.0 + kset_sym[ik]->gkvec().gkvec_cart<index_domain_t::global>(ig).length());
                            dm1(m1, m2, ia) += std::conj(phi1.pw_coeffs(0).prime(ig, m1 + offset[ia])) *
                                phi1.pw_coeffs(0).prime(ig, m2 + offset[ia]) * w;
                        }
                    }
                }
            }
            double diff{0};
            for (int ia = 0; ia < na; ia++) {
                for (int m1 = 0; m1 < 5; m1++) {
                    for (int m2 = 0; m2 < 5; m2++) {
                        diff = std::max(diff, std::abs(dm(m1, m2, ia) - dm1(m1, m2, ia)));
                    }
                }
            }
            if (diff > 1e-12) {
                std::stringstream s;
                s << "max error: " << diff << std::endl
                  << "rotm: " << rotm[2] << std::endl;
                s << "dm using kset_nosym and vk1" << std::endl;
                for (int ia = 0; ia < na; ia++) {
                    s << "ia = " << ia << std::endl;
                    for (int m1 = 0; m1 < 5; m1++) {
                        for (int m2 = 0; m2 < 5; m2++) {
                            if (std::abs(dm(m1, m2, ia)) < 1e-12) {
                                dm(m1, m2, ia) = 0;
                            }
                            s << dm(m1, m2, ia) << " ";
                        }
                        s << std::endl;
                    }
                }
                s << "dm using kset_sym and vk" << std::endl;
                for (int ia = 0; ia < na; ia++) {
                    s << "ia = " << ia << std::endl;
                    for (int m1 = 0; m1 < 5; m1++) {
                        for (int m2 = 0; m2 < 5; m2++) {
                            if (std::abs(dm(m1, m2, ia)) < 1e-12) {
                                dm1(m1, m2, ia) = 0;
                            }
                            s << dm1(m1, m2, ia) << " ";
                        }
                        s << std::endl;
                    }
                }
                RTE_THROW(s);
            }
        }
    }
}

int main(int argn, char** argv)
{
    cmd_args args(argn, argv, {{"pw_cutoff=", "(double) plane-wave cutoff for density and potential"},
                               {"gk_cutoff=", "(double) plane-wave cutoff for wave-functions"},
                              });

    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }

    sirius::initialize(1);
    test_sym(args);
    int rank = Communicator::world().rank();
    sirius::finalize();
    if (rank == 0)  {
        const auto timing_result = ::utils::global_rtgraph_timer.process();
        std::cout<< timing_result.print();
    }
}
