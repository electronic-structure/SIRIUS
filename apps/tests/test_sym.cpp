/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>
#include <testing.hpp>
#include "symmetry/crystal_symmetry.hpp"

using namespace sirius;

int
test_sym(cmd_args const& args__)
{
    auto pw_cutoff = args__.value<double>("pw_cutoff", 10);
    auto gk_cutoff = args__.value<double>("gk_cutoff", 3);

    auto json_conf = R"({
      "parameters" : {
        "electronic_structure_method" : "pseudopotential"
      }
    })"_json;

    json_conf["parameters"]["pw_cutoff"] = pw_cutoff;
    json_conf["parameters"]["gk_cutoff"] = gk_cutoff;
    json_conf["parameters"]["num_bands"] = 1;
    // ctx.so_correction(true);

    std::vector<r3::vector<double>> coord;
    coord.push_back(r3::vector<double>({1.0 / 2, 1.0 / 2, 0}));
    coord.push_back(r3::vector<double>({1.0 / 3, 1.0 / 3, 0}));
    // ctx.unit_cell().add_atom("Cu", {0.113500, 0.613500, 0.886500});
    // ctx.unit_cell().add_atom("Cu", {0.613500, 0.886500, 0.113500});
    // ctx.unit_cell().add_atom("Cu", {0.886500, 0.113500, 0.613500});
    // ctx.unit_cell().add_atom("Cu", {0.386500, 0.386500, 0.386500});

    /* create simulation context */
    auto ctx_ptr = sirius::create_simulation_context(json_conf, {{6.0, 0, 0}, {0, 6.0, 0}, {0, 0, 6.0}}, 2, coord,
                                                     false, false);
    Simulation_context& ctx = *ctx_ptr;

    r3::vector<int> k_grid(4, 4, 1);
    r3::vector<int> k_shift(0, 0, 0);

    K_point_set kset_sym(ctx, k_grid, k_shift, true);
    K_point_set kset_nosym(ctx, k_grid, k_shift, false);

    std::vector<int> atoms;
    for (int i = 0; i < ctx.unit_cell().num_atoms(); i++) {
        atoms.push_back(i);
    }

    auto idxb = [&](int iat) { return &ctx.unit_cell().atom_type(iat).indexb_wfs(); };

    auto nawf = ctx.unit_cell().num_ps_atomic_wf();

    int na = ctx.unit_cell().num_atoms();

    std::vector<wf::Wave_functions<double>> phi_sym;

    for (int ik = 0; ik < kset_sym.num_kpoints(); ik++) {
        auto kp = kset_sym.get<double>(ik);
        phi_sym.emplace_back(kp->gkvec_sptr(), wf::num_mag_dims(0), wf::num_bands(nawf.first), memory_t::host);
        kp->generate_atomic_wave_functions(atoms, idxb, *ctx.ri().ps_atomic_wf_, phi_sym.back());
    }

    std::vector<wf::Wave_functions<double>> phi_nosym;

    for (int ik = 0; ik < kset_nosym.num_kpoints(); ik++) {
        auto kp = kset_nosym.get<double>(ik);
        phi_nosym.emplace_back(kp->gkvec_sptr(), wf::num_mag_dims(0), wf::num_bands(nawf.first), memory_t::host);
        kp->generate_atomic_wave_functions(atoms, idxb, *ctx.ri().ps_atomic_wf_, phi_nosym.back());
    }

    auto& sym = ctx.unit_cell().symmetry();

    for (int ik = 0; ik < kset_sym.num_kpoints(); ik++) {
        for (int isym = 0; isym < sym.size(); isym++) {
            auto R    = sym[isym].spg_op.R;
            int pr    = sym[isym].spg_op.proper;
            auto eang = sym[isym].spg_op.euler_angles;

            auto rotm = sht::rotation_matrix<double>(2, eang, pr);

            auto vk1 = r3::reduce_coordinates(dot(R, kset_sym.get<double>(ik)->vk())).first;

            std::cout << "isym: " << isym << " k: " << kset_sym.get<double>(ik)->vk() << " k1: " << vk1 << std::endl;

            /* compute <phi|G+k>w<G+k|phi> using k1 from the irreducible set */
            mdarray<std::complex<double>, 3> dm({5, 5, na});
            dm.zero();

            int ik1 = kset_nosym.find_kpoint(vk1);
            for (int ia = 0; ia < na; ia++) {
                auto& type = ctx.unit_cell().atom(ia).type();
                /* idex of the block of d-orbitals for atom ia */
                int ib = type.indexb_wfs().index_of(rf_index(2)) + nawf.second[ia];
                for (int m1 = 0; m1 < 5; m1++) {
                    for (int m2 = 0; m2 < 5; m2++) {
                        for (int ig = 0; ig < kset_nosym.get<double>(ik1)->num_gkvec(); ig++) {
                            double w = 1.0 / (1.0 + kset_nosym.get<double>(ik1)
                                                            ->gkvec()
                                                            .gkvec_cart(gvec_index_t::global(ig))
                                                            .length());
                            auto z1  = phi_nosym[ik1].pw_coeffs(ig, wf::spin_index(0), wf::band_index(m1 + ib));
                            auto z2  = phi_nosym[ik1].pw_coeffs(ig, wf::spin_index(0), wf::band_index(m2 + ib));
                            dm(m1, m2, ia) += std::conj(z1) * z2 * w;
                        }
                    }
                }
            }

            /* now rotate the coefficients from the initial k-point */
            /* we know <G+k|phi>, we need to find <G+k|P^{-1} phi> */
            wf::Wave_functions<double> phi1(kset_sym.get<double>(ik)->gkvec_sptr(), wf::num_mag_dims(0),
                                            wf::num_bands(nawf.first), memory_t::host);
            for (int ia = 0; ia < na; ia++) {
                int ja = sym[isym].spg_op.sym_atom[ia];

                double phase   = twopi * dot(kset_sym.get<double>(ik)->vk(), ctx.unit_cell().atom(ia).position());
                auto dephase_k = std::exp(std::complex<double>(0.0, phase));

                phase        = twopi * dot(kset_sym.get<double>(ik)->vk(), ctx.unit_cell().atom(ja).position());
                auto phase_k = std::exp(std::complex<double>(0.0, phase));

                std::cout << "ia : " << ia << " -> " << ja << std::endl;

                auto& type_i = ctx.unit_cell().atom(ia).type();
                auto& type_j = ctx.unit_cell().atom(ja).type();
                /* idex of the block of d-orbitals for atom ia and ja*/
                auto ib = type_i.indexb_wfs().index_of(rf_index(2)) + nawf.second[ia];
                auto jb = type_j.indexb_wfs().index_of(rf_index(2)) + nawf.second[ja];

                for (int ig = 0; ig < kset_sym.get<double>(ik)->num_gkvec(); ig++) {
                    mdarray<std::complex<double>, 1> v1({5});
                    v1.zero();
                    for (int m = 0; m < 5; m++) {
                        for (int mp = 0; mp < 5; mp++) {
                            v1[m] += rotm[2](m, mp) *
                                     phi_sym[ik].pw_coeffs(ig, wf::spin_index(0), wf::band_index(ib + mp));
                        }
                    }
                    for (int mp = 0; mp < 5; mp++) {
                        phi1.pw_coeffs(ig, wf::spin_index(0), wf::band_index(jb + mp)) =
                                v1[mp] * dephase_k * std::conj(phase_k);
                    }
                }
            }

            mdarray<std::complex<double>, 3> dm1({5, 5, na});
            dm1.zero();

            for (int ia = 0; ia < na; ia++) {
                auto& type = ctx.unit_cell().atom(ia).type();
                /* idex of the block of d-orbitals for atom ia */
                auto ib = type.indexb_wfs().index_of(rf_index(2)) + nawf.second[ia];
                for (int m1 = 0; m1 < 5; m1++) {
                    for (int m2 = 0; m2 < 5; m2++) {
                        for (int ig = 0; ig < kset_sym.get<double>(ik)->num_gkvec(); ig++) {
                            double w =
                                    1.0 /
                                    (1.0 +
                                     kset_sym.get<double>(ik)->gkvec().gkvec_cart(gvec_index_t::global(ig)).length());
                            auto z1 = phi1.pw_coeffs(ig, wf::spin_index(0), wf::band_index(m1 + ib));
                            auto z2 = phi1.pw_coeffs(ig, wf::spin_index(0), wf::band_index(m2 + ib));
                            dm1(m1, m2, ia) += std::conj(z1) * z2 * w;
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
                s << "max error: " << diff << std::endl << "rotm: " << rotm[2] << std::endl;
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
    return 0;
}

int
main(int argn, char** argv)
{
    cmd_args args(argn, argv,
                  {
                          {"pw_cutoff=", "(double) plane-wave cutoff for density and potential"},
                          {"gk_cutoff=", "(double) plane-wave cutoff for wave-functions"},
                  });

    sirius::initialize(1);
    int result = call_test("test_sym", test_sym, args);
    sirius::finalize();
    return result;
}
