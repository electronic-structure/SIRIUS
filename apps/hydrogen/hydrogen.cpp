/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>

using namespace sirius;

struct level_conf
{
    int n;
    int l;
    int z;

    level_conf(int n__, int l__, int z__)
        : n(n__)
        , l(l__)
        , z(z__)
    {
    }
};

int
main(int argn, char** argv)
{
    cmd_args args(argn, argv,
                  {{"grid_type=", "{int} type of the radial grid"},
                   {"num_points=", "{int} number of grid points"},
                   {"rmin=", "{double} first grid point"},
                   {"p=", "{double} additional grid parameter"}});

    printf("\n");
    printf("Test of radial solver for bare nuclear potential V(r) = -z / r \n");
    printf("Energy of {n,l} states is En = -(1/2) * z^2 / n^2 \n");
    printf("\n");

    sirius::initialize(1);

    int num_points{20000};
    if (args.exist("num_points")) {
        num_points = args.value<int>("num_points");
    }
    radial_grid_t grid_type = static_cast<radial_grid_t>(args.value<int>("grid_type"));

    double rmin{1e-7};
    if (args.exist("rmin")) {
        rmin = args.value<double>("rmin");
    }
    double p{1};
    if (args.exist("p")) {
        p = args.value<double>("p");
    }

    std::vector<level_conf> levels;
    for (int k = 0; k < 10; k++) {
        int z = 1 + k * 10;

        for (int n = 1; n <= 5 + k; n++) {
            for (int l = 0; l <= n - 1; l++) {
                levels.push_back(level_conf(n, l, z));
            }
        }
    }

    //{
    //    auto r = Radial_grid_factory<double>(grid_type, num_points, rmin, 200.0, p);

    //    //Radial_grid r(grid_type, num_points, rmin, 200.0);
    //    std::string fname = "grid_" + r.name() + ".txt";
    //    FILE* fout = fopen(fname.c_str(), "w");
    //    for (int i = 0; i < r.num_points(); i++) fprintf(fout,"%i %16.12e\n", i, r[i]);
    //    fclose(fout);

    //    printf("radial grid: %s\n", r.name().c_str());
    //}

    std::vector<double> err(levels.size());

    #pragma omp parallel for
    for (int j = 0; j < (int)levels.size(); j++) {
        int n = levels[j].n;
        int l = levels[j].l;
        int z = levels[j].z;

        auto radial_grid = Radial_grid_factory<double>(grid_type, num_points, rmin, 200.0 + z * 3.0, p);

        std::vector<double> v(radial_grid.num_points());
        for (int i = 0; i < radial_grid.num_points(); i++) {
            v[i] = -z / radial_grid[i];
        }

        double enu_exact = -0.5 * std::pow(double(z) / n, 2);

        Bound_state bound_state(relativity_t::none, z, n, l, 0, radial_grid, v, enu_exact);

        double enu = bound_state.enu();

        double rel_err = std::abs(1 - enu / enu_exact);

        /* check residual */
        auto& p = bound_state.p();

        auto rg1 = radial_grid.segment(radial_grid.index_of(200));

        Spline<double> s(rg1);
        for (int i = 0; i < rg1.num_points(); i++) {
            double x = rg1[i];
            s(i)     = -0.5 * p.deriv(2, i) + (v[i] + l * (l + 1) / x / x / 2) * p(i) - enu * p(i);
            s(i)     = std::pow(s(i), 2);
        }
        double rtot = s.interpolate().integrate(0);

        #pragma omp critical
        {
            if (rel_err > 1e-10) {
                printf("Fail! ");
            } else {
                printf("OK! ");
            }

            printf("z = %2i n = %2i l = %2i, enu: %12.6e, enu_exact: %12.6e, relative error: %12.6e, residual: "
                   "%12.6e\n",
                   z, n, l, enu, enu_exact, rel_err, rtot);
        }
        err[j] = rel_err;
    }

    FILE* fout = fopen("err.dat", "w");
    for (int j = 0; j < (int)err.size(); j++) {
        fprintf(fout, "%i %20.16f\n", j, err[j]);
    }
    fclose(fout);

    json dict;
    std::vector<int> xaxis;

    int j = 0;
    for (int n = 1; n <= 20; n++) {
        for (int l = 0; l <= n - 1; l++) {
            xaxis.push_back(j);
            j++;
        }
    }
    dict["xaxis"] = xaxis;

    std::vector<int> xaxis_ticks;
    std::vector<std::string> xaxis_tick_labels;

    j = 0;
    for (int n = 1; n <= 20; n++) {
        std::stringstream s;
        s << "n=" << n;
        xaxis_ticks.push_back(j);
        xaxis_tick_labels.push_back(s.str());
        j += n;
    }

    dict["xaxis_ticks"]       = xaxis_ticks;
    dict["xaxis_tick_labels"] = xaxis_tick_labels;
    dict["plot"]              = json::array();

    int i = 0;
    for (int k = 0; k < 10; k++) {
        int z = 1 + k * 10;
        std::stringstream s;
        s << "z=" << z;

        j = 0;
        xaxis.clear();
        for (int n = 1; n <= 5 + k; n++) {
            for (int l = 0; l <= n - 1; l++) {
                xaxis.push_back(j);
                j++;
            }
        }

        std::vector<double> yvalues;

        j = 0;
        for (int n = 1; n <= 5 + k; n++) {
            for (int l = 0; l <= n - 1; l++) {
                yvalues.push_back(err[i++]);
            }
        }

        dict["plot"].push_back(json::object({{"label", s.str()}, {"xaxis", xaxis}, {"yvalues", yvalues}}));
    }

    std::ofstream ofs("out.json", std::ofstream::out | std::ofstream::trunc);
    ofs << dict.dump(4);

    printf("\n");
    printf("Run 'python hydrogen_plot.py out.json' to produce a PDF plot with relative errors.\n");

    return 0;
}
