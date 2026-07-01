#include "sirius.hpp"
#include "testing.hpp"

using namespace sirius;

class BSpline_basis
{
  private:
    int order_;
    std::vector<double> knots_;

    double
    value(int i__, int k__, double x__) const
    {
        if (k__ == 1) {
            return (knots_[i__] <= x__ && x__ < knots_[i__ + 1]) ? 1.0 : 0.0;
        }

        double v{0};

        double d0 = knots_[i__ + k__ - 1] - knots_[i__];
        if (d0 > 0) {
            v += (x__ - knots_[i__]) / d0 * value(i__, k__ - 1, x__);
        }

        double d1 = knots_[i__ + k__] - knots_[i__ + 1];
        if (d1 > 0) {
            v += (knots_[i__ + k__] - x__) / d1 * value(i__ + 1, k__ - 1, x__);
        }

        return v;
    }

    double
    deriv(int i__, int k__, double x__) const
    {
        if (k__ == 1) {
            return 0.0;
        }

        double v{0};

        double d0 = knots_[i__ + k__ - 1] - knots_[i__];
        if (d0 > 0) {
            v += double(k__ - 1) / d0 * value(i__, k__ - 1, x__);
        }

        double d1 = knots_[i__ + k__] - knots_[i__ + 1];
        if (d1 > 0) {
            v -= double(k__ - 1) / d1 * value(i__ + 1, k__ - 1, x__);
        }

        return v;
    }

  public:
    BSpline_basis(int order__, std::vector<double> knots__)
        : order_(order__)
        , knots_(std::move(knots__))
    {
    }

    int
    order() const
    {
        return order_;
    }

    int
    size() const
    {
        return static_cast<int>(knots_.size()) - order_;
    }

    double
    operator()(int i__, double x__) const
    {
        if (x__ == knots_.back()) {
            return (i__ == size() - 1) ? 1.0 : 0.0;
        }

        return value(i__, order_, x__);
    }

    double
    deriv(int i__, double x__) const
    {
        if (x__ == knots_.back()) {
            x__ = std::nextafter(x__, knots_.front());
        }
        return deriv(i__, order_, x__);
    }

    int
    num_knots() const
    {
        return static_cast<int>(knots_.size());
    }

    double
    knot(int i__) const
    {
        return knots_[i__];
    }
};

static std::vector<double>
make_interp_knots(Radial_grid<double> const& grid__, int order__, int step__ = 1)
{
    int num_points = grid__.num_points();
    RTE_ASSERT(num_points >= order__);
    RTE_ASSERT(step__ >= 1);

    int degree = order__ - 1;

    std::vector<double> inner_knots;
    for (int j = 0; j < num_points - order__; j += step__) {
        double x{0};
        for (int m = 1; m <= degree; m++) {
            x += grid__.x(j + m);
        }
        inner_knots.push_back(x / degree);
    }

    int num_basis = static_cast<int>(inner_knots.size()) + order__;

    std::vector<double> knots(num_basis + order__);

    double x0 = grid__.first();
    double x1 = grid__.last();

    for (int i = 0; i < order__; i++) {
        knots[i] = x0;
    }

    for (int i = 0; i < static_cast<int>(inner_knots.size()); i++) {
        knots[order__ + i] = inner_knots[i];
    }

    for (int i = num_basis; i < num_basis + order__; i++) {
        knots[i] = x1;
    }

    return knots;
}

//static std::vector<double>
//make_interp_knots(Radial_grid<double> const& grid__, int order__)
//{
//    int num_points = grid__.num_points();
//    RTE_ASSERT(num_points >= order__);
//
//    int degree = order__ - 1;
//
//    std::vector<double> knots(num_points + order__);
//
//    double x0 = grid__.first();
//    double x1 = grid__.last();
//
//    for (int i = 0; i < order__; i++) {
//        knots[i] = x0;
//    }
//
//    int num_inner = num_points - order__;
//    for (int j = 0; j < num_inner; j++) {
//        double x{0};
//        for (int m = 1; m <= degree; m++) {
//            x += grid__.x(j + m);
//        }
//        knots[order__ + j] = x / degree;
//    }
//
//    for (int i = num_points; i < num_points + order__; i++) {
//        knots[i] = x1;
//    }
//
//    return knots;
//}

static void
gauss_legendre_rule(int n__, std::vector<double>& x__, std::vector<double>& w__)
{
    x__.resize(n__);
    w__.resize(n__);

    constexpr double eps{1e-14};
    int m = (n__ + 1) / 2;

    for (int i = 0; i < m; i++) {
        double z = std::cos(pi * (double(i) + 0.75) / (double(n__) + 0.5));
        double z1;

        double p1{0};
        double p2{0};
        double pp{0};

        do {
            p1 = 1.0;
            p2 = 0.0;

            for (int j = 1; j <= n__; j++) {
                double p3 = p2;
                p2 = p1;
                p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j;
            }

            pp = n__ * (z * p1 - p2) / (z * z - 1.0);

            z1 = z;
            z  = z1 - p1 / pp;
        } while (std::abs(z - z1) > eps);

        x__[i]          = -z;
        x__[n__ - 1 - i] = z;
        w__[i]          = 2.0 / ((1.0 - z * z) * pp * pp);
        w__[n__ - 1 - i] = w__[i];
    }
}


//int
//test_bspline_interp(cmd_args const& args__)
//{
//    int num_points = args__.value<int>("num_points=", 80);
//    int order      = args__.value<int>("order=", 7);
//
//    RTE_ASSERT(num_points >= order);
//
//    double x0{1e-6};
//    double x1{2.0};
//
//    Radial_grid_lin<double> grid(num_points, x0, x1);
//
//    auto f = [](double x) {
//        return std::sin(x) / (x + 1.0);
//    };
//
//    int num_basis = num_points;
//    auto knots    = make_interp_knots(grid, order, 3);
//    BSpline_basis basis(order, knots);
//
//    mdarray<double, 2> A({num_points, num_basis}, "bspline_collocation");
//    mdarray<double, 1> rhs({num_points}, "bspline_rhs");
//
//    for (int i = 0; i < num_points; i++) {
//        double x = grid.x(i);
//        rhs(i) = f(x);
//
//        for (int j = 0; j < num_basis; j++) {
//            A(i, j) = basis(j, x);
//        }
//    }
//
//    int info = la::wrap(la::lib_t::lapack).gesv(num_basis, 1, A.at(memory_t::host), A.ld(), rhs.at(memory_t::host),
//                                                rhs.ld());
//    if (info) {
//        std::cout << "gesv failed, info = " << info << std::endl;
//        return 1;
//    }
//
//    double max_err{0};
//    int num_check = 10 * num_points;
//
//    for (int i = 0; i < num_check; i++) {
//        double x = x0 + (x1 - x0) * double(i) / double(num_check - 1);
//
//        double y{0};
//        for (int j = 0; j < num_basis; j++) {
//            y += rhs(j) * basis(j, x);
//        }
//
//        max_err = std::max(max_err, std::abs(y - f(x)));
//    }
//
//    if (mpi::Communicator::world().rank() == 0) {
//        std::cout << "num_points : " << num_points << std::endl;
//        std::cout << "order      : " << order << std::endl;
//        std::cout << "max error  : " << max_err << std::endl;
//    }
//
//    return max_err < 1e-8 ? 0 : 1;
//}

static int
hydrogen_bspline(cmd_args const& args__)
{
    int num_points = args__.value<int>("num_points", 200);
    int order      = args__.value<int>("order", 7);
    int l          = args__.value<int>("l", 0);
    int nq         = args__.value<int>("nq", 10);
    auto species_file = args__.value<std::string>("species");
    auto pot_file     = args__.value<std::string>("potential");

    Simulation_parameters params;
    params.electronic_structure_method("full_potential_lapwlo");
    params.valence_relativity(args__.value<std::string>("rel", "none"));
    params.verbosity(4);

    std::ifstream ifs(pot_file);
    nlohmann::json dict;
    ifs >> dict;

    auto x    = dict["x"].get<std::vector<double>>();
    auto veff = dict["spherical_potential_el"].get<std::vector<double>>();
    auto z    = dict["z"].get<double>();

    Atom_type atype(params, 0, "X", species_file);
    atype.init();
    atype.set_radial_grid(x.size(), x.data());

    RTE_ASSERT(l >= 0);

    auto knots = make_interp_knots(atype.radial_grid(), order);
    BSpline_basis basis(order, knots);

    std::vector<int> active_basis;
    for (int i = 1; i < basis.size() - 1; i++) {
        active_basis.push_back(i);
    }

    int n = static_cast<int>(active_basis.size());

    mdarray<double, 2> H({n, n}, "radial_hamiltonian");
    mdarray<double, 2> S({n, n}, "radial_overlap");
    H.zero();
    S.zero();

    std::vector<double> xg;
    std::vector<double> wg;
    gauss_legendre_rule(nq, xg, wg);

    Spline<double> veff_s(atype.radial_grid(), veff);

    for (int ik = 0; ik < basis.num_knots() - 1; ik++) {
        double a = basis.knot(ik);
        double b = basis.knot(ik + 1);

        if (b <= a) {
            continue;
        }

        for (int iq = 0; iq < nq; iq++) {
            double r = 0.5 * ((b - a) * xg[iq] + (b + a));
            double w = 0.5 * (b - a) * wg[iq];

            double veff = veff_s(r) - z / r;
            if (l > 0) {
                veff += 0.5 * double(l * (l + 1)) / (r * r);
            }

            for (int ii = 0; ii < n; ii++) {
                int i = active_basis[ii];

                double Bi  = basis(i, r);
                double dBi = basis.deriv(i, r);

                if (Bi == 0.0 && dBi == 0.0) {
                    continue;
                }

                for (int jj = 0; jj < n; jj++) {
                    int j = active_basis[jj];

                    double Bj  = basis(j, r);
                    double dBj = basis.deriv(j, r);

                    if (Bj == 0.0 && dBj == 0.0) {
                        continue;
                    }

                    S(ii, jj) += w * Bi * Bj;
                    H(ii, jj) += w * (0.5 * dBi * dBj + Bi * veff * Bj);
                }
            }
        }
    }

    std::vector<double> eval(n);

    la::dmatrix<double> h(n, n);
    la::dmatrix<double> s(n, n);
    la::dmatrix<double> evec(n, n);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            h.set(i, j, H(i, j));
            s.set(i, j, S(i, j));
        }
    }

    auto solver = la::Eigensolver_factory("lapack");
    int info    = solver->solve(n, n, h, s, eval.data(), evec);

    if (info) {
        std::cout << "eigensolver failed, info = " << info << std::endl;
        return 1;
    }

    if (mpi::Communicator::world().rank() == 0) {
        std::cout << "order      : " << order << std::endl;
        std::cout << "l          : " << l << std::endl;
        std::cout << "z          : " << z << std::endl;

        for (int i = 0; i < std::min(8, n); i++) {
            std::cout << "eval[" << i << "] : " << std::setprecision(16) << eval[i] << std::endl;
        }
    }

    if (mpi::Communicator::world().rank() == 0) {
        int num_states = std::min(10, n);

        nlohmann::json jout;
        jout["l"]          = l;
        jout["order"]      = order;
        jout["num_points"] = atype.num_mt_points();

        jout["eval"] = std::vector<double>(eval.begin(), eval.begin() + num_states);

        for (int ir = 0; ir < atype.num_mt_points(); ir++) {
            double r = atype.radial_grid(ir);
            jout["r"].push_back(r);

            for (int ist = 0; ist < num_states; ist++) {
                double p{0};
                for (int j = 0; j < n; j++) {
                    p += evec(j, ist) * basis(active_basis[j], r);
                }
                jout["p"][ist].push_back(p);
                jout["u"][ist].push_back(p / r);
            }
        }

        std::ofstream ofs("bspline_radial_functions.json", std::ofstream::out | std::ofstream::trunc);
        ofs << jout.dump(4);
    }

    return 0;
}

int
main(int argn, char** argv)
{
    cmd_args args(argn, argv, {{"species=", "(string) species file"},
                               {"potential=", "(string) spherical potential JSON"},
                               {"num_points=", "{int} number of radial grid points"},
                               {"order=", "{int} B-spline order"},
                               {"l=", "{int} angular momentum"},
                               {"nq=", "{int} number of Gauss-Legendre points per knot interval"}});

    sirius::initialize(1);
    //int result = call_test("test_bspline_interp", test_bspline_interp, args);
    int result = call_test("hydrogen_bspline", hydrogen_bspline, args);
    sirius::finalize();

    return result;
}
