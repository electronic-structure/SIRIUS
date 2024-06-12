#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <tuple>
#include "context/simulation_context.hpp"

namespace sirius {

static std::vector<std::string>
split(std::string str, std::string token)
{
    std::vector<std::string> result;
    while (str.size()) {
        int index = str.find(token);
        if (index != std::string::npos) {
            result.push_back(str.substr(0, index));
            str = str.substr(index + token.size());
            if (str.size() == 0)
                result.push_back(str);
        } else {
            result.push_back(str);
            str = "";
        }
    }
    return result;
}

static auto
parse_atom_string(std::string& at_str__) -> std::tuple<std::string, int, int>
{
    std::vector<std::string> split_str_ = split(at_str__, "_");
    int n_                              = -1;
    int l_                              = -1;

    // a bit pedantic....
    n_ = std::stoi(split_str_[1]);

    switch (split_str_[1][1]) {
        case 's':
        case 'S':
            l_ = 0;
            break;
        case 'p':
        case 'P':
            l_ = 1;
            break;
        case 'd':
        case 'D':
            l_ = 2;
            break;
        case 'f':
        case 'F':
            l_ = 3;
            break;
        case 'g':
        case 'G':
            l_ = 4;
            break;
        case 'h':
        case 'H':
            l_ = 5;
            break;
        case 'I':
        case 'i':
            l_ = 6;
            break;
    }

    return std::make_tuple(std::string{split_str_[0]}, n_, l_);
}

void
add_hubbard_atom_pair(Simulation_context& ctx__, int* const atom_pair__, int* const translation__, int* const n__,
                      int* const l__, const double coupling__)
{
    json elem;
    std::vector<int> atom_pair(atom_pair__, atom_pair__ + 2);
    /* Fortran indices start from 1 */
    atom_pair[0] -= 1;
    atom_pair[1] -= 1;
    std::vector<int> n(n__, n__ + 2);
    std::vector<int> l(l__, l__ + 2);
    std::vector<int> translation(translation__, translation__ + 3);

    elem["atom_pair"] = atom_pair;
    elem["T"]         = translation;
    elem["n"]         = n;
    elem["l"]         = l;
    elem["V"]         = coupling__;

    bool test{false};

    for (int idx = 0; idx < ctx__.cfg().hubbard().nonlocal().size(); idx++) {
        auto v     = ctx__.cfg().hubbard().nonlocal(idx);
        auto at_pr = v.atom_pair();
        /* search if the pair is already present */
        if ((at_pr[0] == atom_pair[0]) && (at_pr[1] == atom_pair[1])) {
            auto tr = v.T();
            if ((tr[0] == translation[0]) && (tr[1] == translation[1]) && (tr[2] == translation[2])) {
                auto lvl = v.n();
                if ((lvl[0] == n[0]) && (lvl[0] == n[1])) {
                    auto li = v.l();
                    if ((li[0] == l[0]) && (li[1] == l[1])) {
                        test = true;
                        break;
                    }
                }
            }
        }
    }

    if (!test) {
        ctx__.cfg().hubbard().nonlocal().append(elem);
    } else {
        RTE_THROW("Atom pair for hubbard correction is already present");
    }
}

void
parse_hubbard_file(Simulation_context& ctx__, const std::string& data_file__)
{
    std::ifstream infile(data_file__);

    if (!infile.is_open()) {
        std::stringstream s;
        s << "The file " << data_file__ << " is unreadable.";
        RTE_THROW(s);
    }

    std::string processed_str;
    // std::vector<std::tuple<int, int, int, int>> construct_neighbors_(ctx__.unit_cell().num_atoms() * 27);

    // for (int atom_id_ = 0 ; atom_id_ < ctx__.unit_cell().num_atoms(); atom_id_++) {
    //   construct_neighbors_[atom_id_] = std::make_tuple<int, int, int, int>(0, 0, 0, atom_id_);
    // }

    // index_ = ctx__.unit_cell().num_atoms();
    // we only need this list to initialize the input dictionary
    // we can get the reverse from the index with this

    // for (int nx = -1; nx < 2; nx++) {
    //   for (int ny = -1; ny < 2; ny++) {
    //     for (int nz = -1; nz < 2; nz++) {
    //       for (int at = 0; at < ctx__.unit_cell().num_atoms(); at++) {
    //         construct_neighbors_[index_] = std::make_tuple<int, int, int, int>(nz,
    //                                                                            ny,
    //                                                                            nx,
    //                                                                            ((3 * (nx + 1) + (ny + 1)) * 3 + nz +
    //                                                                            1) * ctx__.unit_cell().num_atoms() +
    //                                                                            at);
    //         index_++;
    //       }
    //     }
    //   }
    // }

    while (std::getline(infile, processed_str)) {
        std::vector<std::string> split_string = split(processed_str, std::string(" "));

        if ((split_string.size() != 3) || (split_string.size() != 6) || (split_string[0] != "#")) {
            std::stringstream s;
            s << "The file " << data_file__
              << " seems to be corrupted.\nEach line should either be\nU atom_type-orbital real\n or \nV atom_type_1 "
                 "atom_type_2 int int real\n";
            RTE_THROW(s);
        }

        if ((split_string[0] == "U") || (split_string[0] == "u")) {
            const double U_ = std::stod(split_string[2]);
            std::string atom_type_;
            int n_;
            int l_;
            std::tie(atom_type_, n_, l_) = parse_atom_string(split_string[1]);
            json elem;
            elem["atom_type"]               = atom_type_;
            elem["n"]                       = n_;
            elem["l"]                       = l_;
            elem["total_initial_occupancy"] = 0;
            elem["U"]                       = U_;
            ctx__.cfg().hubbard().local().append(elem);
        }

        if (((split_string[0] == "V") || (split_string[0] == "v"))) {
            std::string atom_type1_;
            std::string atom_type2_;
            int n_pair_[2];
            int l_pair_[2];
            std::tie(atom_type1_, n_pair_[0], l_pair_[0]) = parse_atom_string(split_string[1]);
            std::tie(atom_type2_, n_pair_[1], l_pair_[1]) = parse_atom_string(split_string[2]);
            const int atom_index1_                        = std::stoi(split_string[3]);
            const int atom_index2_                        = std::stoi(split_string[4]);
            const double V_                               = std::stoi(split_string[5]);
            int translation1[3]                           = {0, 0, 0};
            int atom_pair_[2];
            int translation2[3] = {0, 0, 0};
            atom_pair_[0]       = atom_index1_ % ctx__.unit_cell().num_atoms();
            translation1[0]     = (atom_index1_ / ctx__.unit_cell().num_atoms()) % 3 - 1;
            translation1[1]     = (atom_index1_ / 3 * ctx__.unit_cell().num_atoms()) % 3 - 1;
            translation1[2]     = (atom_index1_ / 9 * ctx__.unit_cell().num_atoms()) % 3 - 1;

            atom_pair_[1]   = atom_index1_ % ctx__.unit_cell().num_atoms();
            translation2[0] = (atom_index2_ / ctx__.unit_cell().num_atoms()) % 3 - 1;
            translation2[1] = (atom_index2_ / 3 * ctx__.unit_cell().num_atoms()) % 3 - 1;
            translation2[2] = (atom_index2_ / 9 * ctx__.unit_cell().num_atoms()) % 3 - 1;

            translation2[0] -= translation1[0];
            translation2[1] -= translation1[1];
            translation2[2] -= translation1[2];
            add_hubbard_atom_pair(ctx__, atom_pair_, translation2, n_pair_, l_pair_, V_);
        }
    }
    // ignore all the other lines
}
} // namespace sirius
