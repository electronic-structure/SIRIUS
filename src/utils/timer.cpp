#include "utils/timer.hpp"

namespace utils {

time_point_t& utils::timer::global_starting_time()
{
    static bool initialized{false};
    static time_point_t t_;
    /* record the starting time */
    if (!initialized) {
        t_          = std::chrono::high_resolution_clock::now();
        initialized = true;
    }
    return t_;
}

utils::timer::timer(timer&& src__)
{
    this->label_         = src__.label_;
    this->starting_time_ = src__.starting_time_;
    this->active_        = src__.active_;
    src__.active_        = false;
}

double utils::timer::stop()
{
#if defined(__USE_TIMER)
    if (!active_) {
        return 0;
    }

    /* remove this timer name from the list; now last element contains
       the name of the parent timer */
    stack().pop_back();

    /* measure the time difference */
    auto t2    = std::chrono::high_resolution_clock::now();
    auto tdiff = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - starting_time_);
    double val = tdiff.count();

    auto& ts = timer_values(label_);
#ifdef __TIMER_SEQUENCE
    ts.sequence.push_back(starting_time_);
    ts.sequence.push_back(t2);
#endif
    ts.min_val = std::min(ts.min_val, val);
    ts.max_val = std::max(ts.max_val, val);
    ts.tot_val += val;
    ts.count++;

    if (stack().size() != 0) {
        /* last element contains the name of the parent timer */
        auto parent_label = stack().back();
        /* add value to the parent timer */
        if (timer_values_ex().count(parent_label) == 0) {
            timer_values_ex()[parent_label] = std::map<std::string, double>();
        }
        if (timer_values_ex()[parent_label].count(label_) == 0) {
            timer_values_ex()[parent_label][label_] = 0;
        }
        timer_values_ex()[parent_label][label_] += val;
    }
    active_ = false;
    return val;
#else
    return 0;
#endif
}

void utils::timer::print()
{
#if defined(__USE_TIMER)
    for (int i = 0; i < 140; i++) {
        printf("-");
    }
    printf("\n");
    printf("name                                                                 count      total        min        "
           "max    average    self (%%)\n");
    for (int i = 0; i < 140; i++) {
        printf("-");
    }
    printf("\n");
    for (auto& it : timer_values()) {

        double te{0};
        if (timer_values_ex().count(it.first)) {
            for (auto& it2 : timer_values_ex()[it.first]) {
                te += it2.second;
            }
        }
        if (te > it.second.tot_val) {
            printf("wrong timer values: %f %f\n", te, it.second.tot_val);
            throw std::runtime_error("terminating...");
        }
        printf("%-65s : %6i %10.4f %10.4f %10.4f %10.4f     %6.2f\n", it.first.c_str(), it.second.count,
               it.second.tot_val, it.second.min_val, it.second.max_val, it.second.tot_val / it.second.count,
               (it.second.tot_val - te) / it.second.tot_val * 100);
    }
#endif
}

void utils::timer::print_tree()
{
#if defined(__USE_TIMER)
    if (!timer_values().count(main_timer_label)) {
        return;
    }
    for (int i = 0; i < 140; i++) {
        printf("-");
    }
    printf("\n");

    double ttot = timer_values()[main_timer_label].tot_val;

    for (auto& it : timer_values()) {
        if (timer_values_ex().count(it.first)) {
            /* collect external times */
            double te{0};
            for (auto& it2 : timer_values_ex()[it.first]) {
                te += it2.second;
            }
            double f = it.second.tot_val / ttot;
            if (f > 0.01) {
                printf("%s (%10.4fs, %.2f %% of self, %.2f %% of total)\n", it.first.c_str(), it.second.tot_val,
                       (it.second.tot_val - te) / it.second.tot_val * 100, f * 100);

                std::vector<std::pair<double, std::string>> tmp;

                for (auto& it2 : timer_values_ex()[it.first]) {
                    tmp.push_back(std::pair<double, std::string>(it2.second / it.second.tot_val, it2.first));
                }
                std::sort(tmp.rbegin(), tmp.rend());
                for (auto& e : tmp) {
                    printf("|--%s (%10.4fs, %.2f %%) \n", e.second.c_str(), timer_values_ex()[it.first][e.second],
                           e.first * 100);
                }
            }
        }
    }
#endif
}

nlohmann::json utils::timer::serialize()
{
    nlohmann::json dict;
#if defined(__USE_TIMER)
    /* collect local timers */
    for (auto& it : timer::timer_values()) {
        timer_stats_t ts;
        nlohmann::json node;
        node["count"] = it.second.count;
        node["total"] = it.second.tot_val;
        node["min"]   = it.second.min_val;
        node["max"]   = it.second.max_val;
        node["avg"]   = it.second.tot_val / it.second.count;
#ifdef __TIMER_SEQUENCE
        std::vector<double> tseq;
        for (auto& s : it.second.sequence) {
            auto tdiff = std::chrono::duration_cast<std::chrono::duration<double>>(s - global_starting_time());
            tseq.push_back(tdiff.count());
        }
        node["sequence"] = tseq;
#endif
        dict[it.first] = node;
    }
#endif
    return dict;
}

nlohmann::json utils::timer::serialize_tree()
{
    nlohmann::json dict;
#if defined(__USE_TIMER)
    if (!timer_values().count(main_timer_label)) {
        return {};
    }
    /* total execution time */
    double ttot = timer_values()[main_timer_label].tot_val;

    /* loop over the timer; iterator `it` is a <key, valu> pair */
    for (auto& it : timer_values()) {
        /* if this timer is a parent timer for somebody and timer has a non-negligible contribution */
        if (timer_values_ex().count(it.first) && (it.second.tot_val / ttot) > 0.01) {
            /* collect child (external) times */
            double te{0};
            for (auto& it2 : timer_values_ex()[it.first]) {
                te += it2.second;
            }
            nlohmann::json node;

            node["total_time"]              = it.second.tot_val;
            node["child_time"]              = te;
            node["self_time"]               = it.second.tot_val - te;
            node["fraction_of_global_time"] = it.second.tot_val / ttot;
            node["call"]                    = {};

            /* add all children values */
            for (auto& it2 : timer_values_ex()[it.first]) {
                nlohmann::json n;
                n["time"]               = timer_values_ex()[it.first][it2.first];
                n["fraction_of_parent"] = it2.second / it.second.tot_val;
                node["call"][it2.first] = n;
            }
            dict[it.first] = node;
        }
    }
#endif
    return dict;
}

} // namespace utils
