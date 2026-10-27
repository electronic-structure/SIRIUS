/*
 * Copyright (c) 2019 Simon Frasch
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "rt_graph.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <list>
#include <numeric>
#include <ostream>
#include <ratio>
#include <sstream>
#include <string>
#include <tuple>

namespace rt_graph {

// ======================
// internal helper
// ======================
namespace internal {
namespace {

struct StatFormat
{
    StatFormat(Stat stat_)
        : stat(stat_)
    {
        switch (stat_) {
            case Stat::Count:
                header = "#";
                space  = 9;
                break;
            case Stat::Total:
                header = "Total";
                space  = 14;
                break;
            case Stat::Self:
                header = "Self";
                space  = 14;
                break;
            case Stat::Mean:
                header = "Mean";
                space  = 14;
                break;
            case Stat::Median:
                header = "Median";
                space  = 14;
                break;
            case Stat::QuartileHigh:
                header = "Quartile High";
                space  = 14;
                break;
            case Stat::QuartileLow:
                header = "Quartile Low";
                space  = 14;
                break;
            case Stat::Min:
                header = "Min";
                space  = 14;
                break;
            case Stat::Max:
                header = "Max";
                space  = 14;
                break;
            case Stat::Percentage:
                header = "%";
                space  = 11;
                break;
            case Stat::ParentPercentage:
                header = "Parent %";
                space  = 11;
                break;
            case Stat::SelfPercentage:
                header = "Self %";
                space  = 11;
                break;
        }
    }

    Stat stat;
    std::string header;
    std::size_t space;
};

auto
unit_prefix(const double value) -> std::pair<double, char>
{
    if (value == 0.0 || value == -0.0) {
        return {value, '\0'};
    }

    const double exponent = std::log10(std::abs(value));
    const int siExponent  = static_cast<int>(std::floor(exponent / 3.0)) * 3;

    char prefix = '\0';
    switch (siExponent) {
        case 24:
            prefix = 'Y';
            break;
        case 21:
            prefix = 'Z';
            break;
        case 18:
            prefix = 'E';
            break;
        case 15:
            prefix = 'P';
            break;
        case 12:
            prefix = 'T';
            break;
        case 9:
            prefix = 'G';
            break;
        case 6:
            prefix = 'M';
            break;
        case 3:
            prefix = 'k';
            break;
        case 0:
            break;
        case -3:
            prefix = 'm';
            break;
        case -6:
            prefix = 'u';
            break;
        case -9:
            prefix = 'n';
            break;
        case -12:
            prefix = 'p';
            break;
        case -15:
            prefix = 'f';
            break;
        case -18:
            prefix = 'a';
            break;
        case -21:
            prefix = 'z';
            break;
        case -24:
            prefix = 'y';
            break;
        default:
            prefix = '?';
    }

    return {value * std::pow(10.0, static_cast<double>(-siExponent)), prefix};
}

// format time input in seconds into string with appropriate unit
auto
format_time(const double time_seconds) -> std::string
{
    if (time_seconds <= 0.0)
        return std::string("0.00 s ");

    double value;
    char prefix;
    std::tie(value, prefix) = unit_prefix(time_seconds);

    std::stringstream result;
    result << std::fixed << std::setprecision(2);
    result << value;
    result << " ";
    if (prefix)
        result << prefix;
    result << "s";
    if (!prefix)
        result << " ";
    return result.str();
}

auto
calc_median(const std::vector<double>::const_iterator& begin, const std::vector<double>::const_iterator& end) -> double
{
    const auto n = end - begin;
    if (n == 0)
        return 0.0;
    if (n % 2 == 0) {
        return (*(begin + n / 2) + *(begin + n / 2 - 1)) / 2.0;
    } else {
        return *(begin + n / 2);
    }
}

auto
print_stat(std::ostream& out, const StatFormat& format, const std::vector<double>& sortedTimings, double totalSum,
           double parentSum, double currentSum, double subSum) -> void
{
    switch (format.stat) {
        case Stat::Count:
            if (sortedTimings.size() >= 100000) {
                double value;
                char prefix;
                std::tie(value, prefix) = unit_prefix(sortedTimings.size());
                out << std::right << std::setw(format.space) << std::to_string(static_cast<int>(value)) + prefix;
            } else {
                out << std::right << std::setw(format.space) << sortedTimings.size();
            }
            break;
        case Stat::Total:
            out << std::right << std::setw(format.space) << format_time(currentSum);
            break;
        case Stat::Self:
            out << std::right << std::setw(format.space) << format_time(std::max<double>(currentSum - subSum, 0.0));
            break;
        case Stat::Mean:
            out << std::right << std::setw(format.space) << format_time(currentSum / sortedTimings.size());
            break;
        case Stat::Median:
            out << std::right << std::setw(format.space)
                << format_time(calc_median(sortedTimings.begin(), sortedTimings.end()));
            break;
        case Stat::QuartileHigh: {
            const double upperQuartile = calc_median(sortedTimings.begin() + sortedTimings.size() / 2 +
                                                             (sortedTimings.size() % 2) * (sortedTimings.size() > 1),
                                                     sortedTimings.end());
            out << std::right << std::setw(format.space) << format_time(upperQuartile);
        } break;
        case Stat::QuartileLow: {
            const double lowerQuartile =
                    calc_median(sortedTimings.begin(), sortedTimings.begin() + sortedTimings.size() / 2);
            out << std::right << std::setw(format.space) << format_time(lowerQuartile);
        } break;
        case Stat::Min:
            out << std::right << std::setw(format.space) << format_time(sortedTimings.front());
            break;
        case Stat::Max:
            out << std::right << std::setw(format.space) << format_time(sortedTimings.back());
            break;
        case Stat::Percentage: {
            const double p = (totalSum < currentSum || totalSum == 0) ? 100.0 : currentSum / totalSum * 100.0;
            out << std::right << std::fixed << std::setprecision(2) << std::setw(format.space) << p;
        } break;
        case Stat::ParentPercentage: {
            const double p = (parentSum < currentSum || parentSum == 0) ? 100.0 : currentSum / parentSum * 100.0;
            out << std::right << std::fixed << std::setprecision(2) << std::setw(format.space) << p;
        } break;
        case Stat::SelfPercentage: {
            const double p =
                    (currentSum == 0) ? 100.0 : std::max<double>(currentSum - subSum, 0.0) / currentSum * 100.0;
            out << std::right << std::fixed << std::setprecision(2) << std::setw(format.space) << p;
        } break;
    }
}

// Helper struct for creating a tree of timings
struct TimeStampPair
{
    std::string identifier;
    double time                   = 0.0;
    double startTime              = 0.0;
    std::size_t startIdx          = 0;
    std::size_t stopIdx           = 0;
    internal::TimingNode* nodePtr = nullptr;
};

// print nodes recursively
auto
print_node(const std::size_t level, std::ostream& out, const std::vector<internal::StatFormat> formats,
           std::size_t const identifierSpace, std::string nodePrefix, const internal::TimingNode& node,
           bool isLastSubnode, double parentTime, double totalTime) -> void
{
    // print label
    out << std::left;
    std::string identifier = nodePrefix;
    if (level > 0) {
        // std::setw counts the special characters as 1 instead of 3 -> add missing 2
        out << std::setw(identifierSpace + 2);
        if (isLastSubnode)
            identifier += "\u2514 "; // "up and right"
        else
            identifier += "\u251c "; // "vertical and right"
    } else {
        out << std::setw(identifierSpace);
    }
    identifier += node.identifier;
    out << identifier;

    auto sortedTimings = node.timings;
    std::sort(sortedTimings.begin(), sortedTimings.end());

    double subTime = 0.0;
    for (const auto& subNode : node.subNodes) {
        subTime += subNode.totalTime;
    }
    for (const auto& format : formats) {
        print_stat(out, format, sortedTimings, totalTime, parentTime, node.totalTime, subTime);
    }
    out << std::endl;
    for (const auto& subNode : node.subNodes) {
        std::string newNodePrefix      = nodePrefix;
        std::size_t newIdentifierSpace = identifierSpace;
        if (level > 0) {
            if (isLastSubnode) {
                newNodePrefix += "  ";
            } else {
                newNodePrefix += "\u2502 "; // "vertical"
                // std::setw counts the special characters as 1 instead of 3 -> add missing 2
                newIdentifierSpace += 2;
            }
        }
        print_node(level + 1, out, formats, newIdentifierSpace, newNodePrefix, subNode,
                   &subNode == &node.subNodes.back(), node.totalTime, totalTime);
    }
}

// determine length of padding required for printing entire tree identifiers recursively
auto
max_node_identifier_length(const internal::TimingNode& node, const std::size_t recursionDepth,
                           const std::size_t addPerLevel, const std::size_t parentMax) -> std::size_t
{
    std::size_t currentLength = node.identifier.length() + recursionDepth * addPerLevel;
    std::size_t max           = currentLength > parentMax ? currentLength : parentMax;
    for (const auto& subNode : node.subNodes) {
        const std::size_t subMax = max_node_identifier_length(subNode, recursionDepth + 1, addPerLevel, max);
        if (subMax > max)
            max = subMax;
    }

    return max;
}

auto
export_node_json(const std::string& padding, const std::list<internal::TimingNode>& nodeList, std::ostream& stream)
        -> void
{
    stream << "{" << std::endl;
    const std::string nodePadding    = padding + "  ";
    const std::string subNodePadding = nodePadding + "  ";
    for (const auto& node : nodeList) {
        stream << nodePadding << "\"" << node.identifier << "\" : {" << std::endl;
        stream << subNodePadding << "\"timings\" : [";
        for (const auto& value : node.timings) {
            stream << value;
            if (&value != &(node.timings.back()))
                stream << ", ";
        }
        stream << "]," << std::endl;
        stream << subNodePadding << "\"start-times\" : [";
        for (const auto& value : node.startTimes) {
            stream << value;
            if (&value != &(node.startTimes.back()))
                stream << ", ";
        }
        stream << "]," << std::endl;
        stream << subNodePadding << "\"sub-timings\" : ";
        export_node_json(subNodePadding, node.subNodes, stream);
        stream << nodePadding << "}";
        if (&node != &(nodeList.back()))
            stream << ",";
        stream << std::endl;
    }
    stream << padding << "}" << std::endl;
}

auto
extract_timings(const std::string& identifier, const std::list<TimingNode>& nodes, std::vector<double>& timings) -> void
{
    for (const auto& node : nodes) {
        if (node.identifier == identifier) {
            timings.insert(timings.end(), node.timings.begin(), node.timings.end());
        }
        extract_timings(identifier, node.subNodes, timings);
    }
}

auto
sort_timings_nodes(std::list<TimingNode>& nodes) -> void
{
    // sort large to small
    nodes.sort([](const TimingNode& n1, const TimingNode& n2) -> bool { return n1.totalTime > n2.totalTime; });
    for (auto& n : nodes) {
        sort_timings_nodes(n.subNodes);
    }
}

auto
flatten_timings_nodes(std::list<TimingNode>& rootNodes, std::list<TimingNode>& nodes) -> void
{
    // call recursively first, since nodes will be invalidated next
    for (auto& n : nodes) {
        flatten_timings_nodes(rootNodes, n.subNodes);
    }

    for (auto& n : nodes) {
        auto it = std::find_if(rootNodes.begin(), rootNodes.end(),
                               [&n](const TimingNode& element) -> bool { return element.identifier == n.identifier; });
        if (it == rootNodes.end()) {
            // identifier not in rootNodes -> append full TimingNode
            rootNodes.emplace_back(std::move(n));
        } else {
            // identifier already in rootNodes -> only append timings
            it->timings.insert(it->timings.end(), n.timings.begin(), n.timings.end());
            it->startTimes.insert(it->startTimes.end(), n.startTimes.begin(), n.startTimes.end());
            it->totalTime += n.totalTime;
        }
    }

    // drop all nodes
    nodes.clear();
}

auto
flatten_timings_nodes_from_level(std::list<TimingNode>& nodes, std::size_t targetLevel, std::size_t currentLevel)
        -> void
{
    if (targetLevel > currentLevel) {
        for (auto& n : nodes) {
            flatten_timings_nodes_from_level(n.subNodes, targetLevel, currentLevel + 1);
        }
    } else {
        for (auto& n : nodes) {
            flatten_timings_nodes(nodes, n.subNodes);
        }
    }
}

} // namespace
} // namespace internal

// ======================
// Timer
// ======================
auto
Timer::process() const -> TimingResult
{
    std::list<internal::TimingNode> results;
    std::stringstream warnings;

    try {
        std::vector<internal::TimeStampPair> timePairs;
        timePairs.reserve(timeStamps_.size() / 2);

        // create pairs of start / stop timings
        for (std::size_t i = 0; i < timeStamps_.size(); ++i) {
            if (timeStamps_[i].type == internal::TimeStampType::Start) {
                internal::TimeStampPair pair;
                pair.startIdx                           = i;
                pair.identifier                         = std::string(timeStamps_[i].identifierPtr);
                std::size_t numInnerMatchingIdentifiers = 0;
                // search for matching stop after start
                for (std::size_t j = i + 1; j < timeStamps_.size(); ++j) {
                    // only consider matching identifiers
                    if (std::string(timeStamps_[j].identifierPtr) == std::string(timeStamps_[i].identifierPtr)) {
                        if (timeStamps_[j].type == internal::TimeStampType::Stop && numInnerMatchingIdentifiers == 0) {
                            // Matching stop found
                            std::chrono::duration<double> duration = timeStamps_[j].time - timeStamps_[i].time;
                            pair.time                              = duration.count();
                            duration                               = timeStamps_[i].time - timeStamps_[0].time;
                            pair.startTime                         = duration.count();
                            pair.stopIdx                           = j;
                            timePairs.push_back(pair);
                            if (pair.time < 0 || pair.startTime < 0) {
                                warnings << "rt_graph WARNING:Measured time is negative. Non-steady system-clock?!"
                                         << std::endl;
                            }
                            break;
                        } else if (timeStamps_[j].type == internal::TimeStampType::Stop &&
                                   numInnerMatchingIdentifiers > 0) {
                            // inner stop with matching identifier
                            --numInnerMatchingIdentifiers;
                        } else if (timeStamps_[j].type == internal::TimeStampType::Start) {
                            // inner start with matching identifier
                            ++numInnerMatchingIdentifiers;
                        }
                    }
                }
                if (pair.stopIdx == 0) {
                    warnings << "rt_graph WARNING: Start / stop time stamps do not match for \""
                             << timeStamps_[i].identifierPtr << "\"!" << std::endl;
                }
            }
        }

        // create tree of timings where sub-nodes represent timings fully enclosed by another start /
        // stop pair. Use the fact that timePairs is sorted by startIdx
        for (std::size_t i = 0; i < timePairs.size(); ++i) {
            auto& pair = timePairs[i];

            // find potential parent by going backwards through pairs, starting with the current pair
            // position
            for (auto timePairIt = timePairs.rbegin() + (timePairs.size() - i); timePairIt != timePairs.rend();
                 ++timePairIt) {
                if (timePairIt->stopIdx > pair.stopIdx && timePairIt->nodePtr != nullptr) {
                    auto& parentNode = *(timePairIt->nodePtr);
                    // check if sub-node with identifier exists
                    bool nodeFound = false;
                    for (auto& subNode : parentNode.subNodes) {
                        if (subNode.identifier == pair.identifier) {
                            nodeFound = true;
                            subNode.add_time(pair.startTime, pair.time);
                            // mark node position in pair for finding sub-nodes
                            pair.nodePtr = &(subNode);
                            break;
                        }
                    }
                    if (!nodeFound) {
                        // create new sub-node
                        internal::TimingNode newNode;
                        newNode.identifier = pair.identifier;
                        newNode.add_time(pair.startTime, pair.time);
                        parentNode.subNodes.push_back(std::move(newNode));
                        // mark node position in pair for finding sub-nodes
                        pair.nodePtr = &(parentNode.subNodes.back());
                    }
                    break;
                }
            }

            // No parent found, must be top level node
            if (pair.nodePtr == nullptr) {
                // Check if top level node with same name exists
                for (auto& topNode : results) {
                    if (topNode.identifier == pair.identifier) {
                        topNode.add_time(pair.startTime, pair.time);
                        pair.nodePtr = &(topNode);
                        break;
                    }
                }
            }

            // New top level node
            if (pair.nodePtr == nullptr) {
                internal::TimingNode newNode;
                newNode.identifier = pair.identifier;
                newNode.add_time(pair.startTime, pair.time);
                // newNode.parent = nullptr;
                results.push_back(std::move(newNode));

                // mark node position in pair for finding sub-nodes
                pair.nodePtr = &(results.back());
            }
        }
    } catch (const std::exception& e) {
        warnings << "rt_graph WARNING: Processing of timings failed: " << e.what() << std::endl;
    } catch (...) {
        warnings << "rt_graph WARNING: Processing of timings failed!" << std::endl;
    }

    return TimingResult(std::move(results), warnings.str());
}

auto
TimingResult::json() const -> std::string
{
    std::stringstream jsonStream;
    jsonStream << std::scientific;
    internal::export_node_json("", rootNodes_, jsonStream);
    return jsonStream.str();
}

auto
TimingResult::get_timings(const std::string& identifier) const -> std::vector<double>
{
    std::vector<double> timings;
    internal::extract_timings(identifier, rootNodes_, timings);
    return timings;
}

auto
TimingResult::print(std::vector<Stat> statistic) const -> std::string
{
    std::stringstream stream;

    // print warnings
    stream << warnings_;

    // calculate space for printing identifiers
    std::size_t identifierSpace = 0;
    for (const auto& node : rootNodes_) {
        const auto nodeMax = internal::max_node_identifier_length(node, 0, 2, identifierSpace);
        if (nodeMax > identifierSpace)
            identifierSpace = nodeMax;
    }

    auto totalSpace = identifierSpace;

    std::vector<internal::StatFormat> formats;
    formats.reserve(statistic.size());
    for (const auto& stat : statistic) {
        formats.emplace_back(stat);
        totalSpace += formats.back().space;
    }

    // Construct table header

    // Table start
    stream << std::string(totalSpace, '=') << std::endl;

    // header
    stream << std::right << std::setw(identifierSpace) << "";
    for (const auto& format : formats) {
        stream << std::right << std::setw(format.space) << format.header;
    }
    stream << std::endl;

    // Header separtion line
    stream << std::string(totalSpace, '-') << std::endl;

    // print all timings
    for (const auto& node : rootNodes_) {
        internal::print_node(0, stream, formats, identifierSpace, "", node, true, node.totalTime, node.totalTime);
        stream << std::endl;
    }

    // End table
    stream << std::string(totalSpace, '=') << std::endl;

    return stream.str();
}

auto
TimingResult::flatten(std::size_t level) -> TimingResult&
{
    internal::flatten_timings_nodes_from_level(rootNodes_, level, 0);
    return *this;
}

auto
TimingResult::sort_nodes() -> TimingResult&
{
    internal::sort_timings_nodes(rootNodes_);
    return *this;
}

} // namespace rt_graph
